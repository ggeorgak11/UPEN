import torch
import torch.nn as nn
import numpy as np
import argparse
import habitat
import habitat_sim
import gzip
import json
from habitat.datasets.utils import get_action_shortest_path
import random
import os
import datasets.util.viz_utils as viz_utils
import test_utils as tutils
import datasets.util.utils as utils
try:
    from habitat_sim.errors import GreedyFollowerError
except ImportError:
    GreedyFollower = BaseException


class Params(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--split', type=str, dest='split', default='val',
                                 choices=['train', 'val', 'test'])

        self.parser.add_argument('--turn_angle', type=int, dest='turn_angle', default=10)
        self.parser.add_argument('--forward_step_size', type=float, dest='forward_step_size', default=0.25)
        
        self.parser.add_argument('--num_episodes', type=int, dest='num_episodes', default=30,
                                 help='number of episodes to generate for each object for each scene')
        self.parser.add_argument('--shortest_path_success_distance', type=float, dest='shortest_path_success_distance', default=0.1,
                                 help='success distance during shortest path estimation')
        self.parser.add_argument('--verification_success_distance', type=float, dest='verification_success_distance', default=0.2, # 1.0
                                 help='success distance during episode verification')
        
        self.parser.add_argument('--geodesic_to_euclid_min_ratio', type=float, dest='geodesic_to_euclid_min_ratio', default=2.5,
                                help='ratio to ensure episode complexity')

        self.parser.add_argument('--shortest_path_max_steps', type=int, dest='shortest_path_max_steps', default=500,
                                 help='after how many steps the pathfinder gives up')
        
        self.parser.add_argument('--closest_dist_limit', type=float, dest='closest_dist_limit', default=4,
                                 help='distance to goal must be larger than this value')
        self.parser.add_argument('--max_dist_limit', type=float, dest='max_dist_limit', default=13,
                                 help='distance to goal must be smaller than this value')
        self.parser.add_argument('--island_radius_limit', type=float, dest='island_radius_limit', default=1.5,
                                 help='minimum distance away from obstacles when samping points')

        self.parser.add_argument('--episode_max_steps', type=int, dest='episode_max_steps', default=90,
                                 help='episodes shortest path max number of steps, for practical reasons')
        self.parser.add_argument('--episode_min_steps', type=int, dest='episode_min_steps', default=11,
                                 help='reject generated episode with less number of steps')

        self.parser.add_argument('--config_file', type=str, dest='config_file',
                                default='configs/my_pointnav_mp3d_val.yaml',
                                help='path to habitat dataset config file')

        self.parser.add_argument('--root_path', type=str, dest='root_path', default="/")
        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='habitat-api/data/scene_datasets/')
        self.parser.add_argument('--episodes_save_dir', type=str, dest='episodes_save_dir', default="habitat-api/data/datasets/pointnav/mp3d/v2/")



# verify the episode by running the simulator over the estimated path and checking the metrics
def verify_episode(sim,
                   start_position,
                   start_rotation,
                   shortest_path,
                   goal_pos,
                   start_end_episode_distance,
                   success_distance,
                   scene_id,
                   episode_counter,
                   save_path_images=False):

    sim.reset()
    sim.set_agent_state(start_position, start_rotation)

    sim_obs = sim.get_sensor_observations()
    observations = sim._sensor_suite.get_observations(sim_obs)

    if save_path_images:
        save_dir = "episode_examples/"+scene_id+"/"+str(episode_counter)+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    agent_episode_distance = 0.0 # distance covered by agent at any given time in the episode
    stop = False
    previous_pos = sim.get_agent_state().position

    for k in range(len(shortest_path)):

        if save_path_images:
            img = observations['rgb'][:,:,:3]
            depth = observations['depth'].reshape(256, 256, 1)
            semantic = observations['semantic']
            viz_utils.display_sample(img, np.squeeze(depth), semantic, savepath=save_dir+str(k)+".png")

        observations = None

        action_id = shortest_path[k]
        if action_id==None: # assume stop action is chosen
            stop = True
            break

        observations = sim.step(action_id)

        # estimate distance covered by agent
        current_pos = sim.get_agent_state().position
        agent_episode_distance += utils.euclidean_distance(current_pos, previous_pos)
        previous_pos = current_pos

    metrics = tutils.get_metrics(sim, 
                            goal_pos, 
                            success_distance, 
                            start_end_episode_distance,
                            agent_episode_distance, 
                            stop)

    return metrics['success']


# Taken from: https://github.com/facebookresearch/habitat-lab/blob/master/habitat/datasets/pointnav/pointnav_generator.py
def _ratio_sample_rate(ratio: float, ratio_threshold: float):
    r"""Sampling function for aggressive filtering of straight-line
    episodes with shortest path geodesic distance to Euclid distance ratio
    threshold.
    :param ratio: geodesic distance ratio to Euclid distance
    :param ratio_threshold: geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: value between 0.008 and 0.144 for ratio [1, 1.1]
    """
    assert ratio < ratio_threshold
    return 20 * (ratio - 0.98) ** 2


def generate_pointnav_episodes(sim, 
                             scene_id,
                             num_episodes=50,
                             shortest_path_success_distance=0.1, 
                             shortest_path_max_steps=500,
                             closest_dist_limit=1,
                             max_dist_limit=20,
                             island_radius_limit=1.5,
                             episode_max_steps=70,
                             episode_min_steps=11,
                             verification_success_distance=1.0,
                             geodesic_to_euclid_min_ratio=1.1):

    episodes = []
    episode_counter=0

    while episode_counter < num_episodes:

        init_pos = sim.sample_navigable_point()
        goal_pos = sim.sample_navigable_point()

        geo_dist = sim.geodesic_distance(init_pos, goal_pos)

        if geo_dist == np.inf: # unlikely case
            continue

        if np.abs(init_pos[1] - goal_pos[1]) > 0.5:
            continue # check height diff to ensure init position and goal pos are on same floor
        
        if geo_dist <= closest_dist_limit:
            continue
        
        if geo_dist >= max_dist_limit:
            continue

        if sim.island_radius(goal_pos) < island_radius_limit:
            continue

        euclid_dist = utils.euclidean_distance(np.asarray(init_pos), np.asarray(goal_pos))
        
        # A good measure of the navigation complexity of an episode is the ratio of
        # geodesic shortest path position to Euclidean distance between start and
        # goal positions to the corresponding Euclidean distance.
        # If the ratio is nearly 1, it indicates there are few obstacles, and the
        # episode is easy; if the ratio is larger than 1, the
        # episode is difficult because strategic navigation is required.
        distances_ratio = geo_dist / euclid_dist
        if distances_ratio < geodesic_to_euclid_min_ratio:
            continue

        angle = np.random.uniform(0, 2 * np.pi)
        init_rotation = [0, np.sin(angle / 2), 0, np.cos(angle / 2)]

        try:
            shortest_path = get_action_shortest_path(sim,
                                                    source_position=init_pos,
                                                    source_rotation=init_rotation,
                                                    goal_position=goal_pos,
                                                    success_distance=shortest_path_success_distance,
                                                    max_episode_steps=shortest_path_max_steps)
        except GreedyFollowerError: # follower throws exception when cannot find path
            continue

        if len(shortest_path) > episode_max_steps or len(shortest_path) <= episode_min_steps:
            continue

        # grab only action sequence
        path=[]
        for k in range(len(shortest_path)):
            path.append(shortest_path[k].action)
        path.append(None) # marks end of path, stop action expected

        # verify the episode by running the simulator over the estimated path and checking the metrics
        verified = verify_episode(sim,
                                  start_position=init_pos,
                                  start_rotation=init_rotation,
                                  shortest_path=path,
                                  goal_pos=goal_pos,
                                  start_end_episode_distance=geo_dist,
                                  success_distance=verification_success_distance,
                                  scene_id=scene_id,
                                  episode_counter=episode_counter,
                                  save_path_images=False)

        if not verified:
            print("Episode rejected after simulation verification!")
            continue

        print("Episode successfully verified!")
        ep = {}
        ep['episode_id'] = episode_counter
        ep['scene_id'] = 'data/scene_datasets/mp3d/'+scene_id+'/'+scene_id+".glb"
        ep['start_position'] = init_pos
        ep['start_rotation'] = init_rotation
        ep['goals'] = [{'position':goal_pos, 'radius':None}]
        ep['start_room'] = None
        ep['shortest_paths'] = [path]
        ep['info'] = {'geodesic_distance':geo_dist,
                      'euclidean_distance':euclid_dist,
                      'geo_to_euclid_ratio':distances_ratio,
                      'difficulty':'hard',
                      }

        episode_counter+=1
        episodes.append(ep)

    return episodes


def init_sim(options, scene_id):
    # Load config file with sensor information
    cfg = habitat.get_config(options.config_file)
    cfg.defrost()
    cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb'
    cfg.SIMULATOR.TURN_ANGLE = options.turn_angle #10
    cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size #0.25
    cfg.freeze()

    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    
    seed = 0
    sim.seed(seed)
    return sim, seed


if __name__ == '__main__':

    options = Params().parser.parse_args()

    scene_ids_list = ['2azQ1b91cZZ', '8194nk5LbLH', 'EU6Fwq7SyZv', 'QUCTc6BB5sX', 'TbHJrupSAjP', 'X7HyMhZNoso',
                         'Z6MFQCViBuw', 'oLBMNvg9in8', 'x8F5xyUWy9e', 'zsNo4HB9uLZ'] # 'pLe4wQe7qrG'

    all_scene_episodes = []

    for scene_id in scene_ids_list:
        print(scene_id)

        sim, seed = init_sim(options, scene_id)
        
        all_scene_episodes += generate_pointnav_episodes(sim, 
                                                        scene_id, 
                                                        num_episodes=options.num_episodes,
                                                        shortest_path_success_distance=options.shortest_path_success_distance, 
                                                        shortest_path_max_steps=options.shortest_path_max_steps,
                                                        closest_dist_limit=options.closest_dist_limit,
                                                        max_dist_limit=options.max_dist_limit,
                                                        island_radius_limit=options.island_radius_limit,
                                                        episode_max_steps=options.episode_max_steps,
                                                        episode_min_steps=options.episode_min_steps,
                                                        verification_success_distance=options.verification_success_distance,
                                                        geodesic_to_euclid_min_ratio=options.geodesic_to_euclid_min_ratio)

        sim.close()

    scene_new_data = {'episodes': all_scene_episodes}

    out_save_dir = options.root_path + options.episodes_save_dir + options.split + "/"
    if not os.path.exists(out_save_dir):
        os.makedirs(out_save_dir)

    out_file = out_save_dir + options.split + ".json.gz"
    with gzip.open(out_file, "wt") as f:
        json.dump(scene_new_data, f)
    print("Saved episodes at:", out_file)

    print("Total episodes created:", len(all_scene_episodes))