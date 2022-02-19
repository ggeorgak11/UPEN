import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataloader import HabitatDataScene
from models.predictors import get_predictor_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import test_utils as tutils
from models.semantic_grid import SemanticGrid
import os
import metrics
import json
import cv2
import random
import math
from planning.ddppo_policy import DdppoPolicy
from planning.rrt_star import RRTStar
from frontier_exploration.frontier_search import FrontierSearch


class NavTester(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options, scene_id):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # build summary dir
        summary_dir = os.path.join(self.options.log_dir, scene_id)
        summary_dir = os.path.join(summary_dir, 'tensorboard')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # tensorboardX SummaryWriter for use in save_summaries
        self.summary_writer = SummaryWriter(summary_dir)

        # point to our generated test episodes
        self.options.episodes_root = "habitat-api/data/datasets/pointnav/mp3d/"+self.options.test_set+"/"

        self.scene_id = scene_id

        if self.options.split=="val":
            if self.options.noisy_actions:
                config_file = self.options.config_val_file_noisy
            else:
                config_file = self.options.config_val_file
        elif self.options.split=="test":
            if self.options.noisy_actions:
                config_file = self.options.config_test_file_noisy
            else:
                config_file = self.options.config_test_file

        self.test_ds = HabitatDataScene(self.options, config_file=config_file, scene_id=self.scene_id)

        ensemble_exp = os.listdir(self.options.ensemble_dir) # ensemble_dir should be a dir that holds multiple experiments
        ensemble_exp.sort() # in case the models are numbered put them in order
        self.models_dict = {} # keys are the ids of the models in the ensemble
        for n in range(self.options.ensemble_size):
            self.models_dict[n] = {'predictor_model': get_predictor_from_options(self.options)}
            self.models_dict[n] = {k:v.to(self.device) for k,v in self.models_dict[n].items()}

            # Needed only for models trained with multi-gpu setting
            self.models_dict[n]['predictor_model'] = nn.DataParallel(self.models_dict[n]['predictor_model'])

            checkpoint_dir = self.options.ensemble_dir + "/" + ensemble_exp[n]
            latest_checkpoint = tutils.get_latest_model(save_dir=checkpoint_dir)
            print("Model", n, "loading checkpoint", latest_checkpoint)
            self.models_dict[n] = tutils.load_model(models=self.models_dict[n], checkpoint_file=latest_checkpoint)
            self.models_dict[n]["predictor_model"].eval()

        self.step_count = 0

        self.map_metrics = ['map_accuracy', 'map_accuracy_m2', 'iou', 'f1', 'cov', 'cov_per']
        self.metrics = ['distance_to_goal', 'success', 'spl', 'softspl']
        self.diff_list = ['easy', 'medium', 'hard']
        # initialize metrics
        self.results = {}
        for diff in self.diff_list:
            self.results[diff] = {}
            for met in self.metrics:
                self.results[diff][met] = []
        self.results['all'] = {}
        for met in self.metrics:
            self.results['all'][met] = []
        for map_met in self.map_metrics:
            self.results[map_met] = []
        
        # init local policy model
        if self.options.local_policy_model=="4plus":
            model_ext = 'gibson-4plus-mp3d-train-val-test-resnet50.pth'
        else:
            model_ext = 'gibson-2plus-resnet50.pth'
        model_path = self.options.root_path + "local_policy_models/" + model_ext
        self.l_policy = DdppoPolicy(path=model_path)
        self.l_policy = self.l_policy.to(self.device)


    def test_navigation(self):

        with torch.no_grad():

            list_dist_to_goal, list_success, list_spl, list_soft_spl = [],[],[],[]
            list_map_accuracy, list_map_accuracy_m2, list_iou, list_f1, list_cov, list_cov_per = [], [], [], [], [], []

            for idx in range(len(self.test_ds)):

                episode = self.test_ds.scene_data['episodes'][idx]

                print("Ep:", idx, "Geo dist:", episode['info']['geodesic_distance'], "Difficulty:", episode['info']['difficulty'])
                self.step_count+=1 # episode counter for tensorboard
                
                self.test_ds.sim.reset()
                self.test_ds.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
                sim_obs = self.test_ds.sim.get_sensor_observations()
                observations = self.test_ds.sim._sensor_suite.get_observations(sim_obs)

                # For each episode we need a new instance of a fresh global grid
                sg = SemanticGrid(1, self.test_ds.grid_dim, self.test_ds.crop_size[0], self.test_ds.cell_size,
                                                spatial_labels=self.test_ds.spatial_labels, ensemble_size=self.options.ensemble_size)

                ### Get goal position in 2D map coords
                if self.options.exploration:
                    # when exploration mode, use a dummy unreachable goal
                    goal_pose_coords = torch.zeros((1, 1, 2), dtype=torch.int64).to(self.device)
                    goal_pose_coords[0,0,0], goal_pose_coords[0,0,1] = -100, -100 
                    # ensure that the goal sampling rate is below 0
                    self.options.goal_sample_rate = -1
                else:
                    init_agent_pose, init_agent_height = tutils.get_2d_pose(position=episode["start_position"], rotation=episode["start_rotation"])
                    goal_3d = episode['goals'][0]['position']
                    goal_pose, goal_height = tutils.get_2d_pose(position=goal_3d)
                    agent_rel_goal = utils.get_rel_pose(pos2=goal_pose, pos1=init_agent_pose)
                    agent_rel_goal = torch.Tensor(agent_rel_goal).unsqueeze(0).float()
                    agent_rel_goal = agent_rel_goal.to(self.device)
                    goal_pose_coords = tutils.get_coord_pose(sg, agent_rel_goal, init_agent_pose, self.test_ds.grid_dim[0], self.test_ds.cell_size, self.device) # B x T x 3


                # Initialize the local policy hidden state
                self.l_policy.reset()

                abs_poses = []
                rel_poses_list = []
                abs_poses_noisy = []
                pose_coords_list = []
                pose_coords_noisy_list = []
                stg_pos_list = []
                agent_height = []
                t = 0
                stg_counter=0
                stg_goal_coords = goal_pose_coords.clone() # initialize short-term goal with final goal location
                agent_episode_distance = 0.0 # distance covered by agent at any given time in the episode
                previous_pos = self.test_ds.sim.get_agent_state().position
                prev_path = None
                prev_action = None
                prev_observations = None


                while t < self.options.max_steps:

                    img = observations['rgb'][:,:,:3]
                    depth = observations['depth'].reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1)

                    if self.test_ds.cfg_norm_depth:
                        depth_abs = utils.unnormalize_depth(depth.clone(), min=self.test_ds.min_depth, max=self.test_ds.max_depth)

                    # 3d info
                    local3D_step = utils.depth_to_3D(depth_abs, self.test_ds.img_size, self.test_ds.xs, self.test_ds.ys, self.test_ds.inv_K)

                    agent_pose, y_height = utils.get_sim_location(agent_state=self.test_ds.sim.get_agent_state())

                    abs_poses.append(agent_pose)
                    agent_height.append(y_height)

                    # get the relative pose with respect to the first pose in the sequence
                    rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
                    _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
                    _rel_pose = _rel_pose.to(self.device)
                    rel_poses_list.append(_rel_pose.clone())

                    pose_coords = tutils.get_coord_pose(sg, _rel_pose, abs_poses[0], self.test_ds.grid_dim[0], self.test_ds.cell_size, self.device) # B x T x 3
                    pose_coords_list.append(pose_coords.clone().cpu().numpy())

                    if t==0:
                        # get gt map from initial agent pose for visualization at end of episode
                        x, y, label_seq = map_utils.slice_scene(x=self.test_ds.pcloud[0].copy(),
                                                                y=self.test_ds.pcloud[1].copy(),
                                                                z=self.test_ds.pcloud[2].copy(),
                                                                label_seq=self.test_ds.label_seq_spatial.copy(),
                                                                height=agent_height[0])
                        gt_map_initial = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[0],
                                                                    grid_dim=self.test_ds.grid_dim, cell_size=self.test_ds.cell_size)

                    # Add pose noise from Neural SLAM
                    if self.options.noisy_pose:
                        if t==0:
                            abs_poses_noisy.append(agent_pose)
                        else:
                            # following process from here: https://github.com/devendrachaplot/Neural-SLAM/blob/master/env/habitat/exploration_env.py#L230
                            rel_one_step_gt = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[t-1]) # get gt pose change for one step
                            rel_one_step_gt = torch.Tensor(rel_one_step_gt).unsqueeze(0).float()
                            rel_one_step_gt = rel_one_step_gt.to(self.device)
                            rel_one_step_noisy = self.add_pose_noise(rel_one_step_gt, action_id) # add noise to the gt pose change for one step
                            noisy_pose = utils.get_new_pose(abs_poses_noisy[t-1], rel_one_step_noisy.squeeze(0).cpu().numpy())
                            abs_poses_noisy.append(noisy_pose)
                            # overwrite the gt _rel_pose and pose coords with their noisy counterparts
                            rel = utils.get_rel_pose(pos2=abs_poses_noisy[t], pos1=abs_poses_noisy[0])
                            _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
                            _rel_pose = _rel_pose.to(self.device)
                            pose_coords = tutils.get_coord_pose(sg, _rel_pose, abs_poses[0], self.test_ds.grid_dim[0], self.test_ds.cell_size, self.device) # B x T x 3
                            pose_coords_noisy_list.append(pose_coords.clone().cpu().numpy())                

                    # do ground-projection, update the map
                    ego_grid_sseg_3 = map_utils.est_occ_from_depth([local3D_step], grid_dim=self.test_ds.grid_dim, cell_size=self.test_ds.cell_size, 
                                                                                    device=self.device, occupancy_height_thresh=self.options.occupancy_height_thresh)

                    # Transform the ground projected egocentric grids to geocentric using relative pose
                    geo_grid_sseg = sg.spatialTransformer(grid=ego_grid_sseg_3, pose=rel_poses_list[t], abs_pose=torch.tensor(abs_poses).to(self.device))
                    # step_geo_grid contains the map snapshot every time a new observation is added
                    step_geo_grid_sseg = sg.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
                    # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
                    step_ego_grid_sseg = sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=rel_poses_list[t], abs_pose=torch.tensor(abs_poses).to(self.device))
                    # Crop the grid around the agent at each timestep
                    step_ego_grid_crops = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=self.test_ds.crop_size)

                    mean_ensemble_spatial, ensemble_spatial_maps = self.run_map_predictor(step_ego_grid_crops)

                    # add occupancy prediction to semantic map
                    sg.register_occ_pred(prediction_crop=mean_ensemble_spatial, pose=_rel_pose, abs_pose=torch.tensor(abs_poses, device=self.device))

                    # registers each model in the ensemble to a separate global map
                    sg.register_model_occ_pred(ensemble_prediction_crop=ensemble_spatial_maps, pose=_rel_pose, abs_pose=torch.tensor(abs_poses, device=self.device))

                    # Option to save visualizations of steps
                    if self.options.save_nav_images:
                        save_img_dir_ = self.options.log_dir + "/" + self.scene_id + "/" + self.options.save_img_dir_+'ep_'+str(idx)+'/'
                        if not os.path.exists(save_img_dir_):
                            os.makedirs(save_img_dir_)
                        # saves current state of map, mean_ensemble, var_ensemble
                        viz_utils.save_map_snapshot(sg, pose_coords_list, stg_goal_coords.clone().cpu().numpy(), 
                                                        goal_pose_coords.clone().cpu().numpy(), save_img_dir_, t, self.options.exploration)
                        # saves egocentric rgb, depth observations
                        viz_utils.display_sample(img.cpu().numpy(), np.squeeze(depth_abs.cpu().numpy()), 
                                                                    savepath=save_img_dir_+"path_"+str(t)+'.png')
                        # saves predicted areas (egocentric)
                        viz_utils.save_map_pred_steps(step_ego_grid_crops, mean_ensemble_spatial, save_img_dir_, t)


                    stg_dist = torch.linalg.norm(stg_goal_coords.clone().float()-pose_coords.float())*self.options.cell_size # distance to short term goal
                    # Get the short-term goal either every k steps or if we have already reached it
                    if ((stg_counter % self.options.steps_after_plan == 0) or (stg_dist < 0.1)): # or we reached stg
                        
                        planning_grid = sg.occ_grid.clone()

                        if self.options.with_rrt_planning:
                            rrt_goal, rrt_best_path, path_dict = self.get_rrt_goal(pose_coords.clone(), goal_pose_coords.clone(), 
                                                                            grid=planning_grid, ensemble=sg.model_occ_grid, prev_path=prev_path)
                            stg_counter=0
                            if rrt_goal is not None:
                                stg_goal_coords = rrt_goal
                                stg_pos_list.append(stg_goal_coords)
                                prev_path = rrt_best_path
                                if self.options.save_nav_images:
                                    viz_utils.save_rrt_path(sg, rrt_best_path, t, save_img_dir_, stg_goal_coords.clone().cpu().detach(),
                                                                        pose_coords.clone().cpu().detach(), goal_pose_coords.clone().cpu().detach(), self.options.exploration)
                                    viz_utils.save_all_paths(sg, path_dict, pose_coords.clone().cpu().numpy(), 
                                                                        goal_pose_coords.clone().cpu().numpy(), save_img_dir_, t, self.options.exploration)
                            else:
                                prev_path = None
                                print(t, "Path not found!")

                        elif self.options.fbe:
                            planning_grid = planning_grid.detach().squeeze(0).cpu().numpy()
                            fbe = FrontierSearch(idx, t, planning_grid, 0, 'closest')
                            stg_goal_coords = torch.tensor(fbe.nextGoal(pose_coords.cpu().numpy(), _rel_pose.cpu().numpy(), min_thresh=20))
                            stg_goal_coords = stg_goal_coords.to(self.device)
                            stg_pos_list.append(stg_goal_coords)
                            stg_counter=0

                    stg_counter+=1

                    # Estimate current distance to final goal
                    goal_dist = torch.linalg.norm(goal_pose_coords.clone().float()-pose_coords.float())*self.options.cell_size

                    # Use DD-PPO model
                    action_id = self.run_local_policy(depth=depth, goal=stg_goal_coords.clone(),
                                                            pose_coords=pose_coords.clone(), rel_agent_o=rel[2], step=t)
                    if tutils.decide_stop(goal_dist, self.options.stop_dist) or t==self.options.max_steps-1:
                        t+=1
                        break
                    if action_id==0: # when ddppo predicts stop, then randomly choose an action
                        action_id = random.randint(1,3)    
                    prev_action = action_id

                    # explicitly clear observation otherwise they will be kept in memory the whole time
                    observations = None
                    
                    # Apply next action
                    observations = self.test_ds.sim.step(action_id)

                    # estimate distance covered by agent
                    current_pos = self.test_ds.sim.get_agent_state().position
                    agent_episode_distance += utils.euclidean_distance(current_pos, previous_pos)
                    previous_pos = current_pos

                    t+=1

                
                ## Episode ended ##
                output = {'metrics': {}}
                if not self.options.exploration:
                    nav_metrics = tutils.get_metrics(sim=self.test_ds.sim,
                                                episode_goal_positions=goal_3d,
                                                success_distance=self.test_ds.success_distance,
                                                start_end_episode_distance=episode['info']['geodesic_distance'],
                                                agent_episode_distance=agent_episode_distance,
                                                stop_signal=True)
                    
                    for met in self.metrics:
                        self.results[episode['info']['difficulty']][met].append(nav_metrics[met])
                        self.results['all'][met].append(nav_metrics[met])

                    # for tensorboard
                    list_dist_to_goal.append(nav_metrics['distance_to_goal'])
                    list_success.append(nav_metrics['success'])
                    list_spl.append(nav_metrics['spl'])
                    list_soft_spl.append(nav_metrics['softspl'])

                    output['metrics']['mean_dist_to_goal'] = np.mean(np.asarray(list_dist_to_goal.copy()))
                    output['metrics']['mean_success'] = np.mean(np.asarray(list_success.copy()))
                    output['metrics']['mean_spl'] = np.mean(np.asarray(list_spl.copy()))
                    output['metrics']['mean_soft_spl'] = np.mean(np.asarray(list_soft_spl.copy()))


                ## Visualization of entire map and agent trajectory
                save_path = self.options.log_dir + "/" + self.scene_id + "/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                
                viz_utils.save_trajectory(sg, gt_map_initial, goal_pose_coords.clone().cpu().numpy(), 
                                            stg_pos_list, pose_coords_list, pose_coords_noisy_list, self.options.exploration, filename=save_path+str(idx)+".png")
                
                ## Get map metrics
                map_metrics = tutils.get_map_metrics(init_gt_map=gt_map_initial, 
                                                     sg=sg, 
                                                     cell_size=self.test_ds.cell_size, 
                                                     spatial_labels=self.options.n_spatial_classes)
                list_map_accuracy.append(map_metrics['map_accuracy'])
                list_iou.append(map_metrics['iou'])
                list_cov.append(map_metrics['cov'])
                list_cov_per.append(map_metrics['cov_per'])
                list_f1.append(map_metrics['f1'])
                list_map_accuracy_m2.append(map_metrics['map_accuracy_m2'])

                output['metrics']['mean_map_acc'] = np.mean(np.asarray(list_map_accuracy.copy()))
                output['metrics']['mean_iou'] = np.mean(np.asarray(list_iou.copy()))
                output['metrics']['mean_cov'] = np.mean(np.asarray(list_cov.copy()))
                output['metrics']['mean_cov_per'] = np.mean(np.asarray(list_cov_per.copy()))
                output['metrics']['mean_f1'] = np.mean(np.asarray(list_f1.copy()))
                output['metrics']['mean_map_acc_m2'] = np.mean(np.asarray(list_map_accuracy_m2.copy()))

                for map_met in self.map_metrics:
                    self.results[map_met].append(map_metrics[map_met])

                self.save_test_summaries(output)

            ## Scene ended ##
            # write results to json
            for diff in self.diff_list:
                for met in self.metrics:
                    self.results[diff]["mean_"+met] = np.mean(np.asarray(self.results[diff][met])) # per difficulty per metric mean
            
            for met in self.metrics:
                self.results['all']["mean_"+met] = np.mean(np.asarray(self.results["all"][met]))
            for map_met in self.map_metrics:
                self.results["mean_"+map_met] = np.mean(np.asarray(self.results[map_met]))

            with open(self.options.log_dir+'/results_'+self.scene_id+'.json', 'w') as outfile:
                json.dump(self.results, outfile, indent=4)

            # Close current scene
            self.test_ds.sim.close()

    
    def add_pose_noise(self, rel_pose, action_id):
        if action_id == 1:
            x_err, y_err, o_err = self.test_ds.sensor_noise_fwd.sample()[0][0]
        elif action_id == 2:
            x_err, y_err, o_err = self.test_ds.sensor_noise_left.sample()[0][0]
        elif action_id == 3:
            x_err, y_err, o_err = self.test_ds.sensor_noise_right.sample()[0][0]
        else:
            x_err, y_err, o_err = 0., 0., 0.
        rel_pose[0,0] += x_err*self.options.noise_level
        rel_pose[0,1] += y_err*self.options.noise_level
        rel_pose[0,2] += torch.tensor(np.deg2rad(o_err*self.options.noise_level))
        return rel_pose


    def dot(self, v1, v2):
        return v1[0]*v2[0]+v1[1]*v2[1]

    def get_angle(self, line1, line2):
        v1 = [(line1[0][0]-line1[1][0]), (line1[0][1]-line1[1][1])]
        v2 = [(line2[0][0]-line2[1][0]), (line2[0][1]-line2[1][1])]
        dot_prod = self.dot(v1, v2)
        mag1 = self.dot(v1, v1)**0.5 + 1e-5
        mag2 = self.dot(v2, v2)**0.5 + 1e-5
        cos_ = dot_prod/mag1/mag2 
        angle = math.acos(dot_prod/mag2/mag1)
        ang_deg = math.degrees(angle)%360

        if ang_deg-180>=0:
            ang_deg = 360 - ang_deg

        return ang_deg


    def eval_path(self, ensemble, path, prev_path):
        reach_per_model = []
        for k in range(ensemble.shape[0]):
            model = ensemble[k].squeeze(0)
            reachability = []    
            for idx in range(min(self.options.reach_horizon,len(path))-1):
                node1 = path[idx]
                node2 = path[idx+1]

                maxdist = max(abs(node1[0]-node2[0]), abs(node1[1]-node2[1])) +1

                xs = np.linspace(int(node1[0]), int(node1[0]), int(maxdist))
                ys = np.linspace(int(node1[1]), int(node2[1]), int(maxdist))
                for i in range(len(xs)):
                    x = int(xs[i])
                    y = int(ys[i])
                    reachability.append(model[1,x,y]) # probability of occupancy
            reach_per_model.append(max(reachability))
        avg = torch.mean(torch.tensor(reach_per_model))
        std = torch.sqrt(torch.var(torch.tensor(reach_per_model)))
        path_len = len(path) / 100 # normalize by a pseudo max length
        #print(path_len)
        result = avg - self.options.a_1*std + self.options.a_2*path_len
        
        if prev_path:
            angle = (self.get_angle((path[0], path[min(self.options.reach_horizon,len(path))-1]), (prev_path[0], prev_path[min(self.options.reach_horizon,len(prev_path))-1]))) / 360.0
            result += self.options.a_3 * angle

        return result


    def eval_path_expl(self, ensemble, paths):
        # evaluate each path based on its average occupancy uncertainty
        #N, B, C, H, W = ensemble.shape # number of models, batch, classes, height, width
        ### Estimate the variance only of the occupied class (1) for each location # 1 x B x object_classes x grid_dim x grid_dim
        ensemble_occupancy_var = torch.var(ensemble[:,:,1,:,:], dim=0, keepdim=True).squeeze(0) # 1 x H x W
        path_sum_var = []
        for k in range(len(paths)):
            path = paths[k]
            path_var = []
            for idx in range(min(self.options.reach_horizon,len(path))-1):
                node1 = path[idx]
                node2 = path[idx+1]
                maxdist = max(abs(node1[0]-node2[0]), abs(node1[1]-node2[1])) +1
                xs = np.linspace(int(node1[0]), int(node1[0]), int(maxdist))
                ys = np.linspace(int(node1[1]), int(node2[1]), int(maxdist))
                for i in range(len(xs)):
                    x = int(xs[i])
                    y = int(ys[i])          
                    path_var.append(ensemble_occupancy_var[0,x,y])
            path_sum_var.append( np.sum(np.asarray(path_var)) )
        return path_sum_var


    def get_rrt_goal(self, pose_coords, goal, grid, ensemble, prev_path):
        probability_map, indexes = torch.max(grid,dim=1)
        probability_map = probability_map[0]
        indexes = indexes[0]
        binarymap = (indexes == 1)
        start = [int(pose_coords[0][0][1]), int(pose_coords[0][0][0])]
        finish = [int(goal[0][0][1]), int(goal[0][0][0])]
        rrt_star = RRTStar(start=start, 
                           obstacle_list=None, 
                           goal=finish, 
                           rand_area=[0,binarymap.shape[0]], 
                           max_iter=self.options.rrt_max_iters,
                           expand_dis=self.options.expand_dis,
                           goal_sample_rate=self.options.goal_sample_rate,
                           connect_circle_dist=self.options.connect_circle_dist,
                           occupancy_map=binarymap)
        best_path = None
        
        path_dict = {'paths':[], 'value':[]} # visualizing all the paths
        if self.options.exploration:
            paths = rrt_star.planning(animation=False, use_straight_line=self.options.rrt_straight_line, exploration=self.options.exploration, horizon=self.options.reach_horizon)
            ## evaluate each path on the exploration objective
            path_sum_var = self.eval_path_expl(ensemble, paths)
            path_dict['paths'] = paths
            path_dict['value'] = path_sum_var

            best_path_var = 0 # we need to select the path with maximum overall uncertainty
            for i in range(len(paths)):
                if path_sum_var[i] > best_path_var:
                    best_path_var = path_sum_var[i]
                    best_path = paths[i]

        else:
            best_path_reachability = float('inf')        
            for i in range(self.options.rrt_num_path):
                path = rrt_star.planning(animation=False, use_straight_line=self.options.rrt_straight_line)
                if path:
                    if self.options.rrt_path_metric == "reachability":
                        reachability = self.eval_path(ensemble, path, prev_path)
                    elif self.options.rrt_path_metric == "shortest":
                        reachability = len(path)
                    path_dict['paths'].append(path)
                    path_dict['value'].append(reachability)
                    
                    if reachability < best_path_reachability:
                        best_path_reachability = reachability
                        best_path = path

        if best_path:
            best_path.reverse()
            last_node = min(len(best_path)-1, self.options.reach_horizon)
            return torch.tensor([[[int(best_path[last_node][1]), int(best_path[last_node][0])]]]).cuda(), best_path, path_dict
        return None, None, None


    def run_local_policy(self, depth, goal, pose_coords, rel_agent_o, step):
        planning_goal = goal.squeeze(0).squeeze(0)
        planning_pose = pose_coords.squeeze(0).squeeze(0)
        sq = torch.square(planning_goal[0]-planning_pose[0])+torch.square(planning_goal[1]-planning_pose[1])
        rho = torch.sqrt(sq.float())
        phi = torch.atan2((planning_pose[0]-planning_goal[0]),(planning_pose[1]-planning_goal[1]))
        phi = phi - rel_agent_o
        rho = rho*self.test_ds.cell_size
        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32).to(self.device)
        depth = depth.reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
        return self.l_policy.plan(depth, point_goal_with_gps_compass, step)

    def run_map_predictor(self, step_ego_grid_crops):

        input_batch = {'step_ego_grid_crops_spatial': step_ego_grid_crops.unsqueeze(0)}
        input_batch = {k: v.to(self.device) for k, v in input_batch.items()}

        model_pred_output = {}
        ensemble_spatial_maps = []
        for n in range(self.options.ensemble_size):
            model_pred_output[n] = self.models_dict[n]['predictor_model'](input_batch)
            ensemble_spatial_maps.append(model_pred_output[n]['pred_maps_spatial'].clone())
        ensemble_spatial_maps = torch.stack(ensemble_spatial_maps) # N x B x T x C x cH x cW

        ### Estimate average predictions from the ensemble
        mean_ensemble_spatial = torch.mean(ensemble_spatial_maps, dim=0) # B x T x C x cH x cW
        return mean_ensemble_spatial, ensemble_spatial_maps


    def save_test_summaries(self, output):
        prefix = 'test/' + self.scene_id + '/'
        for k in output['metrics']:
            self.summary_writer.add_scalar(prefix + k, output['metrics'][k], self.step_count)
