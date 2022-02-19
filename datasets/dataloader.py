
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
import habitat
from habitat.config.default import get_config
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import os
import gzip
import json
import pickle


class HabitatDataOffline(Dataset):

    def __init__(self, options, config_file, finetune=False):
        config = get_config(config_file)
        self.config = config
        
        self.finetune = finetune # whether we are running a finetuning active job

        self.episodes_file_list = []
        self.episodes_file_list += self.collect_stored_episodes(options, split=config.DATASET.SPLIT)
        
        if options.dataset_percentage < 1: # Randomly choose the subset of the dataset to be used
            random.shuffle(self.episodes_file_list)
            self.episodes_file_list = self.episodes_file_list[ :int(len(self.episodes_file_list)*options.dataset_percentage) ]
        self.number_of_episodes = len(self.episodes_file_list)


    def collect_stored_episodes(self, options, split):
        episodes_dir = options.stored_episodes_dir + split + "/"
        episodes_file_list = []
        _scenes_dir = os.listdir(episodes_dir)
        scenes_dir = [ x for x in _scenes_dir if os.path.isdir(episodes_dir+x) ]
        for scene in scenes_dir:
            for fil in os.listdir(episodes_dir+scene+"/"):
                episodes_file_list.append(episodes_dir+scene+"/"+fil)
        return episodes_file_list


    def __len__(self):
        return self.number_of_episodes


    def __getitem__(self, idx):
        # Load from the pre-stored objnav training episodes
        ep_file = self.episodes_file_list[idx]
        ep = np.load(ep_file)

        abs_pose = ep['abs_pose']
        step_ego_grid_crops_spatial = torch.from_numpy(ep['step_ego_grid_crops_spatial'])
        gt_grid_crops_spatial = torch.from_numpy(ep['gt_grid_crops_spatial'])

        ### Transform abs_pose to rel_pose
        rel_pose = []
        for i in range(abs_pose.shape[0]):
            rel_pose.append(utils.get_rel_pose(pos2=abs_pose[i,:], pos1=abs_pose[0,:]))

        item = {}
        item['pose'] = torch.from_numpy(np.asarray(rel_pose)).float()
        item['abs_pose'] = torch.from_numpy(abs_pose).float()
        item['step_ego_grid_crops_spatial'] = step_ego_grid_crops_spatial
        item['gt_grid_crops_spatial'] = gt_grid_crops_spatial # Long tensor, int64

        return item


## Loads the simulator and episodes separately to enable per_scene collection of data
class HabitatDataScene(Dataset):

    def __init__(self, options, config_file, scene_id, existing_episode_list=[]):
        self.scene_id = scene_id

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = habitat.get_config(config_file)
        cfg.defrost()
        cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb' # scene_dataset_path
        cfg.SIMULATOR.TURN_ANGLE = options.turn_angle
        cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size
        cfg.ENVIRONMENT.MAX_EPISODE_STEPS = options.max_steps
        cfg.freeze()

        self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
        
        # Load pose noise models from Neural SLAM
        if options.noisy_pose:
            self.sensor_noise_fwd = \
                    pickle.load(open("noise_models/sensor_noise_fwd.pkl", 'rb'))
            self.sensor_noise_right = \
                    pickle.load(open("noise_models/sensor_noise_right.pkl", 'rb'))
            self.sensor_noise_left = \
                    pickle.load(open("noise_models/sensor_noise_left.pkl", 'rb'))

        seed = 0
        self.sim.seed(seed)

        ## Load episodes of scene_id

        # Pointnav val episodes are all in a single file
        ep_file_path = options.root_path + options.episodes_root + cfg.DATASET.SPLIT + "/" + cfg.DATASET.SPLIT + ".json.gz"
        with gzip.open(ep_file_path, "rt") as fp:
            self.data = json.load(fp)

        if cfg.DATASET.SPLIT=="test": # partition episodes based on difficulty
            self.add_difficulty()

        # Need to keep only episodes that belong to current scene
        self.scene_data = {'episodes': []}
        for i in range(len(self.data['episodes'])):
            sc_path = self.data['episodes'][i]['scene_id']
            sc_id = sc_path.split('/')[-1].split('.')[0]
            if sc_id == self.scene_id:
                self.scene_data['episodes'].append(self.data['episodes'][i])
        self.number_of_episodes = len(self.scene_data["episodes"])

        ## Uncomment following lines to show episode statistics
        '''
        geo_dict={'easy':[], 'medium':[], 'hard':[]}
        geo_to_euclid_dict={'easy':[], 'medium':[], 'hard':[]}
        geo_list, geo_to_euclid_list = [], []
        for i in range(len(self.data['episodes'])):
            episode = self.data['episodes'][i]
            eucl = utils.euclidean_distance(position_a=np.asarray(episode['start_position']),
                                            position_b=np.asarray(episode['goals'][0]['position']))
            geo_dict[episode['info']['difficulty']].append(episode['info']['geodesic_distance'])
            geo_to_euclid_dict[episode['info']['difficulty']].append(episode['info']['geodesic_distance']/eucl)
            geo_list.append(episode['info']['geodesic_distance'])
            geo_to_euclid_list.append(episode['info']['geodesic_distance']/eucl)
        for diff in geo_dict.keys():
            print('N:', len(geo_dict[diff]))
            print("Geodesic distance: Mean:", diff, np.mean(np.asarray(geo_dict[diff])), 'Std:', np.std(np.asarray(geo_dict[diff])))
            print("Geodesic to euclidean ratio: Mean:", diff, np.mean(np.asarray(geo_to_euclid_dict[diff])), 'Std:', np.std(np.asarray(geo_to_euclid_dict[diff])))
        print('Geodesic distance Overall Mean:', np.mean(np.asarray(geo_list)))
        print('Geodesic to euclidean ratio Overall Mean:', np.mean(np.asarray(geo_to_euclid_list)))
        print('Min GED:', np.min(np.asarray(geo_to_euclid_list)))
        '''
        
        self.success_distance = cfg.TASK.SUCCESS.SUCCESS_DISTANCE

        ## Dataloader params
        self.hfov = float(cfg.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
        self.cfg_norm_depth = cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        self.max_depth = cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.min_depth = cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH
        self.spatial_labels = options.n_spatial_classes
        self.grid_dim = (options.grid_dim, options.grid_dim)
        self.cell_size = options.cell_size
        self.crop_size = (options.crop_size, options.crop_size)
        self.img_size = (options.img_size, options.img_size)
        self.normalize = True
        self.pixFormat = 'NCHW'
        self.preprocessed_scenes_dir = options.root_path + options.scenes_dir + "mp3d_scene_pclouds/"

        self.episode_len = options.episode_len
        self.truncate_ep = options.truncate_ep

         # get point cloud and labels of scene
        self.pcloud, self.label_seq_spatial = utils.load_scene_pcloud(self.preprocessed_scenes_dir, self.scene_id)
        if len(existing_episode_list)!=0:
            self.existing_episode_list = [ int(x.split('_')[2]) for x in existing_episode_list ]
        else:
            self.existing_episode_list=[]

        self.occupancy_height_thresh = options.occupancy_height_thresh

        # Build 3D transformation matrices
        self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_size[0]), np.linspace(1,-1,self.img_size[1])), device='cuda')
        self.xs = self.xs.reshape(1,self.img_size[0],self.img_size[1])
        self.ys = self.ys.reshape(1,self.img_size[0],self.img_size[1])
        K = np.array([
            [1 / np.tan(self.hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(self.hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.inv_K = torch.tensor(np.linalg.inv(K), device='cuda')
        # create the points2D containing all image coordinates
        x, y = torch.tensor(np.meshgrid(np.linspace(0, self.img_size[0]-1, self.img_size[0]), np.linspace(0, self.img_size[1]-1, self.img_size[1])), device='cuda')
        xy_img = torch.vstack((x.reshape(1,self.img_size[0],self.img_size[1]), y.reshape(1,self.img_size[0],self.img_size[1])))
        points2D_step = xy_img.reshape(2, -1)
        self.points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2


    def add_difficulty(self):
        ## Val episodes already have difficulty info where geo dist >13m is hard, >7m is medium, test episodes do not
        for i in range(len(self.data['episodes'])):
            geo_dist = self.data['episodes'][i]['info']['geodesic_distance']
            if geo_dist > 13.0:
                diffic = 'hard'
            elif geo_dist > 7.0:
                diffic = 'medium'
            else:
                diffic = 'easy'
            self.data['episodes'][i]['info']['difficulty'] = diffic


    def __len__(self):
        return self.number_of_episodes

    ### ** __getitem__ is used only during store_episodes_parallel to generate training episodes
    def __getitem__(self, idx):
        episode = self.scene_data['episodes'][idx]

        len_shortest_path = len(episode['shortest_paths'][0])
        objectgoal = episode['object_category']

        if len_shortest_path > 50: # skip that episode to avoid memory issues
            return None
        if len_shortest_path < self.episode_len+1:
            return None

        if idx in self.existing_episode_list:
            print("Episode", idx, 'already exists!')
            return None

        scene = self.sim.semantic_annotations()
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        # convert the labels to the reduced set of categories
        instance_id_to_label_id_3 = instance_id_to_label_id.copy()
        for inst_id in instance_id_to_label_id.keys():
            curr_lbl = instance_id_to_label_id[inst_id]
            instance_id_to_label_id_3[inst_id] = viz_utils.label_conversion_40_3[curr_lbl]

        # if truncated, run episode only up to the chosen step start_ind+episode_len
        if self.truncate_ep:
            start_ind = random.randint(0, len_shortest_path-self.episode_len-1)
            episode_extend = start_ind+self.episode_len
        else:
            episode_extend = len_shortest_path

        imgs = torch.zeros((episode_extend, 3, self.img_size[0], self.img_size[1]), dtype=torch.float32, device=self.device)
        depth_imgs = torch.zeros((episode_extend, 1, self.img_size[0], self.img_size[1]), dtype=torch.float32, device=self.device)

        points2D, local3D, abs_poses, rel_poses, action_seq, agent_height = [], [], [], [], [], []

        self.sim.reset()
        self.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
        sim_obs = self.sim.get_sensor_observations()
        observations = self.sim._sensor_suite.get_observations(sim_obs)


        for i in range(episode_extend):
            img = observations['rgb'][:,:,:3]
            depth_obsv = observations['depth'].permute(2,0,1).unsqueeze(0)

            depth = F.interpolate(depth_obsv.clone(), size=self.img_size, mode='nearest')
            depth = depth.squeeze(0).permute(1,2,0)

            if self.cfg_norm_depth:
                depth = utils.unnormalize_depth(depth, min=self.min_depth, max=self.max_depth)            

            # visual and 3d info
            imgData = utils.preprocess_img(img, cropSize=self.img_size, pixFormat=self.pixFormat, normalize=self.normalize)
            local3D_step = utils.depth_to_3D(depth, self.img_size, self.xs, self.ys, self.inv_K)

            agent_pose, y_height = utils.get_sim_location(agent_state=self.sim.get_agent_state())

            imgs[i,:,:,:] = imgData
            depth_resize = F.interpolate(depth_obsv.clone(), size=self.img_size, mode='nearest')
            depth_imgs[i,:,:,:] = depth_resize.squeeze(0)

            abs_poses.append(agent_pose)
            agent_height.append(y_height)
            points2D.append(self.points2D_step)
            local3D.append(local3D_step)

            # get the relative pose with respect to the first pose in the sequence
            rel = utils.get_rel_pose(pos2=abs_poses[i], pos1=abs_poses[0])
            rel_poses.append(rel)

            # explicitly clear observation otherwise they will be kept in memory the whole time
            observations = None

            action_id = episode['shortest_paths'][0][i]
            if action_id==None:
                break
            observations = self.sim.step(action_id)


        pose = torch.from_numpy(np.asarray(rel_poses)).float()
        abs_pose = torch.from_numpy(np.asarray(abs_poses)).float()

        # Create the ground-projected grids
        ego_grid_sseg_3 = map_utils.est_occ_from_depth(local3D, grid_dim=self.grid_dim, cell_size=self.cell_size, 
                                                    device=self.device, occupancy_height_thresh=self.occupancy_height_thresh)

        ego_grid_crops_3 = map_utils.crop_grid(grid=ego_grid_sseg_3, crop_size=self.crop_size)
        step_ego_grid_3 = map_utils.get_acc_proj_grid(ego_grid_sseg_3, pose, abs_pose, self.crop_size, self.cell_size)
        step_ego_grid_crops_3 = map_utils.crop_grid(grid=step_ego_grid_3, crop_size=self.crop_size)
        # Get cropped gt
        gt_grid_crops_spatial = map_utils.get_gt_crops(abs_pose, self.pcloud, self.label_seq_spatial, agent_height,
                                                            self.grid_dim, self.crop_size, self.cell_size)

        item = {}
        item['images'] = imgs
        item['depth_imgs'] = depth_imgs
        item['episode_id'] = idx
        item['scene_id'] = self.scene_id
        item['abs_pose'] = abs_pose
        item['step_ego_grid_crops_spatial'] = step_ego_grid_crops_3
        item['gt_grid_crops_spatial'] = gt_grid_crops_spatial
        return item
