from pytorch_utils.base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    """ Parses command line arguments for training
    This overwrites options from BaseOptions
    """
    def __init__(self): # pylint: disable=super-init-not-called
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=3600000,
                         help='Total time to run in seconds')
        gen.add_argument('--resume', dest='resume', default=False,
                         action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=0,
                         help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                         help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                         help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        in_out = self.parser.add_argument_group('io')
        in_out.add_argument('--log_dir', default='~/semantic_grid/logs', help='Directory to store logs')
        in_out.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        in_out.add_argument('--from_json', default=None,
                            help='Load options from json file instead of the command line')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=1000,
                           help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_nav_batch_size', type=int, default=1, help='Batch size during navigation test')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_test = train.add_mutually_exclusive_group()
        shuffle_test.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                                  help='Shuffle testing data')
        shuffle_test.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false',
                                  help='Don\'t shuffle testing data')

        # Dataset related options
        train.add_argument('--data_type', dest='data_type', type=str, default='train',
                            choices=['train', 'val'],
                            help='Choose which dataset to run on, valid only with --use_store')
        train.add_argument('--dataset_percentage', dest='dataset_percentage', type=float, default=1.0,
                            help='percentage of dataset to be used during training for ensemble learning')

        train.add_argument('--summary_steps', type=int, default=1000,
                           help='Summary saving frequency')
        train.add_argument('--image_summary_steps', type=int, default=5000,
                           help='Image summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=30000,
                           help='Chekpoint saving frequency')

        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')


        train.add_argument('--is_train', dest='is_train', action='store_true',
                            help='Define whether training or testing mode')


        train.add_argument('--config_train_file', type=str, dest='config_train_file',
                            default='configs/my_pointnav_mp3d_train.yaml',
                            help='path to habitat dataset train config file')

        self.parser.add_argument('--config_test_file', type=str, dest='config_test_file',
                                default='configs/my_pointnav_mp3d_test.yaml',
                                help='path to test config file')

        self.parser.add_argument('--config_val_file', type=str, dest='config_val_file',
                                default='configs/my_pointnav_mp3d_val.yaml',
                                help='path to habitat dataset val config file')

        self.parser.add_argument('--config_test_file_noisy', type=str, dest='config_test_file_noisy',
                                default='configs/my_pointnav_mp3d_test_noisy.yaml',
                                help='path to noisy test config file')

        self.parser.add_argument('--config_val_file_noisy', type=str, dest='config_val_file_noisy',
                                default='configs/my_pointnav_mp3d_val_noisy.yaml',
                                help='path to habitat dataset noisy val config file')


        self.parser.add_argument('--ensemble_dir', type=str, dest='ensemble_dir', default=None,
                                help='Path containing the experiments comprising the ensemble')

        self.parser.add_argument('--n_spatial_classes', type=int, default=3, dest='n_spatial_classes',
                                help='number of categories for spatial prediction')

        self.parser.add_argument('--grid_dim', type=int, default=768, dest='grid_dim',
                                    help='Semantic grid size (grid_dim, grid_dim)')
        self.parser.add_argument('--cell_size', type=float, default=0.05, dest="cell_size",
                                    help='Physical dimensions (meters) of each cell in the grid')
        self.parser.add_argument('--crop_size', type=int, default=160, dest='crop_size',
                                    help='Size of crop around the agent')

        self.parser.add_argument('--img_size', dest='img_size', type=int, default=256)

        train.add_argument('--map_loss_scale', type=float, default=1.0, dest='map_loss_scale')

        train.add_argument('--init_gaussian_weights', dest='init_gaussian_weights', action='store_true',
                            help='initializes the model weights from gaussian distribution')

        train.set_defaults(shuffle_train=True, shuffle_test=True)

        optim = self.parser.add_argument_group('Optim')
        optim.add_argument("--lr_decay", type=float,
                           default=0.99, help="Exponential decay rate")
        optim.add_argument("--wd", type=float,
                           default=0, help="Weight decay weight")

        self.parser.add_argument('--test_iters', type=int, default=20000)

        optimizer_options = self.parser.add_argument_group('Optimizer')
        optimizer_options.add_argument('--lr', type=float, default=0.0002)
        optimizer_options.add_argument('--beta1', type=float, default=0.5)

        model_options = self.parser.add_argument_group('Model')

        ## Hyperparameters for planning in test navigation

        self.parser.add_argument('--max_steps', type=int, dest='max_steps', default=500, choices=[500,1000],
                                  help='Maximum steps for each test episode')

        self.parser.add_argument('--steps_after_plan', type=int, dest='steps_after_plan', default=10, # 20
                                 help='how many times to use the local policy before selecting long-term-goal and replanning')

        self.parser.add_argument('--stop_dist', type=float, dest='stop_dist', default=0.19, # 0.1
                                 help='decision to stop distance')

        self.parser.add_argument('--turn_angle', dest='turn_angle', type=int, default=10,
                                help='angle to rotate left or right in degrees for habitat simulator')
        self.parser.add_argument('--forward_step_size', dest='forward_step_size', type=float, default=0.25,
                                help='distance to move forward in meters for habitat simulator')

        self.parser.add_argument('--save_nav_images', dest='save_nav_images', default=False, action='store_true',
                                 help='Keep track and store maps during navigation testing')


        # options relating to active training (using scenes dataloader)
        self.parser.add_argument('--ensemble_size', type=int, dest='ensemble_size', default=4)

        self.parser.add_argument('--root_path', type=str, dest='root_path', default="~/")
        self.parser.add_argument('--episodes_root', type=str, dest='episodes_root', default="")
        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='habitat-api/data/scene_datasets/')

        self.parser.add_argument('--stored_episodes_dir', type=str, dest='stored_episodes_dir', 
                                default='mp3d_pointnav_episodes_0/')

        
        self.parser.add_argument('--split', type=str, dest='split', default='val',
                                 choices=['val', 'test'], help='which point-nav episodes to use in nav tester')        

        self.parser.add_argument('--episode_len', type=int, dest='episode_len', default=10)
        self.parser.add_argument('--truncate_ep', dest='truncate_ep', default=False,
                                  help='truncate episode run in dataloader in order to do only the necessary steps, used in store_episodes_parallel')


        self.parser.add_argument('--local_policy_model', type=str, dest='local_policy_model', default='4plus',
                                choices=['2plus', '4plus'])

        self.parser.add_argument('--scenes_list', nargs='+')
        self.parser.add_argument('--gpu_capacity', type=int, dest='gpu_capacity', default=2)

        self.parser.add_argument('--test_set', type=str, dest='test_set', default='v1', choices=['v1','v2'],
                                help='which set of test episodes to use. v2 is our val-hard.')

        self.parser.add_argument('--occupancy_height_thresh', type=float, dest='occupancy_height_thresh', default=-1.0,
                                help='used when estimating occupancy from depth')

        self.parser.add_argument('--save_img_dir_', dest='save_img_dir_', type=str, default='test_nav_examples/') 

        self.parser.add_argument('--noisy_pose', dest='noisy_pose', default=False, action='store_true',
                                help='adding noise in the simulator pose using models from Neural SLAM')
        self.parser.add_argument('--noise_level', dest='noise_level', type=float, default=1,
                                help='multiplier for the noise added to pose')
        self.parser.add_argument('--noisy_actions', dest='noisy_actions', default=False, action='store_true',
                                help='adding actuation noise using the provided pyrobot_noisy_controls')

        self.parser.add_argument('--fbe', dest='fbe', default=False, action='store_true',
                                help='enables short-term goal selection with Frontier-based search')

        ## RRT-planning specific hyperparams
        self.parser.add_argument('--with_rrt_planning', dest='with_rrt_planning', default=False, action='store_true')
        self.parser.add_argument('--rrt_num_path', dest='rrt_num_path', type=int, default=10,
                                help='how many paths for RRT to generate every iteration')
        self.parser.add_argument('--expand_dis', dest='expand_dis', type=int, default=5,
                                help='expand distance for one step, so essentially number of pixels between two nodes in rrt')
        self.parser.add_argument('--reach_horizon', dest='reach_horizon', type=int, default=10,
                                help='number of nodes we evaluate the reachability over, i.e. horizon')
        self.parser.add_argument('--rrt_max_iters', dest='rrt_max_iters', type=int, default=2500,
                                help='number of rrt iterations for each time-step')
        self.parser.add_argument('--rrt_path_metric', dest='rrt_path_metric', type=str, default='reachability', choices=['reachability', 'shortest'],
                                help='metric to evaluate rrt path, shortest or based on reachability')
        self.parser.add_argument('--goal_sample_rate', dest='goal_sample_rate', type=int, default=20,
                                help='how often RRT samples a node towards the direction of the node')
        self.parser.add_argument('--connect_circle_dist', dest='connect_circle_dist', type=int, default=20,
                                help='range around node to look for connections')
        self.parser.add_argument('--rrt_straight_line', dest='rrt_straight_line', default=False, action='store_true',
                                help='use straight line shortcut when goal is reachable')
        
        self.parser.add_argument('--a_1', type=float, dest='a_1', default=0.1,
                                 help='hyperparameter for scaling the standard deviation')
        self.parser.add_argument('--a_2', type=float, dest='a_2', default=0.0,
                                 help='hyperparameter for scaling the path distance')        
        self.parser.add_argument('--a_3', type=float, dest='a_3', default=0.0,
                                 help='hyperparameter for scaling the angle difference (inertia)')

        self.parser.add_argument('--exploration', dest='exploration', default=False, action='store_true',
                                help='exploration flag, rrt returns multiple node paths within horizon') 