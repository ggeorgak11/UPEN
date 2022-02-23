## Uncertainty Driven Planner for Exploration and Navigation
G.Georgakis, B.Bucher, A.Arapin, K.Schmeckpeper, N.Matni, K.Daniilidis,

International Conference on Robotics and Automation (ICRA) 2022

### Dependencies
```
pip install -r requirements.txt
```
[Habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim) need to be installed before using our code. We build our method on the latest stable versions for both, so use `git checkout tags/v0.1.7` before installation. Follow the instructions in their corresponding repositories to install them on your system. Note that our code expects that habitat-sim is installed with the flag `--with-cuda`. 


### Data
We use the Matterport3D (MP3D) dataset (the habitat subset and not the entire Matterport3D) for our experiments. Follow the instructions in the [habitat-lab](https://github.com/facebookresearch/habitat-lab) repository regarding downloading the data and the dataset folder structure. In addition we provide the following:

- [MP3D Scene Pclouds](https://drive.google.com/file/d/1u4SKEYs4L5RnyXrIX-faXGU1jc16CTkJ/view): An .npz file for each scene that we generated and that contains the 3D point cloud with semantic category labels (40 MP3D categories). This was done for our convenience because the semantic.ply files for each scene provided with the dataset contain instance labels. The folder containing the .npz files should be under `/data/scene_datasets/mp3d`.
- [Episodes for MP3D Val-Hard](https://drive.google.com/drive/folders/1DUNx8HaeRBv48vPn5NSmIhSeAr-4HiAO?usp=sharing): The test episodes (v2) we generated to evaluate our method as described in the paper. These should be under `/data/datasets/pointnav/mp3d`.


### Trained Models
We provide the trained occupancy map predictor ensemble [here](https://drive.google.com/drive/folders/1aDZVpRLKk1RTYZLeGquG-7aFAUWWpW27?usp=sharing).


### Instructions
Here we provide instructions on how to use our code. It is advised to set up the root_path (directory that includes habitat-lab), log_dir, and paths to data folders and models before-hand in the `train_options.py`.


#### Testing on our episodes
Testing requires a pretrained DDPPO model available [here](https://dl.fbaipublicfiles.com/habitat/data/baselines/v1/ddppo/ddppo-models/gibson-4plus-mp3d-train-val-test-resnet50.pth). Place it under root_path/local_policy_models/. To run a point-goal navigation evaluation of our method on a scene:
```
python main.py --name test_pointnav_exp --ensemble_dir path/to/ensemble/dir --root_path /path/to/dir/containing/habitat-lab --log_dir /path/to/logs --scenes_list 2azQ1b91cZZ --gpu_capacity 1 --with_rrt_planning --test_set v2 
```
To store visualizations during a test run use `--save_nav_images`.

For the exploration task use `--exploration`, `--test_set v1`, and `--max_steps 1000`


#### Generating training data
We provide code to generate your own training examples:
```
python store_episodes_parallel.py --gpu_capacity 1 --scenes_list HxpKQynjfin --episodes_save_dir /path/to/save/dir/ --root_path /path/to/dir/containing/habitat-lab --episode_len 10
```


#### Training the occupancy map predictor models
If you wish to train your own ensemble, first generate the training data, and then each model in the ensemble can be trained separately:
```
python main.py --name train_map_pred_0 --num_workers 4 --batch_size 4 --map_loss_scale 1 --is_train --log_dir /path/to/logs --root_path /path/to/dir/containing/habitat-lab --stored_episodes_dir /path/to/generated/data/ --dataset_percentage 0.7
```


#### Generating point-nav episodes
We provide the script we used to generate our own test episodes:
```
python pointnav_generator.py 
```

Note that there are dedicated options lists in `store_episodes_parallel.py` and `pointnav_generator.py`.

