
import numpy as np
import os
import math
import torch
import quaternion
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import torch.nn.functional as F


def add_uniform_noise(tensor, a, b):
    return tensor + torch.FloatTensor(tensor.shape).uniform_(a, b).to(tensor.device)

def add_gaussian_noise(tensor, mean, std):
    return tensor + torch.randn(tensor.size()).to(tensor.device) * std + mean

def euclidean_distance(position_a, position_b):
    return np.linalg.norm(position_b - position_a, ord=2)


def preprocess_img(img, cropSize, pixFormat, normalize):
    img = img.permute(2,0,1).unsqueeze(0).float()
    img = F.interpolate(img, size=cropSize, mode='bilinear', align_corners=True)
    img = img.squeeze(0)
    if normalize:
        img = img / 255.0
    return img


# normalize code from habitat lab:
# obs = (obs - MIN_DEPTH) / (MAX_DEPTH - MIN_DEPTH)
def unnormalize_depth(depth, min, max):
    return (depth * (max - min)) + min


def get_entropy(pred):
    log_predictions = torch.log(pred)
    mul_map = -pred*log_predictions
    return torch.sum(mul_map, dim=2, keepdim=True) # B x T x 1 x cH x cW


def get_sim_location(agent_state):
    x = -agent_state.position[2]
    y = -agent_state.position[0]
    height = agent_state.position[1]
    axis = quaternion.as_euler_angles(agent_state.rotation)[0]
    if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
        o = quaternion.as_euler_angles(agent_state.rotation)[1]
    else:
        o = 2*np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
    if o > np.pi:
        o -= 2 * np.pi
    pose = x, y, o
    return pose, height


def get_rel_pose(pos2, pos1):
    x1, y1, o1 = pos1
    if len(pos2)==2: # if pos2 has no rotation
        x2, y2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        return dx, dy
    else:
        x2, y2, o2 = pos2
        dx = x2 - x1
        dy = y2 - y1
        do = o2 - o1
        if do < -math.pi:
            do += 2 * math.pi
        if do > math.pi:
            do -= 2 * math.pi
        return dx, dy, do

# taken from: https://github.com/devendrachaplot/Neural-SLAM/blob/8876e1eafff5414e92d72347d89488d970a4f6b2/env/habitat/utils/pose.py#L24
def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(o) + dy * np.cos(o)
    global_dy = dx * np.cos(o) - dy * np.sin(o)
    x += global_dy
    y += global_dx
    o += do
    if o > math.pi:
        o -= 2 * math.pi
    return x, y, o


def load_scene_pcloud(preprocessed_scenes_dir, scene_id):
    pcloud_path = preprocessed_scenes_dir+scene_id+'_pcloud.npz'
    if not os.path.exists(pcloud_path):
        raise Exception('Preprocessed point cloud for scene', scene_id,'not found!')

    data = np.load(pcloud_path)
    x = data['x']
    y = data['y']
    z = data['z']
    label_seq = data['label_seq']
    data.close()

    label_seq[ label_seq<0.0 ] = 0.0
    # Convert the labels to the reduced set of categories
    label_seq_spatial = label_seq.copy()
    label_seq_objects = label_seq.copy()
    for i in range(label_seq.shape[0]):
        curr_lbl = label_seq[i,0]
        label_seq_spatial[i] = viz_utils.label_conversion_40_3[curr_lbl]
    return (x, y, z), label_seq_spatial


def depth_to_3D(depth_obs, img_size, xs, ys, inv_K):

    depth = depth_obs[...,0].reshape(1, img_size[0], img_size[1])

    # Unproject
    # negate depth as the camera looks along -Z
    # SPEEDUP - create ones in constructor
    xys = torch.vstack((torch.mul(xs, depth) , torch.mul(ys, depth), -depth, torch.ones(depth.shape, device='cuda'))) # 4 x 128 x 128
    xys = xys.reshape(4, -1)
    xy_c0 = torch.matmul(inv_K, xys)

    # SPEEDUP - don't allocate new memory, manipulate existing shapes
    local3D = torch.zeros((xy_c0.shape[1],3), dtype=torch.float32, device='cuda')
    local3D[:,0] = xy_c0[0,:]
    local3D[:,1] = xy_c0[1,:]
    local3D[:,2] = xy_c0[2,:]

    return local3D



# Taken from: https://github.com/pytorch/pytorch/issues/35674
def unravel_index(indices, shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        unravel coordinates, (*, N, D).
    """

    shape = torch.tensor(shape)
    indices = indices % shape.prod()  # prevent out-of-bounds indices

    coord = torch.zeros(indices.size() + shape.size(), dtype=int)

    for i, dim in enumerate(reversed(shape)):
        coord[..., i] = indices % dim
        indices = indices // dim

    return coord.flip(-1)
