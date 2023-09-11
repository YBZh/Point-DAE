import os

import torch
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
import math
from datasets.corrupt_util import corrupt_scale_nonorm, corrupt_tranlate, corrupt_jitter, corrupt_rotate_360, corrupt_rotate_z_360, corrupt_shear, \
corrupt_reflection, dropout_global_random, corrupt_dropout_local, corrupt_add_global, corrupt_add_local, density, dropout_patch_random
## dropout_global_random and dropout_patch_random are conducted on cuda device.

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.size()
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10

    centroid = torch.mean(xyz, dim=1, keepdim=True)  # [B, 1, C]
    dist = torch.sum((xyz - centroid) ** 2, -1)
    farthest = torch.max(dist, -1)[1]

    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    # print(idx.shape)
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def global_transform(points, npoints):
    # Points: B N C
    device = points.device
    # points = points.permute(0, 2, 1)
    idx = farthest_point_sample(points, npoints)  # input BNC
    centroids = index_points(points, idx)   #[B, S, C]
    # U, S, V = batch_svd(centroids)
    U, S, V = torch.svd(points)
    # if train == True:
    #     index = torch.randint(2, (points.size(0), 1, 3)).type(torch.FloatTensor).cuda()
    #     V_ = V * index
    #     V -= 2 * V_
    # else:
    key_p = centroids[:, 0, :].unsqueeze(1)
    angle = torch.matmul(key_p, V)
    index = torch.le(angle, 0).type(torch.FloatTensor).to(device)
    V_ = V * index
    V -= 2 * V_
    # print(V.size()) ## 1 * 3 * 3
    xyz = torch.matmul(points, V)  #.permute(0, 2, 1)
    return xyz

def _pc_normalize(pc):
    """
    Normalize the point cloud to a unit sphere
    :param pc: input point cloud
    :return: normalized point cloud
    """
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1)))
    pc = pc / m
    return pc

def random_sample(pc, num):
    ## input should be numpy arrays.
    if pc.shape[0] >= num:
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
    else:
        gap = num - pc.shape[0]
        indices = np.random.choice(pc.shape[0], gap, replace=True)
        pc = np.vstack((pc, pc[indices]))
        permutation = np.arange(pc.shape[0])
        np.random.shuffle(permutation)
        pc = pc[permutation[:num]]
    return pc

## 先拿几个样本试一下，比如2个飞机，两个其他类别，看一下这样是不是真的可以对齐？ 哇，基本是可以对齐的。
sam = '02691156-4100df683795dfa1f95dfd5eb5f06d19.npy'

input = torch.from_numpy(np.load(sam))[:, :3]
input = _pc_normalize(input).unsqueeze(0)  # 8192 * 3
input_vanilla = global_transform(input, 32)[0]  ## good pose for visualization
## no corruption.
input_unsquee = input_vanilla.unsqueeze(0)
idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
points = index_points(input_unsquee, idx)[0]  # [B, S, C]
d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'nocorruption-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)
# scale
input = np.array(input_vanilla)
input = corrupt_scale_nonorm(input, 4)
points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]
d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'scale-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# affinity
input = np.array(input_vanilla)
input = corrupt_reflection(input, 2)
input = corrupt_scale_nonorm(input, 4)
input = corrupt_tranlate(input, 2)
input = corrupt_shear(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'affinity-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)


# translate
input = np.array(input_vanilla)
input = corrupt_tranlate(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'translate-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# jitter
input = np.array(input_vanilla)
input = corrupt_jitter(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'jitter-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# rotation_360
input = np.array(input_vanilla)
input = corrupt_rotate_360(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'rotation-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# rotationz_360
input = np.array(input_vanilla)
input = corrupt_rotate_z_360(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'rotationz-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# corrupt_dropout_local
input = np.array(input_vanilla)
input = corrupt_dropout_local(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'dropout_local-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# corrupt_dropout_local
input = np.array(input_vanilla)
input = corrupt_add_global(input, 1)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'add_global-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# corrupt_add_local
input = np.array(input_vanilla)
input = corrupt_add_local(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'add_local-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# density
input = np.array(input_vanilla)
input = density(input, 4)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'density-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# corrupt_shear
input = np.array(input_vanilla)
input = corrupt_shear(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]
d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'shear-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# corrupt_reflection
input = np.array(input_vanilla)
input = corrupt_reflection(input, 2)

points = random_sample(input, 1024)
# input_unsquee = torch.from_numpy(input).unsqueeze(0)
# idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
# points = index_points(input_unsquee, idx)[0]  # [B, S, C]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'reflection-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# dropout_global_random
input_unsquee = input_vanilla.unsqueeze(0)
idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
points = index_points(input_unsquee, idx)  # [B, S, C]

points = dropout_global_random(points.cuda(), 2).cpu()[0]

d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'dropout_global-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# dropout_patch_random
input_unsquee = input_vanilla.unsqueeze(0)
idx = farthest_point_sample(input_unsquee, 1024)  # input BNC
points = index_points(input_unsquee, idx)  # [B, S, C]

points = dropout_patch_random(points.cuda(), 2).cpu()[0]
d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
cloud = PyntCloud(pd.DataFrame(data=d))
save_name = 'dropout_patch-' + sam
save_name = save_name.replace('.npy', '.ply')
cloud.to_file(save_name)

# points = global_transform(input, 32)[0]
# save_name = 'svdaligned-' + sam
# d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
# cloud = PyntCloud(pd.DataFrame(data=d))
# save_name = save_name.replace('.npy', '.ply')
# cloud.to_file(save_name)
## 下面的形式没办法mesh lab 可视化
# save_name = 'svdaligned-' + sam
# np.save(save_name, np.array(points))





# root = '/home/yabin/syn_project/point_cloud/CorruptedAE/data/ShapeNet55-34/shapenet_pc_masksurf_with_normal'
# # target_root = '/home/yabin/syn_project/point_cloud/CorruptedAE/data/ShapeNet55-34/shapenet_svd_aligned_pc'
# target_root = '/home/yabin/syn_project/point_cloud/CorruptedAE/data/ShapeNet55-34/shapenet_rand_pose_pc'
# os.makedirs(target_root)
#
# file_list = os.listdir(root)
# for file_item in file_list:
#     file_dir = os.path.join(root, file_item)
#     input = torch.from_numpy(np.load(file_dir))[:, :3]   #  8192 * 3
#     # input = _pc_normalize(input).unsqueeze(0)
#     # points = global_transform(input, 32)[0]
#     input = _pc_normalize(input)
#     points = corrupt_rotate_360(input)
#     # print(file_item)
#     save_name = os.path.join(target_root, file_item)
#     np.save(save_name, np.array(points))


