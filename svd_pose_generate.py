import os

import torch
import numpy as np
from pyntcloud import PyntCloud
import pandas as pd
import math

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




# ## 先拿几个样本试一下，比如2个飞机，两个其他类别，看一下这样是不是真的可以对齐？ 哇，基本是可以对齐的。
# inputa = '02691156-12991e9529a2b2bf9ac9930d2147598f.npy'
# inputb = '02691156-4100df683795dfa1f95dfd5eb5f06d19.npy'
# inputc = '02691156-74797de431f83991bc0909d98a1ff2b4.npy'
#
# inputs = [inputa, inputb, inputc]
# for sam in inputs:
#     input = torch.from_numpy(np.load(sam))[:, :3].unsqueeze(0)  # 8192 * 3
#     points = global_transform(input, 32)[0]
#     save_name = 'svdaligned-' + sam
#     # d = {'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]}
#     # cloud = PyntCloud(pd.DataFrame(data=d))
#     # save_name = save_name.replace('.npy', '.ply')
#     # cloud.to_file(save_name)
#     ## 下面的形式没办法mesh lab 可视化
#     save_name = 'svdaligned-' + sam
#     np.save(save_name, np.array(points))

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

def corrupt_rotate_360(pointcloud):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    pointcloud = np.array(pointcloud)
    angle_clip = math.pi
    angle_clip = angle_clip
    angles = np.random.uniform(-angle_clip, angle_clip, size=(3))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return torch.from_numpy(np.dot(pointcloud, R))


root = '/home/yabin/syn_project/point_cloud/CorruptedAE/data/ShapeNet55-34/shapenet_pc_masksurf_with_normal'
# target_root = '/home/yabin/syn_project/point_cloud/CorruptedAE/data/ShapeNet55-34/shapenet_svd_aligned_pc'
target_root = '/home/yabin/syn_project/point_cloud/CorruptedAE/data/ShapeNet55-34/shapenet_rand_pose_pc'
os.makedirs(target_root)

file_list = os.listdir(root)
for file_item in file_list:
    file_dir = os.path.join(root, file_item)
    input = torch.from_numpy(np.load(file_dir))[:, :3]   #  8192 * 3
    ########## for SVD-Pose
    # input = _pc_normalize(input).unsqueeze(0)
    # points = global_transform(input, 32)[0]
    ########## For Random-Pose
    input = _pc_normalize(input)
    points = corrupt_rotate_360(input)
    save_name = os.path.join(target_root, file_item)
    np.save(save_name, np.array(points))


