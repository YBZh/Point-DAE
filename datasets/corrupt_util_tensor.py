import numpy as np
import math
import random
import torch
import ipdb

def _pc_normalize(pc):
    """
    Normalize the point cloud to a unit sphere
    :param pc: input point cloud
    :return: normalized point cloud
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def _shuffle_pointcloud(pcd):
    """
    Shuffle the points
    :param pcd: input point cloud
    :return: shuffled point clouds
    """
    idx = np.random.rand(pcd.shape[0], 1).argsort(axis=0)
    return np.take_along_axis(pcd, idx, axis=0)


def _gen_random_cluster_sizes(num_clusters, total_cluster_size):
    """
    Generate random cluster sizes
    :param num_clusters: number of clusters
    :param total_cluster_size: total size of all clusters
    :return: a list of each cluster size
    """
    rand_list = np.random.randint(num_clusters, size=total_cluster_size)
    cluster_size_list = [sum(rand_list == i) for i in range(num_clusters)]
    return cluster_size_list


def _sample_points_inside_unit_sphere(number_of_particles):
    """
    Uniformly sample points in a unit sphere
    :param number_of_particles: number of points to sample
    :return: sampled points
    """
    radius = np.random.uniform(0.0, 1.0, (number_of_particles, 1))
    radius = np.power(radius, 1 / 3)
    costheta = np.random.uniform(-1.0, 1.0, (number_of_particles, 1))
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2 * np.pi, (number_of_particles, 1))
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.concatenate([x, y, z], axis=1)


def corrupt_scale_nonorm(pointcloud, center, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: B * GroupNum * GroupSize * 3
    :param center:     B * GroupNum * 3
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        batch_size = pointcloud[0].size(0)
        device = pointcloud[0].device
    else:
        batch_size = pointcloud.size(0)
        device = pointcloud.device
    s = [1.6, 1.7, 1.8, 1.9, 2.0][level]  ## this is adopted in initial experiments.
    # s = 2.0
    xyz = torch.FloatTensor(batch_size, 1, 1, 3).uniform_(1. / s, s).to(device)
    xyz_center = xyz.squeeze(1)
    if isinstance(pointcloud, list):
        new_list_pointcloud, new_list_center = [], []
        for i in range(len(pointcloud)):
            new_list_pointcloud.append(pointcloud[i] * xyz)
            new_list_center.append(center[i] * xyz_center)
        return new_list_pointcloud, new_list_center
    else:
        return pointcloud * xyz, center * xyz_center



def corrupt_tranlate(pointcloud, center, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: B * GroupNum * GroupSize * 3
    :param center:     B * GroupNum * 3
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        batch_size = pointcloud[0].size(0)
        device = pointcloud[0].device
    else:
        batch_size = pointcloud.size(0)
        device = pointcloud.device
    s = [0.1, 0.2, 0.3, 0.4, 0.5][level]
    # s = 0.5
    xyz = torch.FloatTensor(batch_size, 1, 1, 3).uniform_(-s, s).to(device)
    xyz_center = xyz.squeeze(1)
    if isinstance(pointcloud, list):
        new_list_pointcloud, new_list_center = [], []
        for i in range(len(pointcloud)):
            new_list_pointcloud.append(pointcloud[i] * xyz)
            new_list_center.append(center[i] * xyz_center)
        return new_list_pointcloud, new_list_center
    else:
        return pointcloud * xyz, center * xyz_center


def corrupt_jitter(pointcloud, center, level):
    """
    Jitter the input point cloud
    :param pointcloud: B * GroupNum * GroupSize * 3
    :param center:     B * GroupNum * 3
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        device = pointcloud[0].device
    else:
        device = pointcloud.device
    sigma = 0.01 * (level + 1)
    if isinstance(pointcloud, list):
        new_list_pointcloud, new_list_center = [], []
        for i in range(len(pointcloud)):
            new_list_pointcloud.append(pointcloud[i] + sigma * torch.normal(0,1, size=pointcloud[i].size()).to(device))
            new_list_center.append(center[i] + sigma * torch.normal(0,1, size=center[i].size()).to(device))
        return new_list_pointcloud, new_list_center
    else:
        return pointcloud + sigma * torch.normal(0,1, size=pointcloud.size()).to(device), center + sigma * torch.normal(0,1, size=center.size()).to(device)


def corrupt_rotate_360(pointcloud, center, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: B * GroupNum * GroupSize * 3
    :param center:     B * GroupNum * 3
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        batch_size = pointcloud[0].size(0)
        device = pointcloud[0].device
    else:
        batch_size = pointcloud.size(0)
        device = pointcloud.device
    if level == None:
        level = random.random() * 4  # [0,4]
    angle_clip = math.pi
    angle_clip = angle_clip / 5 * (level + 1)
    angles = torch.FloatTensor(batch_size, 3).uniform_(-angle_clip, angle_clip)
    x = torch.eye(3)  # 创建对角矩阵n*n
    Rx = x.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    Rx[:, 0, 1, 1] = torch.cos(angles[:, 0])
    Rx[:, 0, 1, 2] = -torch.sin(angles[:, 0])
    Rx[:, 0, 2, 1] = torch.sin(angles[:, 0])
    Rx[:, 0, 2, 2] = torch.cos(angles[:, 0])
    Rx = Rx.to(device)

    y = torch.eye(3)  # 创建对角矩阵n*n
    Ry = y.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    # Ry = torch.eye(batch_size, 1, 3, 3).float()
    Ry[:, 0, 0, 0] = torch.cos(angles[:, 1])
    Ry[:, 0, 0, 2] = torch.sin(angles[:, 1])
    Ry[:, 0, 2, 0] = -torch.sin(angles[:, 1])
    Ry[:, 0, 2, 2] = torch.cos(angles[:, 1])
    Ry = Ry.to(device)

    z = torch.eye(3)  # 创建对角矩阵n*n
    Rz = z.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    # Rz = torch.eye(batch_size, 1, 3, 3).float()
    Rz[:, 0, 0, 0] = torch.cos(angles[:, 2])
    Rz[:, 0, 0, 1] = -torch.sin(angles[:, 2])
    Rz[:, 0, 1, 0] = torch.sin(angles[:, 2])
    Rz[:, 0, 1, 1] = torch.cos(angles[:, 2])
    Rz = Rz.to(device)

    R = torch.matmul(Rz, torch.matmul(Ry, Rx))
    R_center = R.squeeze(1)
    if isinstance(pointcloud, list):
        new_list_pointcloud, new_list_center = [], []
        for i in range(len(pointcloud)):
            new_list_pointcloud.append(torch.matmul(pointcloud[i], R))
            new_list_center.append(torch.matmul(center[i], R_center))
        return new_list_pointcloud, new_list_center
    else:
        return torch.matmul(pointcloud, R), torch.matmul(center, R_center)

def corrupt_rotate_z_360(pointcloud, center, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        batch_size = pointcloud[0].size(0)
        device = pointcloud[0].device
    else:
        batch_size = pointcloud.size(0)
        device = pointcloud.device
    if level == None:
        level = random.random() * 4  # [0,4]
    angle_clip = math.pi
    # angle_clip = angle_clip / 5 * (level + 1)
    angles = torch.FloatTensor(batch_size, 3).uniform_(-angle_clip, angle_clip)
    # x = torch.eye(3)  # 创建对角矩阵n*n
    # Rx = x.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    # Rx[:, 0, 1, 1] = torch.cos(angles[:, 0])
    # Rx[:, 0, 1, 2] = -torch.sin(angles[:, 0])
    # Rx[:, 0, 2, 1] = torch.sin(angles[:, 0])
    # Rx[:, 0, 2, 2] = torch.cos(angles[:, 0])
    # Rx = Rx.to(device)

    # y = torch.eye(3)  # 创建对角矩阵n*n
    # Ry = y.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    # # Ry = torch.eye(batch_size, 1, 3, 3).float()
    # Ry[:, 0, 0, 0] = torch.cos(angles[:, 1])
    # Ry[:, 0, 0, 2] = torch.sin(angles[:, 1])
    # Ry[:, 0, 2, 0] = -torch.sin(angles[:, 1])
    # Ry[:, 0, 2, 2] = torch.cos(angles[:, 1])
    # Ry = Ry.to(device)

    z = torch.eye(3)  # 创建对角矩阵n*n
    Rz = z.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    # Rz = torch.eye(batch_size, 1, 3, 3).float()
    Rz[:, 0, 0, 0] = torch.cos(angles[:, 2])
    Rz[:, 0, 0, 1] = -torch.sin(angles[:, 2])
    Rz[:, 0, 1, 0] = torch.sin(angles[:, 2])
    Rz[:, 0, 1, 1] = torch.cos(angles[:, 2])
    Rz = Rz.to(device)

    R = Rz
    R_center = R.squeeze(1)
    if isinstance(pointcloud, list):
        new_list_pointcloud, new_list_center = [], []
        for i in range(len(pointcloud)):
            new_list_pointcloud.append(torch.matmul(pointcloud[i], R))
            new_list_center.append(torch.matmul(center[i], R_center))
        return new_list_pointcloud, new_list_center
    else:
        return torch.matmul(pointcloud, R), torch.matmul(center, R_center)


def corrupt_reflection(pointcloud, center, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: B * GroupNum * GroupSize * 3
    :param center:     B * GroupNum * 3
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        batch_size = pointcloud[0].size(0)
        device = pointcloud[0].device
    else:
        batch_size = pointcloud.size(0)
        device = pointcloud.device
    choice = np.array([1, -1])
    reflection = np.random.choice(choice, size=(batch_size, 3))
    reflection = torch.from_numpy(reflection)
    x = torch.eye(3)  # 创建对角矩阵n*n
    Rx = x.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    Rx[:, 0, 0, 0] = reflection[:, 0]
    Rx = Rx.to(device)

    y = torch.eye(3)  # 创建对角矩阵n*n
    Ry = y.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    Ry[:, 0, 1, 1] = reflection[:, 1]
    Ry = Ry.to(device)

    z = torch.eye(3)  # 创建对角矩阵n*n
    Rz = z.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    Rz[:, 0, 0, 0] = reflection[:, 2]
    Rz = Rz.to(device)

    R = torch.matmul(Rz, torch.matmul(Ry, Rx))
    R_center = R.squeeze(1)
    if isinstance(pointcloud, list):
        new_list_pointcloud, new_list_center = [], []
        for i in range(len(pointcloud)):
            new_list_pointcloud.append(torch.matmul(pointcloud[i], R))
            new_list_center.append(torch.matmul(center[i], R_center))
        return new_list_pointcloud, new_list_center
    else:
        return torch.matmul(pointcloud, R), torch.matmul(center, R_center)

    # Rx = np.array([[reflection[0], 0, 0],
    #                [0, 1, 0],
    #                [0, 0, 1]])
    # Ry = np.array([[1, 0, 0],
    #                [0, reflection[1], 0],
    #                [0, 0, 1]])
    # Rz = np.array([[1, 0, 0],
    #                [0, 1, 0],
    #                [0, 0, reflection[2]]])
    # R = np.dot(Rz, np.dot(Ry, Rx))
    # return np.dot(pointcloud, R)

def corrupt_shear(pointcloud, center, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: B * GroupNum * GroupSize * 3
    :param center:     B * GroupNum * 3
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        batch_size = pointcloud[0].size(0)
        device = pointcloud[0].device
    else:
        batch_size = pointcloud.size(0)
        device = pointcloud.device
    if level == None:
        level = random.random() * 4  # [0,4]
    shear_clip = (level + 1) * 0.1  # [0.1, 0.5]
    # shear_clip = 0.5
    shear = np.random.uniform(-shear_clip, shear_clip, size=(batch_size, 6))
    shear = torch.from_numpy(shear)
    x = torch.eye(3)  # 创建对角矩阵n*n
    Rx = x.expand((batch_size, 1, 3, 3)).clone().float()  # 扩展维度到b维
    Rx[:, 0, 0, 1] = shear[:, 0]
    Rx[:, 0, 0, 2] = shear[:, 1]
    Rx[:, 0, 1, 0] = shear[:, 2]
    Rx[:, 0, 1, 2] = shear[:, 3]
    Rx[:, 0, 2, 0] = shear[:, 4]
    Rx[:, 0, 2, 1] = shear[:, 5]
    R = Rx.to(device)
    R_center = R.squeeze(1)
    if isinstance(pointcloud, list):
        new_list_pointcloud, new_list_center = [], []
        for i in range(len(pointcloud)):
            new_list_pointcloud.append(torch.matmul(pointcloud[i], R))
            new_list_center.append(torch.matmul(center[i], R_center))
        return new_list_pointcloud, new_list_center
    else:
        return torch.matmul(pointcloud, R), torch.matmul(center, R_center)

    # Rz = np.array([[1, shear[0], shear[1]],
    #                [shear[2], 1, shear[3]],
    #                [shear[4], shear[5], 1]])
    # return np.dot(pointcloud, Rz)

def corrupt_shear_small(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    shear_clip = (level + 1) * 0.02  # [0.02, 0.1]
    shear = np.random.uniform(-shear_clip, shear_clip, size=(6))

    Rz = np.array([[1, shear[0], shear[1]],
                   [shear[2], 1, shear[3]],
                   [shear[4], shear[5], 1]])
    return np.dot(pointcloud, Rz)

## this should be conducted after the sampling.
# def corrupt_dropout_global(pointcloud, level):
#     """
#     Drop random points globally
#     :param pointcloud: input point cloud
#     :param level: severity level
#     :return: corrupted point cloud
#     """
#     drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
#     num_points = pointcloud.shape[0]
#     pointcloud = _shuffle_pointcloud(pointcloud)
#     pointcloud = pointcloud[:int(num_points * (1 - drop_rate)), :]
#     return pointcloud

# this should be conducted after the sampling, so we place it in the forward process.
def dropout_global_random(pointcloud, center, level=None):
    """
    Drop random points globally
    :param pointcloud: B * GroupNum * GroupSize * 3
    :param center:     B * GroupNum * 3
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        batch_size = pointcloud[0].size(0)
        device = pointcloud[0].device
    else:
        batch_size = pointcloud.size(0)
        device = pointcloud.device
    if level == None:
        levels = [0, 1, 2, 3, 4]
        level = random.choice(levels)
    drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]

    if isinstance(pointcloud, list):
        raise NotImplementedError
        # new_list_pointcloud, new_list_center = [], []
        # for i in range(len(pointcloud)):
        #     new_list_pointcloud.append(torch.matmul(pointcloud[i], R))
        #     new_list_center.append(torch.matmul(center[i], R_center))
        # return new_list_pointcloud, new_list_center
    else:
        num_samples = pointcloud.size(0)
        num_groups = pointcloud.size(1)
        num_points = pointcloud.size(2)
        inx = torch.rand(num_samples, num_groups, num_points, 1).argsort(2).to(pointcloud.device)  # random index
        pointcloud = torch.take_along_dim(pointcloud, inx, dim=2)  # re-arange with random index
        pointcloud = pointcloud[:, :, :int(num_points * (1 - drop_rate)), :]  ### less point in each patch.
        return pointcloud, center


    # num_samples = pointcloud.size(0)
    # num_points = pointcloud.size(1)
    # inx = torch.rand(num_samples, num_points, 1).argsort(1).to(pointcloud.device)  # random index
    # pointcloud = torch.take_along_dim(pointcloud, inx, dim=1)  # re-arange with random index
    # pointcloud = pointcloud[:, :int(num_points * (1 - drop_rate)), :]
    # return pointcloud.contiguous()

def corrupt_dropout_local(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    for i in range(num_clusters):
        K = cluster_size_list[i]
        pointcloud = _shuffle_pointcloud(pointcloud)
        dist = np.sum((pointcloud - pointcloud[:1, :]) ** 2, axis=1, keepdims=True)
        idx = dist.argsort(axis=0)[::-1, :]
        pointcloud = np.take_along_axis(pointcloud, idx, axis=0)
        num_points -= K
        pointcloud = pointcloud[:num_points, :]
    return pointcloud


def corrupt_add_global(pointcloud, level):
    """
    Add random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    npoints = int(num_points * (level + 1) * 0.1)
    additional_pointcloud = _sample_points_inside_unit_sphere(npoints)
    pointcloud = np.concatenate([pointcloud, additional_pointcloud[:npoints]], axis=0)
    return pointcloud


def corrupt_add_local(pointcloud, center, level):
    """
    Randomly add random noise within a point patches.
    :param pointcloud: B * GroupNum * GroupSize * 3
    :param center:     B * GroupNum * 3
    :param level: severity level
    :return: corrupted point cloud
    """
    if isinstance(pointcloud, list):
        batch_size = pointcloud[0].size(0)
        device = pointcloud[0].device
    else:
        batch_size = pointcloud.size(0)
        device = pointcloud.device
    if level == None:
        levels = [0, 1, 2, 3, 4]
        level = random.choice(levels)
    # drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
    add_rate = (level + 1) * 0.1

    if isinstance(pointcloud, list):
        raise NotImplementedError
        # new_list_pointcloud, new_list_center = [], []
        # for i in range(len(pointcloud)):
        #     new_list_pointcloud.append(torch.matmul(pointcloud[i], R))
        #     new_list_center.append(torch.matmul(center[i], R_center))
        # return new_list_pointcloud, new_list_center
    else:
        num_samples = pointcloud.size(0)
        num_groups = pointcloud.size(1)
        num_points = pointcloud.size(2)
        ### random select patch points and then add random noise to them, then concate it with vanilla patch points
        random_noise_num = int(add_rate * num_points)
        Gaussian_std = random.uniform(0.075, 0.125)

        inx = torch.rand(num_samples, num_groups, num_points, 1).argsort(2).to(pointcloud.device)  # random index
        pointcloud = torch.take_along_dim(pointcloud, inx, dim=2)  # re-arange with random index
        random_patch_points = pointcloud[:, :, :random_noise_num, :].clone()
        random_patch_points = random_patch_points + torch.normal(0, Gaussian_std, random_patch_points.size()).to(pointcloud.device)
        pointcloud = torch.cat((pointcloud, random_patch_points), dim=2)
        return pointcloud, center

    # return pointcloud
    # num_points = pointcloud.shape[0]
    # total_cluster_size = int(num_points * (level + 1) * 0.1)
    # num_clusters = np.random.randint(1, 8)
    # cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    # pointcloud = _shuffle_pointcloud(pointcloud)
    # add_pcd = np.zeros_like(pointcloud)
    # num_added = 0
    # for i in range(num_clusters):
    #     K = cluster_size_list[i]
    #     sigma = np.random.uniform(0.075, 0.125)
    #     add_pcd[num_added:num_added + K, :] = np.copy(pointcloud[i:i + 1, :])
    #     add_pcd[num_added:num_added + K, :] = add_pcd[num_added:num_added + K, :] + sigma * np.random.randn(
    #         *add_pcd[num_added:num_added + K, :].shape)
    #     num_added += K
    # assert num_added == total_cluster_size
    # dist = np.sum(add_pcd ** 2, axis=1, keepdims=True).repeat(3, axis=1)
    # add_pcd[dist > 1] = add_pcd[dist > 1] / dist[dist > 1]  # ensure the added points are inside a unit sphere
    # pointcloud = np.concatenate([pointcloud, add_pcd], axis=0)
    # pointcloud = pointcloud[:num_points + total_cluster_size]
    # return pointcloud

# the Non-uniform density in MetaSet.
# When scanning an object with a LiDAR, the closer it is, the denser the point cloud will be.
# larger gate, more far point will be dropped.
################################################################################################################
##### Specific design for transformer backbones, when drop some points, we randomly duplicate some points to ensure the same point number within a batch.
def density(pc, center, level=None):
    # :param pointcloud: B * GroupNum * GroupSize * 3
    # :param center:     B * GroupNum * 3
    if isinstance(pc, list):
        batch_size = pc[0].size(0)
        device = pc[0].device
    else:
        batch_size = pc.size(0)
        device = pc.device
    if level == None:
        level = random.random() * 4  # [0,4]
    else:
        level = random.random() * level  # [0,level]
    # gate = 1
    # gate = level / 4.0 + 0.1  # [0.1, 1.1]
    gate = level + 1  ## [1,5]
    # v_point = np.random.normal(0, 1, 3)
    v_point = torch.normal(0,1,[3])
    norm = torch.norm(v_point)
    v_point = v_point / norm  # one random point on the sphere surface
    v_point = v_point.to(pc.device)
    # dist = np.sqrt((v_point ** 2).sum()) ## 1, since norm
    max_dist = 2
    min_dist = 0

    if isinstance(pc, list):
        raise NotImplementedError
        # new_list_pointcloud, new_list_center = [], []
        # for i in range(len(pc)):
        #     pass
        #     # new_list_pointcloud.append(torch.matmul(pc[i], R))
        #     # new_list_center.append(torch.matmul(center[i], R_center))
        # return new_list_pointcloud, new_list_center
    else:
        dist = torch.norm(pc - v_point.reshape(1, 1, 1, 3), dim=-1)  ## B * G_num * G_size
        dist = (dist - min_dist) / (max_dist - min_dist)  ## [0,1]
        r_list = torch.rand(pc.shape[:3]).to(pc.device)  ## B * G_num * G_size
        value, index = torch.sort((dist < r_list).long(), -1)
        selected_ind = value * (index + 1)  ### the selected ones.  only the unselected one are set to zero
        selected_ind[selected_ind ==0] = 33 ##
        selected_ind = selected_ind - 1
        pc_expand_center = torch.cat((pc, center.unsqueeze(2)), dim=2)  ## 128*64*32+1* 3
        selected_ind = torch.cat((selected_ind, selected_ind[:,:,:1]), dim=2)
        selected_ind = selected_ind.unsqueeze(-1).expand(pc_expand_center.size())
        ### 找出对应0 的index, 然后随机替换为 1 的index.
        patch_size = pc.size(2)
        return pc_expand_center.gather(2, selected_ind)[:,:,:patch_size,:], center

        # index = index.unsqueeze(-1).expand(pc.size())
        # value = value.unsqueeze(-1).expand(pc.size())
        # tmp_pc = pc[dist < gate * (r_list)] ### some group may be all masked;
        # ########### if some group has been all masked, we re-fill it with patch centers.
        # ########### if part of the group has been masked, duplicate the remaining part to keep the total number unchanged.
        #
        # return tmp_pc, center


# apply the patch dropping as PointMAE (FPS + KNN for patch points, and then randomly drop some patches. )
from knn_cuda import KNN
from pointnet2_ops import pointnet2_utils
knn = KNN(k=32, transpose_mode=True)
def dropout_patch_random(pc_tensor, level=None):
    if level == None:
        level = random.random() * 4  # [0,4]
    prob = level / 10.0 + 0.5  # [0.5, 0.9]
    batch_size, num_points, _ = pc_tensor.shape
    fps_idx = pointnet2_utils.furthest_point_sample(pc_tensor[:, :, :3].contiguous(), 64)
    center = pointnet2_utils.gather_operation(pc_tensor.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()  ## B Gn 3
    # knn to get the neighborhood
    _, idx = knn(pc_tensor, center)  # B G M
    assert idx.size(1) == 64 # num_group
    assert idx.size(2) == 32 # group size
    idx_base = torch.arange(0, batch_size, device=pc_tensor.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    neighborhood = pc_tensor.view(batch_size * num_points, -1)[idx, :]
    # batch_size, self.num_group, self.group_size, 3
    neighborhood = neighborhood.view(batch_size, 64, 32, 3).contiguous()
    group_mask = torch.rand(64) > prob  # the prob here could be introduced randomly.
    # at least one group should be selected.
    if group_mask.sum().item() == 0:
        group_mask[0] = True
    neighborhood = neighborhood[:, group_mask.to(pc_tensor.device)]
    # print(neighborhood.size(1))
    ## here we should at least drop 50% patches.  虽然这样可以用了，但是这样导致其必须在forward 中进行，很丑。 我想把他也放到data processing 中。
    return neighborhood.view(batch_size, -1, 3)




# conducting FPS with cpu is slow, while using GPU leads to some problem with multiprocessing.
# from knn_cuda import KNN
# from pointnet2_ops import pointnet2_utils
# num_group = 32
# group_size = 64
# knn = KNN(k=group_size, transpose_mode=True)
# def dropout_patch_random(pc, level=None):
#     # print(torch.multiprocessing.get_start_method())
#     rand = random.randint(0,7)
#     pc_tensor = torch.from_numpy(pc).unsqueeze(0).float().cuda(rand)
#     batch_size, num_points, _ = pc_tensor.shape
#     fps_idx = pointnet2_utils.furthest_point_sample(pc_tensor[:, :, :3].contiguous(), num_group)
#     center = pointnet2_utils.gather_operation(pc_tensor.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()  ## B Gn 3
#     # knn to get the neighborhood
#     _, idx = knn(pc_tensor, center)  # B G M
#     assert idx.size(1) == num_group # num_group
#     assert idx.size(2) == group_size # group size
#     idx_base = torch.arange(0, batch_size, device=pc_tensor.device).view(-1, 1, 1) * num_points
#     idx = idx + idx_base
#     idx = idx.view(-1)
#     neighborhood = pc_tensor.view(batch_size * num_points, -1)[idx, :]
#     # batch_size, self.num_group, self.group_size, 3
#     neighborhood = neighborhood.view(batch_size, num_group, group_size, 3).contiguous()
#     return np.array(neighborhood.view(-1, 3).cpu())
#




# # From MetaSet, to analyses.
# def p_scan(pc, pixel_size=0.017):
#     pixel = int(2 / pixel_size)
#     rotated_pc = rotate_point_cloud_3d(pc)
#     pc_compress = (rotated_pc[:,2] + 1) / 2 * pixel * pixel + (rotated_pc[:,1] + 1) / 2 * pixel
#     points_list = [None for i in range((pixel + 5) * (pixel + 5))]
#     pc_compress = pc_compress.astype(np.int)
#     for index, point in enumerate(rotated_pc):
#         compress_index = pc_compress[index]
#         if compress_index > len(points_list):
#             print('out of index:', compress_index, len(points_list), point, pc[index], (pc[index] ** 2).sum(), (point ** 2).sum())
#         if points_list[compress_index] is None:
#             points_list[compress_index] = index
#         elif point[0] > rotated_pc[points_list[compress_index]][0]:
#             points_list[compress_index] = index
#     points_list = list(filter(lambda x:x is not None, points_list))
#     points_list = pc[points_list]
#     return points_list


# # From MetaSet, to analyses. This is similar to local dropout and patch drop.
# def drop_hole(pc, p):
#     random_point = np.random.randint(0, pc.shape[0])
#     index = np.linalg.norm(pc - pc[random_point].reshape(1,3), axis=1).argsort()
#     return pc[index[int(pc.shape[0] * p):]]

corruptions = {
    'add_global': corrupt_add_global,
    'add_local': corrupt_add_local,
    'jitter': corrupt_jitter,
    #############
    'translate': corrupt_tranlate,  # different scale across xyz + pc_norm
    'scale_nonorm': corrupt_scale_nonorm,  # different scale across xyz
    'rotate': corrupt_rotate_360,  # prior: only for aug
    'rotate_z': corrupt_rotate_z_360,
    'reflection': corrupt_reflection,  # prior: only for aug
    'shear': corrupt_shear,  # prior: only for aug
    'shear_small': corrupt_shear_small,  # prior: only for aug
    ################
    'dropout_global': dropout_global_random,
    # 'dropout_local': corrupt_dropout_local,
    # 'dropout_patch_pointmae': dropout_patch_random,
    'scan': density,
    # different degree of rotation.
    # 'rotate_level0': corrupt_rotate_360_level0,
    # 'rotate_level1': corrupt_rotate_360_level1,
    # 'rotate_level2': corrupt_rotate_360_level2,
    # 'rotate_level3': corrupt_rotate_360_level3,
    # 'rotate_level4': corrupt_rotate_360_level4
}


affine_corruptions = ['translate', 'scale_nonorm', 'rotate', 'reflection', 'shear']
add_corruptions = ['add_global', 'add_local', 'jitter']
dropout_corruptions = ['dropout_local', 'nonuniform_density']  # 还有两个dropout, 是在forward 中实现的。

def corrupt_data(neighborhood, center, type=['clean']):
    # batch_data: B * N * 3.
    for corruption_item in type:
        if corruption_item == 'clean' or corruption_item == 'Drop-Patch':
        # if type == 'clean':
            pass
        elif corruption_item == 'affine_r3':
            ##  5个里面随机不重复选1-3 个， 然后采用随机顺序。
            numbers = [1,2,3]
            number = random.choice(numbers)
            adopted_affine = random.sample(affine_corruptions, number)
            for affine_corruption_item in adopted_affine:
                # levels = [0, 1, 2, 3, 4]
                # level = random.choice(levels)
                level = 4
                neighborhood, center = corruptions[affine_corruption_item](neighborhood, center, level)
        else:
            # levels = [0,1,2,3,4]
            # level = random.choice(levels)
            # level = 4
            neighborhood, center = corruptions[corruption_item](neighborhood, center, level)
            # return corruptions[type](data_instance, level)
    return neighborhood, center

    # ipdb.set_trace()
    # corrupted_data = []
    # for i in range(batch_data.size(0)):
    #     pcd = np.array(batch_data[i].cpu())
    #     corrupted_pcd = corruptions[type](pcd, level)  ## here the input should be M*3 numpy array.
    #     corrupted_data.append(corrupted_pcd)
    # corrupted_data = np.stack(corrupted_data, axis=0)
    #
    # return torch.from_numpy(corrupted_data)

def PointcloudScale(data_instance, scale_low=2. / 3., scale_high=3. / 2.):
    xyz1 = np.random.uniform(low=scale_low, high=scale_high, size=[3])
    # xyz2 = np.random.uniform(low=-translate_range, high=translate_range, size=[3])
    return data_instance * xyz1

def PointcloudTranslate(data_instance, translate_range=0.2):
    xyz2 = np.random.uniform(low=-translate_range, high=translate_range, size=[3])
    return data_instance + xyz2

# for pre-training: apply the pc_norm(center and normalization), scale_translate, rotate, point_resampling, here.
# ? is there any difference between pre-training and fine-tuning?
# for fine-tuning apply the pc_norm(center and normalization), scale_translate, rotate, point_resampling, here.
def augment_data(data_instance, type=['norm']):
    # batch_data: B * N * 3.
    for corruption_item in type:
        if corruption_item == 'norm':
            data_instance = _pc_normalize(data_instance)
        elif corruption_item == 'clean':
            data_instance = data_instance
        elif corruption_item == 'translate':
            data_instance = PointcloudTranslate(data_instance)
        elif corruption_item == 'scale':
            data_instance = PointcloudScale(data_instance)
        # elif corruption_item == 'rotate_z':
        #     data_instance = corrupt_rotate_z(data_instance)

        else:
            raise NotImplementedError
    return data_instance
