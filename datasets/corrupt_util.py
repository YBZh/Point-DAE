import numpy as np
import math
import random
import torch


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

# after scale, a PC norm is applied.  bad results, we should not apply the PC norm after scale.
def corrupt_scale(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    s = 2.0
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    return _pc_normalize(np.multiply(pointcloud, xyz).astype('float32'))

def corrupt_scale_single(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    xyz = np.random.uniform(low=1. / s, high=s, size=[1])
    return _pc_normalize(np.multiply(pointcloud, xyz).astype('float32'))

def corrupt_scale_nonorm_2p(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    s = 2.0
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    return np.multiply(pointcloud, xyz).astype('float32')

def corrupt_scale_nonorm_1p5(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    s = 1.5
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    return np.multiply(pointcloud, xyz).astype('float32')

def corrupt_scale_nonorm_4(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    s = 4.0
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    return np.multiply(pointcloud, xyz).astype('float32')

def corrupt_scale_nonorm_10(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    s = 10.0
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    return np.multiply(pointcloud, xyz).astype('float32')

def corrupt_tranlate(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [0.1, 0.2, 0.3, 0.4, 0.5][level]
    s = 0.5
    xyz = np.random.uniform(low=-s, high=s, size=[3])
    return (pointcloud + xyz).astype('float32')

def corrupt_tranlate_tiny(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [0.1, 0.2, 0.3, 0.4, 0.5][level]
    s = 0.1
    xyz = np.random.uniform(low=-s, high=s, size=[3])
    return (pointcloud + xyz).astype('float32')

def corrupt_tranlate_middle(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [0.1, 0.2, 0.3, 0.4, 0.5][level]
    s = 0.3
    xyz = np.random.uniform(low=-s, high=s, size=[3])
    return (pointcloud + xyz).astype('float32')


def corrupt_tranlate_too_large(pointcloud, level):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # s = [0.1, 0.2, 0.3, 0.4, 0.5][level]
    s = 0.8
    xyz = np.random.uniform(low=-s, high=s, size=[3])
    return (pointcloud + xyz).astype('float32')

def corrupt_jitter(pointcloud, level=None):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    sigma = 0.01 * (level + 1)
    N, C = pointcloud.shape
    pointcloud = pointcloud + sigma * np.random.randn(N, C)
    return pointcloud

def corrupt_jitter_p01(pointcloud, level=None):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    sigma = 0.01
    N, C = pointcloud.shape
    pointcloud = pointcloud + sigma * np.random.randn(N, C)
    return pointcloud

def corrupt_jitter_p03(pointcloud, level=None):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    sigma = 0.03
    N, C = pointcloud.shape
    pointcloud = pointcloud + sigma * np.random.randn(N, C)
    return pointcloud

def corrupt_jitter_p05(pointcloud, level=None):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    sigma = 0.05
    N, C = pointcloud.shape
    pointcloud = pointcloud + sigma * np.random.randn(N, C)
    return pointcloud

def corrupt_jitter_p1(pointcloud, level=None):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    sigma = 0.1
    N, C = pointcloud.shape
    pointcloud = pointcloud + sigma * np.random.randn(N, C)
    return pointcloud

def corrupt_rotate_360(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    angle_clip = math.pi
    # angle_clip = angle_clip / 5 * (level + 1)
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
    return np.dot(pointcloud, R)

def corrupt_rotate_360_level0(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # if level == None:
    #     level = random.random() * 4  # [0,4]
    level = 0
    angle_clip = math.pi
    angle_clip = angle_clip / 5 * (level + 1)
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
    return np.dot(pointcloud, R)

def corrupt_rotate_360_level1(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # if level == None:
    #     level = random.random() * 4  # [0,4]
    level = 1
    angle_clip = math.pi
    angle_clip = angle_clip / 5 * (level + 1)
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
    return np.dot(pointcloud, R)

def corrupt_rotate_360_level2(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # if level == None:
    #     level = random.random() * 4  # [0,4]
    level = 2
    angle_clip = math.pi
    angle_clip = angle_clip / 5 * (level + 1)
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
    return np.dot(pointcloud, R)

def corrupt_rotate_360_level3(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # if level == None:
    #     level = random.random() * 4  # [0,4]
    level = 3
    angle_clip = math.pi
    angle_clip = angle_clip / 5 * (level + 1)
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
    return np.dot(pointcloud, R)

def corrupt_rotate_360_level4(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # if level == None:
    #     level = random.random() * 4  # [0,4]
    level = 4
    angle_clip = math.pi
    angle_clip = angle_clip / 5 * (level + 1)
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
    return np.dot(pointcloud, R)

def corrupt_reflection(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    choice = np.array([1, -1])
    reflection = np.random.choice(choice, size=(3))
    Rx = np.array([[reflection[0], 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]])
    Ry = np.array([[1, 0, 0],
                   [0, reflection[1], 0],
                   [0, 0, 1]])
    Rz = np.array([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, reflection[2]]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return np.dot(pointcloud, R)

####### default p5
def corrupt_shear_p5(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    # shear_clip = (level + 1) * 0.1  # [0.1, 0.5]
    shear_clip = 0.5
    shear = np.random.uniform(-shear_clip, shear_clip, size=(6))

    Rz = np.array([[1, shear[0], shear[1]],
                   [shear[2], 1, shear[3]],
                   [shear[4], shear[5], 1]])
    return np.dot(pointcloud, Rz)

def corrupt_shear_p1(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    # shear_clip = (level + 1) * 0.1  # [0.1, 0.5]
    shear_clip = 0.1
    shear = np.random.uniform(-shear_clip, shear_clip, size=(6))

    Rz = np.array([[1, shear[0], shear[1]],
                   [shear[2], 1, shear[3]],
                   [shear[4], shear[5], 1]])
    return np.dot(pointcloud, Rz)

def corrupt_shear_p3(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    # shear_clip = (level + 1) * 0.1  # [0.1, 0.5]
    shear_clip = 0.3
    shear = np.random.uniform(-shear_clip, shear_clip, size=(6))

    Rz = np.array([[1, shear[0], shear[1]],
                   [shear[2], 1, shear[3]],
                   [shear[4], shear[5], 1]])
    return np.dot(pointcloud, Rz)

def corrupt_shear_p8(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    # shear_clip = (level + 1) * 0.1  # [0.1, 0.5]
    shear_clip = 0.8
    shear = np.random.uniform(-shear_clip, shear_clip, size=(6))

    Rz = np.array([[1, shear[0], shear[1]],
                   [shear[2], 1, shear[3]],
                   [shear[4], shear[5], 1]])
    return np.dot(pointcloud, Rz)

def corrupt_shear_1p(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    # shear_clip = (level + 1) * 0.1  # [0.1, 0.5]
    shear_clip = 1.0
    shear = np.random.uniform(-shear_clip, shear_clip, size=(6))

    Rz = np.array([[1, shear[0], shear[1]],
                   [shear[2], 1, shear[3]],
                   [shear[4], shear[5], 1]])
    return np.dot(pointcloud, Rz)

def corrupt_shear_2p(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    # shear_clip = (level + 1) * 0.1  # [0.1, 0.5]
    shear_clip = 2.0
    shear = np.random.uniform(-shear_clip, shear_clip, size=(6))

    Rz = np.array([[1, shear[0], shear[1]],
                   [shear[2], 1, shear[3]],
                   [shear[4], shear[5], 1]])
    return np.dot(pointcloud, Rz)

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

def corrupt_rotate_z_360(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    if level == None:
        level = random.random() * 4  # [0,4]
    angle_clip = math.pi  # 180 degree
    angle_clip = angle_clip / 5 * (level + 1)
    angles = np.random.uniform(-angle_clip, angle_clip, size=(1))
    Rz = np.array([[np.cos(angles[0]), -np.sin(angles[0]), 0],
                   [np.sin(angles[0]), np.cos(angles[0]), 0],
                   [0, 0, 1]])
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
def dropout_global_random(pointcloud, drop_rate=0.5):
    """
    Drop random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    # if level == None:
    #     levels = [0, 1, 2, 3, 4]
    #     level = random.choice(levels)
    # drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
    num_samples = pointcloud.size(0)
    num_points = pointcloud.size(1)
    inx = torch.rand(num_samples, num_points, 1).argsort(1).to(pointcloud.device)  # random index
    pointcloud = torch.take_along_dim(pointcloud, inx, dim=1)  # re-arange with random index
    pointcloud = pointcloud[:, :int(num_points * (1 - drop_rate)), :]
    return pointcloud.contiguous()

def corrupt_dropout_local(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = np.random.uniform(0.1, 0.5, size=(1))[0]
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
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

def corrupt_dropout_local_c5d1(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.1
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
    num_clusters = np.random.randint(1, 5)
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

def corrupt_dropout_local_c5d3(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.3
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
    num_clusters = np.random.randint(1, 5)
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

def corrupt_dropout_local_c1d3(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.3
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
    num_clusters = 1
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

def corrupt_dropout_local_c2d3(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.3
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
    num_clusters = np.random.randint(1, 2)
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

def corrupt_dropout_local_c3d3(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.3
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
    num_clusters = np.random.randint(1, 3)
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

def corrupt_dropout_local_c8d3(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.3
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
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

def corrupt_dropout_local_c5d5(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.5
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
    num_clusters = np.random.randint(1, 5)
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

def corrupt_dropout_local_c5d7(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.7
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
    num_clusters = np.random.randint(1, 5)
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

def corrupt_dropout_local_c5d9(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    # total_cluster_size = 100 * (level + 1)
    # total_cluster_size = int(num_points * (level + 1) * 0.1)  ## [0.1, 0.5]
    drop_ratio = 0.9
    total_cluster_size = int(num_points * drop_ratio)  ## [0.1, 0.5]
    num_clusters = np.random.randint(1, 5)
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


def corrupt_add_local(pointcloud, level):
    """
    Randomly add local clusters to a point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    total_cluster_size = int(num_points * (level + 1) * 0.1)
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    pointcloud = _shuffle_pointcloud(pointcloud)
    add_pcd = np.zeros_like(pointcloud)
    num_added = 0
    for i in range(num_clusters):
        K = cluster_size_list[i]
        sigma = np.random.uniform(0.075, 0.125)
        add_pcd[num_added:num_added + K, :] = np.copy(pointcloud[i:i + 1, :])
        add_pcd[num_added:num_added + K, :] = add_pcd[num_added:num_added + K, :] + sigma * np.random.randn(
            *add_pcd[num_added:num_added + K, :].shape)
        num_added += K
    assert num_added == total_cluster_size
    dist = np.sum(add_pcd ** 2, axis=1, keepdims=True).repeat(3, axis=1)
    add_pcd[dist > 1] = add_pcd[dist > 1] / dist[dist > 1]  # ensure the added points are inside a unit sphere
    pointcloud = np.concatenate([pointcloud, add_pcd], axis=0)
    pointcloud = pointcloud[:num_points + total_cluster_size]
    return pointcloud

# the Non-uniform density in MetaSet.
# When scanning an object with a LiDAR, the closer it is, the denser the point cloud will be.
# larger gate, more far point will be dropped.
def density(pc, level=None):
    if level == None:
        level = random.random() * 4  # [0,4]
    # gate = 1
    gate = level / 4.0 + 0.1  # [0.1, 1.1]
    # v_point = np.array([1, 0, 0])
    v_point = np.random.normal(0, 1, 3)
    norm = np.linalg.norm(v_point)
    v_point = v_point / norm  # one random point on the sphere surface

    dist = np.sqrt((v_point ** 2).sum())
    max_dist = dist + 1
    min_dist = dist - 1
    dist = np.linalg.norm(pc - v_point.reshape(1,3), axis=1)
    dist = (dist - min_dist) / (max_dist - min_dist)  ## [0,1]
    r_list = np.random.uniform(0, 1, pc.shape[0])
    # print(pc.shape)
    tmp_pc = pc[dist * gate < (r_list)]
    # print(tmp_pc.shape)
    return tmp_pc

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
    'scale': corrupt_scale,  # different scale across xyz + pc_norm; bad results
    'translate': corrupt_tranlate,  # different scale across xyz + pc_norm
    'translate_tiny': corrupt_tranlate_tiny,  # different scale across xyz + pc_norm
    'translate_middle': corrupt_tranlate_middle,  # different scale across xyz + pc_norm
    'translate_too_large': corrupt_tranlate_too_large,  # different scale across xyz + pc_norm
    ######################
    'scale_nonorm': corrupt_scale_nonorm_2p,  # different scale across xyz
    'scale_nonorm_1p5': corrupt_scale_nonorm_1p5,  # different scale across xyz
    'scale_nonorm_4': corrupt_scale_nonorm_4,  # different scale across xyz
    'scale_nonorm_10': corrupt_scale_nonorm_10,  # different scale across xyz
    ######################
    'scale_single': corrupt_scale_single,  # same scale across xyz + pc_norm; bas results.
    ##########
    'jitter': corrupt_jitter,
    'jitter_p01': corrupt_jitter_p01,
    'jitter_p03': corrupt_jitter_p03,
    'jitter_p05': corrupt_jitter_p05,
    'jitter_p1': corrupt_jitter_p1,
    ##############
    'rotate': corrupt_rotate_360,  # prior: only for aug
    'reflection': corrupt_reflection,  # prior: only for aug
    ################################################################
    'shear': corrupt_shear_p5,  # prior: only for aug
    'shear_p1': corrupt_shear_p1,  # prior: only for aug
    'shear_p3': corrupt_shear_p3,  # prior: only for aug
    'shear_p8': corrupt_shear_p8,  # prior: only for aug
    'shear_1p': corrupt_shear_1p,  # prior: only for aug
    'shear_2p': corrupt_shear_2p,  # prior: only for aug
    #########################################
    'shear_small': corrupt_shear_small,  # prior: only for aug
    'rotate_z': corrupt_rotate_z_360,
    # 'dropout_global': corrupt_dropout_global,
    'dropout_local': corrupt_dropout_local,
    'dropout_local_c5d1': corrupt_dropout_local_c5d1,
    'dropout_local_c5d3': corrupt_dropout_local_c5d3, ##### using this.
    'dropout_local_c5d5': corrupt_dropout_local_c5d5,
    'dropout_local_c5d7': corrupt_dropout_local_c5d7,
    'dropout_local_c5d9': corrupt_dropout_local_c5d9,
    ######
    'dropout_local_c1d3': corrupt_dropout_local_c1d3,
    'dropout_local_c2d3': corrupt_dropout_local_c2d3,
    'dropout_local_c3d3': corrupt_dropout_local_c3d3,
    'dropout_local_c8d3': corrupt_dropout_local_c8d3,
    # 'dropout_patch_pointmae': dropout_patch_random,
    'add_global': corrupt_add_global,
    'add_local': corrupt_add_local,
    'nonuniform_density': density,
    # different degree of rotation.
    'rotate_level0': corrupt_rotate_360_level0,
    'rotate_level1': corrupt_rotate_360_level1,
    'rotate_level2': corrupt_rotate_360_level2,
    'rotate_level3': corrupt_rotate_360_level3,
    'rotate_level4': corrupt_rotate_360_level4
}


affine_corruptions = ['translate', 'scale_nonorm', 'rotate', 'reflection', 'shear']
add_corruptions = ['add_global', 'add_local', 'jitter']
dropout_corruptions = ['dropout_local', 'nonuniform_density']  # 还有两个dropout, 是在forward 中实现的。
affine_corruptions_v2 = ['translate', 'scale_nonorm', 'rotate_level1', 'reflection', 'shear_1p']

def corrupt_data(data_instance, type=['clean']):
    # batch_data: B * N * 3.
    for corruption_item in type:
        if corruption_item == 'clean' or corruption_item == 'dropout_patch_pointmae' or 'dropout_global' in corruption_item:
        # if type == 'clean':
            pass
        elif corruption_item == 'affine_r5':
            ##  5个里面随机不重复选1-5 个， 然后采用随机顺序。
            numbers = [1,2,3,4,5]
            number = random.choice(numbers)
            adopted_affine = random.sample(affine_corruptions, number)
            for affine_corruption_item in adopted_affine:
                levels = [0, 1, 2, 3, 4]
                level = random.choice(levels)
                data_instance = corruptions[affine_corruption_item](data_instance, level)
        elif corruption_item == 'affine_r3':
            ##  5个里面随机不重复选1-3 个， 然后采用随机顺序。
            numbers = [1,2,3]
            number = random.choice(numbers)
            adopted_affine = random.sample(affine_corruptions, number)
            for affine_corruption_item in adopted_affine:
                levels = [0, 1, 2, 3, 4]
                level = random.choice(levels)
                data_instance = corruptions[affine_corruption_item](data_instance, level)
        elif corruption_item == 'affine_r5_v2':
            ##  5个里面随机不重复选1-5 个， 然后采用随机顺序。
            numbers = [1,2,3,4,5]
            number = random.choice(numbers)
            adopted_affine = random.sample(affine_corruptions_v2, number)
            for affine_corruption_item in adopted_affine:
                levels = [0, 1, 2, 3, 4]
                level = random.choice(levels)
                data_instance = corruptions[affine_corruption_item](data_instance, level)
        elif corruption_item == 'affine_r3_v2':
            ##  5个里面随机不重复选1-3 个， 然后采用随机顺序。
            numbers = [1,2,3]
            number = random.choice(numbers)
            adopted_affine = random.sample(affine_corruptions_v2, number)
            for affine_corruption_item in adopted_affine:
                levels = [0, 1, 2, 3, 4]
                level = random.choice(levels)
                data_instance = corruptions[affine_corruption_item](data_instance, level)
        else:
            levels = [0,1,2,3,4]
            level = random.choice(levels)
            data_instance = corruptions[corruption_item](data_instance, level)
            # return corruptions[type](data_instance, level)
    return data_instance

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

### For SO(3) rotation experiments only
def aug_rotate_z(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    angle_clip = math.pi
    angles = np.random.uniform(-angle_clip, angle_clip, size=(1))
    Rz = np.array([[np.cos(angles[0]), -np.sin(angles[0]), 0],
                   [np.sin(angles[0]), np.cos(angles[0]), 0],
                   [0, 0, 1]])
    return np.dot(pointcloud, Rz)

### For SO(3) rotation experiments only
def aug_rotate_360(pointcloud, level=None):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    angle_clip = math.pi
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
    return np.dot(pointcloud, R)


# for pre-training: apply the pc_norm(center and normalization), scale_translate, rotate, point_resampling, here.
# ? is there any difference between pre-training and fine-tuning?
# for fine-tuning apply the pc_norm(center and normalization), scale_translate, rotate, point_resampling, here.
def augment_data(data_instance, type=['clean']):
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
        # elif corruption_item == 'jitter':
        #     data_instance = corrupt_jitter(data_instance)
        elif corruption_item == 'rotate_z':
            data_instance = aug_rotate_z(data_instance)
        elif corruption_item == 'rotate':
            data_instance = aug_rotate_360(data_instance)
        else:
            raise NotImplementedError

    return data_instance
