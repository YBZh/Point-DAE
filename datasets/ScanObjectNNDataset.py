import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *
from .corrupt_util import corrupt_data, augment_data
from pointnet2_ops import pointnet2_utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


@DATASETS.register_module()
class ScanObjectNN(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        self.aug_type = config.aug_type
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        current_points = augment_data(current_points, self.aug_type)  # one PC norm is always applied at the beginning of aug.

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]



@DATASETS.register_module()
class ScanObjectNN_hardest(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        self.npoints = config.npoints
        self.aug_type = config.aug_type

        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy() #  2048 * 3
        # data processing/augmentation
        data = augment_data(current_points, self.aug_type)  # No PC norm applied.
        # data resampling; this is very slow here.
        # if self.subset == 'train':
        #     # conduct resample here.
        #     if self.npoints == 1024:
        #         point_all = 1200
        #     elif self.npoints == 2048:
        #         point_all = 2400
        #     elif self.npoints == 4096:
        #         point_all = 4800
        #     elif self.npoints == 8192:
        #         point_all = 8192
        #     else:
        #         raise NotImplementedError()
        #     if data.shape[0] < point_all:
        #         point_all = data.shape[0]
        #     data = torch.from_numpy(data).float().unsqueeze(0)
        #     fps_idx = farthest_point_sample(data, point_all)  # (1, npoint)
        #     fps_idx = fps_idx[:, np.random.choice(point_all, self.npoints, False)]
        #     points = index_points(data, fps_idx)
        #     # points = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (1, N, 3)
        #     current_points = points.squeeze(0)
        # else:
        #     # sample the required number with FPS.
        #     if data.shape[0] < self.npoints:
        #         self.npoints = data.shape[0]
        #     data = torch.from_numpy(data).float().unsqueeze(0)
        #     fps_idx = farthest_point_sample(data, self.npoints)  # (1, npoint)
        #     points = index_points(data, fps_idx)
        #     # points = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()  # (1, N, 3)
        #     current_points = points.squeeze(0)

        current_points = torch.from_numpy(data).float()
        label = self.labels[idx]
        
        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]