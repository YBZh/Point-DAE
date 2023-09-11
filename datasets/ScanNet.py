import os
import json
import torch
import numpy as np
import torch.utils.data as data
from .corrupt_util import corrupt_data, augment_data

from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class ScanNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH ## data/ScanNet
        self.folder = config.FOLDER  ## scannet
        self.npoints = int(config.N_POINTS)
        self.aug_type = config.aug_type
        self.split = config.SPLIT  ### medium
        ## data/ScanNet/catelog_medium.json
        with open(os.path.join(self.data_root, f'catalog_{self.split}.json')) as fp:
            self.data_objs = json.load(fp)
        print_log(f'[DATASET] {len(self.data_objs)} instances loaded from {self.split} split.', logger = 'ScanNet')

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc

    def sample_pts(self, point_cloud, num):
        if len(point_cloud) >= num:
            idx = np.random.choice(len(point_cloud), num, replace=False)
        else:
            idx = np.random.choice(len(point_cloud), num, replace=True)
        return point_cloud[idx,:]

    def __getitem__(self, idx):
        ### data/ScanNet/scannet/
        data = np.load(os.path.join(self.data_root, self.folder, self.data_objs[idx]))
        data = data[:, 0:3]

        data = self.sample_pts(data, self.npoints)
        # data = self.pc_norm(data)
        data = augment_data(data, self.aug_type)
        data = torch.from_numpy(data).float()
        return 0, 0, data, data

    def __len__(self):
        return len(self.data_objs)
