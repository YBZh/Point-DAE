import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *
from torchvision import transforms
from datasets import data_transforms
from .corrupt_util import corrupt_data, augment_data
import ipdb

# train_transforms = transforms.Compose(
#     [
#         # data_transforms.PointcloudScale(),
#         # data_transforms.PointcloudRotate(),
#         # data_transforms.PointcloudRotatePerturbation(),
#         # data_transforms.PointcloudTranslate(),
#         # data_transforms.PointcloudJitter(),
#         # data_transforms.PointcloudRandomInputDropout(),
#         data_transforms.PointcloudScaleAndTranslate(), # the code here is for batch data.
#     ]
# )


@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.npoints = config.N_POINTS
        self.subset = config.subset
        self.aug_type = config.aug_type
        self.corrupt_type = config.corrupt_type
        
        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')

        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        print_log(f'[DATASET] aug_type {self.aug_type}', logger = 'ShapeNet-55')
        print_log(f'[DATASET] corrupt_type {self.corrupt_type}', logger = 'ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')

        # self.permutation = np.arange(self.npoints)

    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        if pc.shape[0] >= num:
            self.permutation = np.arange(pc.shape[0])
            np.random.shuffle(self.permutation)
            pc = pc[self.permutation[:num]]
        else:
            gap = num - pc.shape[0]
            indices = np.random.choice(pc.shape[0], gap, replace=True)
            pc = np.vstack((pc, pc[indices]))
            self.permutation = np.arange(pc.shape[0])
            np.random.shuffle(self.permutation)
            pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]
        # print(sample['file_path'])
        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)

        # data[:, :3] = self.pc_norm(data[:, :3])
        # data = self.pc_norm(data[:, :3]) # only utilize the point input.
        # data = data[:, :3]
        data[:, :3] = augment_data(data[:, :3], self.aug_type) # one PC norm is always applied at the beginning of aug, unless the scale aug is applied.
        clean_data = self.random_sample(data, self.sample_points_num)
        # print(clean_data.shape) # 1024
        # print(data.shape) # 8192

        # #### for MaskSurf,  (1) non-transformer backbone, drop local.
        # #                   (2) transformer backbone, non, since masking is conducted in the forward.
        # #### for PointMA2E: (1) non-transformer backbone, affine + drop_local
        # ####                (2) transformer backbone, only affine, since patch masking is conducted in the forward.
        ################ version 1, although masking is adopted, the resulting point cloud still has 1K points; Clean & corrupted are resampled data.
        # corrupted_data = data
        # corrupted_data_output = corrupt_data(corrupted_data[:, :3], self.corrupt_type)  # reconstruct the clean data from the corrupted_data.
        # # corrupted_data[:, :3] = corrupt_data(corrupted_data[:, :3], self.corrupt_type) # reconstruct the clean data from the corrupted_data.
        # # print(corrupted_data.shape)  # M
        # corrupted_data = self.random_sample(corrupted_data_output, self.sample_points_num)
        ################ version 2, after masking, the resulting point cloud contains less than 1K points; corrupted data are sampled from clean data.
        corrupted_data = corrupt_data(clean_data[:, :3], self.corrupt_type)  # reconstruct the clean data from the corrupted_data.

        clean_data = torch.from_numpy(clean_data).float()
        corrupted_data = torch.from_numpy(corrupted_data).float()
        return sample['taxonomy_id'], sample['model_id'], corrupted_data, clean_data

    def __len__(self):
        return len(self.file_list)