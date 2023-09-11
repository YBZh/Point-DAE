#  Copyright (c) 2022. Yabin Zhang
#  Ref: https://github.com/krrish94/chamferdist
#  Ref: https://github.com/chrdiller/pyTorchChamferDistance
#  Ref: https://github.com/wentaoyuan/pcn/blob/master/models/pcn_cd.py
#  Ref: https://github.com/AnTao97/UnsupervisedPointCloudReconstruction/blob/master/model.py
#  Ref: OcCo
#  Ref: PointMAE
import torch
import torch.nn as nn
import itertools
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
import torch, torch.nn as nn, torch.nn.functional as F
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
from .pointnetv2_util import PointNetv2_encoder
# from .corrupt_util import corrupt_data
import ipdb
import time
from datasets.corrupt_util import dropout_patch_random, dropout_global_random

class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        # for complexity calculation.
        # print(xyz.shape)
        # if xyz.shape[0] == 1:
        #     xyz = xyz[0]
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        _, center = misc.fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

@MODELS.register_module()
class Point_CAE_PointNetv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_PointNet] ', logger ='Point_CAE_PointNet')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.pointnetv2_encoder = PointNetv2_encoder()
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]
        self.folding1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(1024 + 2 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_grid(self, batch_size):
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        # substitute for tf.tile:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/tile
        # Ref: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
        return tensor

    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        # pts: B*N*D (D=3 for pure point cloud)
        corrupted_pts = corrupted_pts[:, :, :3].contiguous()
        pts = pts[:, :, :3].contiguous()
        # corrupted_pts = corrupt_data(pts, type=self.corruption_type)  ## We conduct the point cloud corruptions here.
        # corrupted_pts = corrupted_pts.to(self.device)
        # start = time.time()
        # corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass
        feature = self.pointnetv2_encoder(corrupted_pts)
        # print('encoder time', time.time() - start) # 3.5, this occupies most of the training time.

        coarse = self.folding1(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)

        grid = self.build_grid(pts.shape[0])
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = point_feat.view([-1, self.num_fine, 3])

        global_feat = self.tile(self.expand_dims(feature, 1), [1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)

        center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.view([-1, self.num_fine, 3])
        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center
        # print('decoder time', time.time() - start)

        loss_coarse = self.loss_func(coarse, pts)
        loss_fine = self.loss_func(fine, pts)
        # print('loss time', time.time() - start)
        return loss_coarse, loss_fine

@MODELS.register_module()
class Point_MA2E_PointNetv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_PointNet] ', logger ='Point_MA2E_PointNetv2')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 6
        self.grid_scale = 0.05
        self.num_coarse = 64
        self.pointnetv2_encoder = PointNetv2_encoder()
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 1024)
        )
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        # self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
        #                  [-self.grid_scale, self.grid_scale, self.grid_size]]
        self.meshgrid = [[-0.3, 0.3, 6], [-0.3, 0.3, 6]]
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        # self.folding2 = nn.Sequential(
        #     nn.Conv1d(1024 + 3, 512, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 512, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 3, 1))
        self.folding1 = nn.Sequential(
            nn.Conv1d(1024 + 2, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(1024 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = config.loss
        self.group_divider = Group(num_group = self.num_coarse, group_size = 32)
        # loss
        self.build_loss_func(self.loss)

    # def build_grid(self, batch_size):
    #     x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
    #     points = np.array(list(itertools.product(x, y)))
    #     points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
    #
    #     return torch.tensor(points).float().to(self.device)


    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        # pts: B*N*D (D=3 for pure point cloud)
        corrupted_pts = corrupted_pts[:, :, :3].contiguous()
        pts = pts[:, :, :3].contiguous()
        # corrupted_pts = corrupt_data(pts, type=self.corruption_type)  ## We conduct the point cloud corruptions here.
        # corrupted_pts = corrupted_pts.to(self.device)
        # start = time.time()
        # corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass

        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        # print(center.size())  ## 32*64*3
        pos_emd = self.pos_embed(center)  ### (batch_size, group number, feat_dims)
        # print(pos_emd.size())
        feature = self.pointnetv2_encoder(corrupted_pts)
        ### 128 * 1024
        # print('encoder time', time.time() - start) # 3.5, this occupies most of the training time.

        coarse = self.coarse_pred(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)

        feature_expand = feature.unsqueeze(1)  ## 128 * 1 * 1024
        feature_expand = feature_expand.expand(-1, self.num_coarse, -1) ## 128 * 64 * 1024
        B, M, C = feature_expand.shape
        feature_expand = feature_expand.reshape(B * M, C)
        feature_expand = feature_expand.unsqueeze(-1).expand(-1,-1, 36)  # (batch_size*group number, feat_dims, num_points)

        pos_emd = pos_emd.reshape(B * M, C)
        pos_emd = pos_emd.unsqueeze(-1).expand(-1,-1, 36)  # (batch_size*group number, feat_dims, num_points)

        points = self.build_grid(B * M).transpose(1,2)
        # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)

        if feature.get_device() != -1:
            points = points.cuda(feature.get_device())

        cat1 = torch.cat((feature_expand + pos_emd, points),
                         dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
        cat2 = torch.cat((feature_expand + pos_emd, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
        folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)

        # print(folding_result2.size())
        # grid = self.build_grid(pts.shape[0])
        # grid_feat = grid.repeat(1, self.num_coarse, 1)
        #
        # point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        # point_feat = point_feat.view([-1, self.num_fine, 3])
        #
        # global_feat = self.tile(self.expand_dims(feature, 1), [1, self.num_fine, 1])
        # feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)
        #
        # center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        # center = center.view([-1, self.num_fine, 3])
        # fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center
        # # print('decoder time', time.time() - start)
        loss_coarse = self.loss_func(coarse, center)
        B, M, G_size, num_point = neighborhood_normalized.shape
        loss_fine = self.loss_func(folding_result2, neighborhood_normalized.view(B*M, G_size, num_point))
        # print('loss time', time.time() - start)
        return loss_fine, loss_coarse

@MODELS.register_module()
class Point_MA2E_PointNetv2_local_only(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_PointNet] ', logger ='Point_MA2E_PointNetv2')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 6
        self.grid_scale = 0.05
        self.num_coarse = 64
        self.pointnetv2_encoder = PointNetv2_encoder()
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 1024)
        )
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        # self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
        #                  [-self.grid_scale, self.grid_scale, self.grid_size]]
        self.meshgrid = [[-0.3, 0.3, 6], [-0.3, 0.3, 6]]
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        # self.folding2 = nn.Sequential(
        #     nn.Conv1d(1024 + 3, 512, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 512, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 3, 1))
        self.folding1 = nn.Sequential(
            nn.Conv1d(1024 + 2, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(1024 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = config.loss
        self.group_divider = Group(num_group = self.num_coarse, group_size = 32)
        # loss
        self.build_loss_func(self.loss)

    # def build_grid(self, batch_size):
    #     x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
    #     points = np.array(list(itertools.product(x, y)))
    #     points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
    #
    #     return torch.tensor(points).float().to(self.device)


    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        # pts: B*N*D (D=3 for pure point cloud)
        corrupted_pts = corrupted_pts[:, :, :3].contiguous()
        pts = pts[:, :, :3].contiguous()
        # corrupted_pts = corrupt_data(pts, type=self.corruption_type)  ## We conduct the point cloud corruptions here.
        # corrupted_pts = corrupted_pts.to(self.device)
        # start = time.time()
        # corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass

        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        # print(center.size())  ## 32*64*3
        pos_emd = self.pos_embed(center)  ### (batch_size, group number, feat_dims)
        # print(pos_emd.size())
        feature = self.pointnetv2_encoder(corrupted_pts)
        ### 128 * 1024
        # print('encoder time', time.time() - start) # 3.5, this occupies most of the training time.

        coarse = self.coarse_pred(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)

        feature_expand = feature.unsqueeze(1)  ## 128 * 1 * 1024
        feature_expand = feature_expand.expand(-1, self.num_coarse, -1) ## 128 * 64 * 1024
        B, M, C = feature_expand.shape
        feature_expand = feature_expand.reshape(B * M, C)
        feature_expand = feature_expand.unsqueeze(-1).expand(-1,-1, 36)  # (batch_size*group number, feat_dims, num_points)

        pos_emd = pos_emd.reshape(B * M, C)
        pos_emd = pos_emd.unsqueeze(-1).expand(-1,-1, 36)  # (batch_size*group number, feat_dims, num_points)

        points = self.build_grid(B * M).transpose(1,2)
        # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)

        if feature.get_device() != -1:
            points = points.cuda(feature.get_device())

        cat1 = torch.cat((feature_expand + pos_emd, points),
                         dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
        folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
        cat2 = torch.cat((feature_expand + pos_emd, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
        folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)

        # print(folding_result2.size())
        # grid = self.build_grid(pts.shape[0])
        # grid_feat = grid.repeat(1, self.num_coarse, 1)
        #
        # point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        # point_feat = point_feat.view([-1, self.num_fine, 3])
        #
        # global_feat = self.tile(self.expand_dims(feature, 1), [1, self.num_fine, 1])
        # feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)
        #
        # center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        # center = center.view([-1, self.num_fine, 3])
        # fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center
        # # print('decoder time', time.time() - start)
        # loss_coarse = self.loss_func(coarse, center)
        B, M, G_size, num_point = neighborhood_normalized.shape
        loss_fine = self.loss_func(folding_result2, neighborhood_normalized.view(B*M, G_size, num_point))
        # print('loss time', time.time() - start)
        return loss_fine, torch.zeros(1).to(loss_fine.device)

@MODELS.register_module()
class Point_MA2E_PointNetv2_global_only(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_PointNet] ', logger ='Point_MA2E_PointNetv2')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 6
        self.grid_scale = 0.05
        self.num_coarse = 64
        self.pointnetv2_encoder = PointNetv2_encoder()
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, 1024)
        )
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        # self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
        #                  [-self.grid_scale, self.grid_scale, self.grid_size]]
        self.meshgrid = [[-0.3, 0.3, 6], [-0.3, 0.3, 6]]
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        # self.folding2 = nn.Sequential(
        #     nn.Conv1d(1024 + 3, 512, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 512, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(512, 3, 1))
        self.folding1 = nn.Sequential(
            nn.Conv1d(1024 + 2, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(1024 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1),
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = config.loss
        self.group_divider = Group(num_group = self.num_coarse, group_size = 32)
        # loss
        self.build_loss_func(self.loss)

    # def build_grid(self, batch_size):
    #     x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
    #     points = np.array(list(itertools.product(x, y)))
    #     points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
    #
    #     return torch.tensor(points).float().to(self.device)


    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        # pts: B*N*D (D=3 for pure point cloud)
        corrupted_pts = corrupted_pts[:, :, :3].contiguous()
        pts = pts[:, :, :3].contiguous()
        # corrupted_pts = corrupt_data(pts, type=self.corruption_type)  ## We conduct the point cloud corruptions here.
        # corrupted_pts = corrupted_pts.to(self.device)
        # start = time.time()
        # corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass

        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        # print(center.size())  ## 32*64*3
        pos_emd = self.pos_embed(center)  ### (batch_size, group number, feat_dims)
        # print(pos_emd.size())
        feature = self.pointnetv2_encoder(corrupted_pts)
        ### 128 * 1024
        # print('encoder time', time.time() - start) # 3.5, this occupies most of the training time.

        coarse = self.coarse_pred(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)

        # feature_expand = feature.unsqueeze(1)  ## 128 * 1 * 1024
        # feature_expand = feature_expand.expand(-1, self.num_coarse, -1) ## 128 * 64 * 1024
        # B, M, C = feature_expand.shape
        # feature_expand = feature_expand.reshape(B * M, C)
        # feature_expand = feature_expand.unsqueeze(-1).expand(-1,-1, 36)  # (batch_size*group number, feat_dims, num_points)
        #
        # pos_emd = pos_emd.reshape(B * M, C)
        # pos_emd = pos_emd.unsqueeze(-1).expand(-1,-1, 36)  # (batch_size*group number, feat_dims, num_points)
        #
        # points = self.build_grid(B * M).transpose(1,2)
        # # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
        #
        # if feature.get_device() != -1:
        #     points = points.cuda(feature.get_device())
        #
        # cat1 = torch.cat((feature_expand + pos_emd, points),
        #                  dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
        # folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
        # cat2 = torch.cat((feature_expand + pos_emd, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
        # folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
        # folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)

        loss_coarse = self.loss_func(coarse, center)
        # B, M, G_size, num_point = neighborhood_normalized.shape
        # loss_fine = self.loss_func(folding_result2, neighborhood_normalized.view(B*M, G_size, num_point))
        # print('loss time', time.time() - start)
        return loss_coarse, torch.zeros(1).to(loss_coarse.device)



@MODELS.register_module()
class Point_CAE_PointNetv2_Proj(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_PointNetv2_Proj] ', logger ='Point_CAE_PointNetv2_Proj')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.pointnetv2_encoder = PointNetv2_encoder()
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]
        self.proj = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1024))

        self.folding1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.folding2 = nn.Sequential(
            nn.Conv1d(1024 + 2 + 3, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_grid(self, batch_size):
        x, y = np.linspace(*self.meshgrid[0]), np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)

        return torch.tensor(points).float().to(self.device)

    def tile(self, tensor, multiples):
        # substitute for tf.tile:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/tile
        # Ref: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        def tile_single_axis(a, dim, n_tile):
            init_dim = a.size()[dim]
            repeat_idx = [1] * a.dim()
            repeat_idx[dim] = n_tile
            a = a.repeat(*repeat_idx)
            order_index = torch.Tensor(
                np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).long()
            return torch.index_select(a, dim, order_index.to(self.device))

        for dim, n_tile in enumerate(multiples):
            if n_tile == 1:
                continue
            tensor = tile_single_axis(tensor, dim, n_tile)
        return tensor

    @staticmethod
    def expand_dims(tensor, dim):
        # substitute for tf.expand_dims:
        # https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/expand_dims
        return tensor.unsqueeze(-1).transpose(-1, dim)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        # pts: B*N*D (D=3 for pure point cloud)
        corrupted_pts = corrupted_pts[:, :, :3].contiguous()
        pts = pts[:, :, :3].contiguous()
        # corrupted_pts = corrupt_data(pts, type=self.corruption_type)  ## We conduct the point cloud corruptions here.
        # corrupted_pts = corrupted_pts.to(self.device)
        # start = time.time()
        # corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass
        feature = self.pointnetv2_encoder(corrupted_pts)
        feature = self.proj(feature)
        coarse = self.folding1(feature)
        coarse = coarse.view(-1, self.num_coarse, 3)

        grid = self.build_grid(pts.shape[0])
        grid_feat = grid.repeat(1, self.num_coarse, 1)

        point_feat = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        point_feat = point_feat.view([-1, self.num_fine, 3])

        global_feat = self.tile(self.expand_dims(feature, 1), [1, self.num_fine, 1])
        feat = torch.cat([grid_feat, point_feat, global_feat], dim=2)

        center = self.tile(self.expand_dims(coarse, 2), [1, 1, self.grid_size ** 2, 1])
        center = center.view([-1, self.num_fine, 3])
        fine = self.folding2(feat.transpose(2, 1)).transpose(2, 1) + center

        loss_coarse = self.loss_func(coarse, pts)
        loss_fine = self.loss_func(fine, pts)

        return loss_coarse, loss_fine

# finetune model
@MODELS.register_module()
class PointNetv2(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.smoothing = config.smoothloss
        self.pointnetv2_encoder = PointNetv2_encoder()
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        if self.smoothing:
            eps = 0.3
            n_class = ret.size()[1]
            one_hot = torch.zeros_like(ret).scatter(1, gt.long().view(-1, 1), 1)  # (num_points, num_class)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(ret, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()  # ~ F.nll_loss(log_prb, gold)
        else:
            loss = F.cross_entropy(ret, gt.long(), reduction='mean')
        # loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        pts = pts[:, :, :3].contiguous()
        # pts = pts.transpose(1,2)
        feature = self.pointnetv2_encoder(pts)
        ret = self.cls_head_finetune(feature)

        return ret


# finetune model
@MODELS.register_module()
class PointNetv2_Linear(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.smoothing = config.smoothloss
        self.pointnetv2_encoder = PointNetv2_encoder()
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(1024, self.cls_dim)
            )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        if self.smoothing:
            eps = 0.3
            n_class = ret.size()[1]
            one_hot = torch.zeros_like(ret).scatter(1, gt.long().view(-1, 1), 1)  # (num_points, num_class)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(ret, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()  # ~ F.nll_loss(log_prb, gold)
        else:
            loss = F.cross_entropy(ret, gt.long(), reduction='mean')
        # loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        pts = pts[:, :, :3].contiguous()
        # pts = pts.transpose(1,2)
        feature = self.pointnetv2_encoder(pts)
        ret = self.cls_head_finetune(feature)

        return ret





# finetune model
@MODELS.register_module()
class PointNetv2_feat(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.smoothing = config.smoothloss
        self.pointnetv2_encoder = PointNetv2_encoder()

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        if self.smoothing:
            eps = 0.3
            n_class = ret.size()[1]
            one_hot = torch.zeros_like(ret).scatter(1, gt.long().view(-1, 1), 1)  # (num_points, num_class)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(ret, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()  # ~ F.nll_loss(log_prb, gold)
        else:
            loss = F.cross_entropy(ret, gt.long(), reduction='mean')
        # loss = self.loss_ce(ret, gt.long())
        pred = ret.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))
        return loss, acc * 100

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder') :
                    base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
                    del base_ckpt[k]
                elif k.startswith('base_model'):
                    base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
                    del base_ckpt[k]
            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print_log('missing_keys', logger='Transformer')
                print_log(
                    get_missing_parameters_message(incompatible.missing_keys),
                    logger='Transformer'
                )
            if incompatible.unexpected_keys:
                print_log('unexpected_keys', logger='Transformer')
                print_log(
                    get_unexpected_parameters_message(incompatible.unexpected_keys),
                    logger='Transformer'
                )

            print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
        else:
            print_log('Training from scratch!!!', logger='Transformer')
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, pts):

        pts = pts[:, :, :3].contiguous()
        # pts = pts.transpose(1,2)
        feature = self.pointnetv2_encoder(pts)
        # ret = self.cls_head_finetune(feature)

        return feature




