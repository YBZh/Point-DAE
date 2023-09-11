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
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2
import torch, torch.nn as nn, torch.nn.functional as F
from .dgcnn_util import dgcnn_encoder, dgcnn_encoder_nopooling
# from .corrupt_util import corrupt_data
import ipdb
from datasets.corrupt_util import dropout_patch_random, dropout_global_random

@MODELS.register_module()
class Point_CAE_DGCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_DGCNN] ', logger ='Point_CAE_DGCNN')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.dgcnn_encoder = dgcnn_encoder(channel=3)
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
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            elif corruption_item == 'random_dropout':
                if random.random() > 0.5:
                    corrupted_pts = dropout_patch_random(corrupted_pts)
                else:
                    corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass
        corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N

        feature = self.dgcnn_encoder(corrupted_pts)
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
        if vis:
            return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
        else:
            return loss_coarse, loss_fine

@MODELS.register_module()
class Point_CAE_DGCNN_FCOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_DGCNN] ', logger ='Point_CAE_DGCNN')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.dgcnn_encoder = dgcnn_encoder(channel=3)
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]
        self.folding1 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.num_coarse * 3))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

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

    def forward(self, corrupted_pts, pts, vis=False, return_feat=False, **kwargs):
        if return_feat:
            pts = pts[:, :, :3]
            pts = pts.transpose(1, 2).contiguous()
            feature = self.dgcnn_encoder(pts)
            return feature
        else:
            # pts: B*N*D (D=3 for pure point cloud)
            corrupted_pts = corrupted_pts[:, :, :3].contiguous()
            pts = pts[:, :, :3].contiguous()
            # print(self.loss_func(corrupted_pts, pts))
            # corrupted_pts = corrupt_data(pts, type=self.corruption_type)  ## We conduct the point cloud corruptions here.
            # corrupted_pts = corrupted_pts.to(self.device)
            for corruption_item in self.corrupt_type:
                if corruption_item == 'dropout_patch_pointmae':
                    corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
                elif corruption_item == 'dropout_global':
                    corrupted_pts = dropout_global_random(corrupted_pts)  ##
                elif corruption_item == 'dropout_global_p1':
                    corrupted_pts = dropout_global_random(corrupted_pts, drop_rate=0.1)  ##
                elif corruption_item == 'dropout_global_p3':
                    corrupted_pts = dropout_global_random(corrupted_pts, drop_rate=0.3)  ##
                elif corruption_item == 'dropout_global_p5':
                    corrupted_pts = dropout_global_random(corrupted_pts, drop_rate=0.5)  ##
                elif corruption_item == 'dropout_global_p7':
                    corrupted_pts = dropout_global_random(corrupted_pts, drop_rate=0.7)  ##
                elif corruption_item == 'dropout_global_p9':
                    corrupted_pts = dropout_global_random(corrupted_pts, drop_rate=0.9)  ##
                elif corruption_item == 'random_dropout':
                    if random.random() > 0.5:
                        corrupted_pts = dropout_patch_random(corrupted_pts)
                    else:
                        corrupted_pts = dropout_global_random(corrupted_pts)  ##
                else:
                    pass
            corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N

            feature = self.dgcnn_encoder(corrupted_pts)
            coarse = self.folding1(feature)
            coarse = coarse.view(-1, self.num_coarse, 3)
            loss_coarse = self.loss_func(coarse, pts)
            if vis:
                return corrupted_pts.transpose(1, 2).contiguous(), coarse, coarse, pts
            else:
                return loss_coarse, torch.zeros(1).to(loss_coarse.device)


@MODELS.register_module()
class Point_CAE_DGCNN_FoldOnly(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_DGCNN] ', logger ='Point_CAE_DGCNN')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.dgcnn_encoder = dgcnn_encoder(channel=3)
        self.num_fine = self.grid_size ** 2 * self.num_coarse  # 16384
        self.meshgrid = [[-self.grid_scale, self.grid_scale, self.grid_size],
                         [-self.grid_scale, self.grid_scale, self.grid_size]]
        # self.folding1 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, self.num_coarse * 3))
        self.folding1 = nn.Sequential(
            nn.Conv1d(1024 + 2, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(1024 + 3, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, 1),
            nn.ReLU(),
            nn.Conv1d(1024, 3, 1),
        )
        self.meshgrid = [[-0.3, 0.3, 32], [-0.3, 0.3, 32]]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

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
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            elif corruption_item == 'random_dropout':
                if random.random() > 0.5:
                    corrupted_pts = dropout_patch_random(corrupted_pts)
                else:
                    corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass
        corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N

        feature = self.dgcnn_encoder(corrupted_pts)  ## 16*1024
        feature = feature.unsqueeze(-1).repeat(1, 1, 1024)  # ## 16*1024 * 1024
        # print(feature.size())
        points = self.build_grid(feature.shape[0]).transpose(1,2)  ## 16*2*1024
        # print(points.size())
        if feature.get_device() != -1:
            points = points.cuda(feature.get_device())
        cat1 = torch.cat((feature, points),dim=1)
        folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
        cat2 = torch.cat((feature, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
        folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
        folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)
        # coarse = self.folding1(feature)
        # coarse = coarse.view(-1, self.num_coarse, 3)
        loss_coarse = self.loss_func(folding_result2, pts)

        return loss_coarse, torch.zeros(1).to(loss_coarse.device)


@MODELS.register_module()
class Point_AE_Corruption_DGCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_DGCNN] ', logger ='Point_AE_Corruption_DGCNN')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.dgcnn_encoder = dgcnn_encoder(channel=3)
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
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            elif corruption_item == 'random_dropout':
                if random.random() > 0.5:
                    corrupted_pts = dropout_patch_random(corrupted_pts)
                else:
                    corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass
        pts = corrupted_pts  ## B * N * 3, target of reconstruction is the corrupted samples.
        corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N

        feature = self.dgcnn_encoder(corrupted_pts)
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


@MODELS.register_module()
class Point_CAE_DGCNN_proj(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_CAE_DGCNN_proj] ', logger ='Point_CAE_DGCNN_proj')
        self.config = config
        self.corrupt_type = config.corrupt_type
        self.grid_size = 4
        self.grid_scale = 0.05
        self.num_coarse = 1024
        self.dgcnn_encoder = dgcnn_encoder(channel=3)
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
        for corruption_item in self.corrupt_type:
            if corruption_item == 'dropout_patch_pointmae':
                corrupted_pts = dropout_patch_random(corrupted_pts)  ## only the patch drop is applied here, since it is finished with CUDA.
            elif corruption_item == 'dropout_global':
                corrupted_pts = dropout_global_random(corrupted_pts)  ##
            else:
                pass
        corrupted_pts = corrupted_pts.transpose(1, 2).contiguous()  ## the input to PointNet shoube be B*3*N

        feature = self.dgcnn_encoder(corrupted_pts)
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
class DGCNN(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.smoothing = config.smoothloss
        self.dgcnn_encoder = dgcnn_encoder(channel=3)
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
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

        pts = pts[:, :, :3]
        pts = pts.transpose(1,2).contiguous()
        feature = self.dgcnn_encoder(pts)
        ret = self.cls_head_finetune(feature)

        return ret


@MODELS.register_module()
class DGCNN_Linear(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.smoothing = config.smoothloss
        self.dgcnn_encoder = dgcnn_encoder(channel=3)
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

        pts = pts[:, :, :3]
        pts = pts.transpose(1,2).contiguous()
        feature = self.dgcnn_encoder(pts)
        ret = self.cls_head_finetune(feature)

        return ret


# finetune model
@MODELS.register_module()
class DGCNN_feat(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.smoothing = config.smoothloss
        self.dgcnn_encoder = dgcnn_encoder(channel=3)
        # self.cls_head_finetune = nn.Sequential(
        #         nn.Linear(1024, 512),
        #         nn.BatchNorm1d(512),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.Linear(512, 256),
        #         nn.BatchNorm1d(256),
        #         nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #         nn.Dropout(0.5),
        #         nn.Linear(256, self.cls_dim)
        #     )

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
        # ipdb.set_trace()
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

        pts = pts[:, :, :3]
        pts = pts.transpose(1,2).contiguous()
        feature = self.dgcnn_encoder(pts)
        # ret = self.cls_head_finetune(feature)

        return feature


# dgcnn for mask feature, return point wise feature representations.
@MODELS.register_module()
class DGCNN_MaskFeat(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.cls_dim = config.cls_dim
        self.dgcnn_encoder = dgcnn_encoder_nopooling(channel=3)
        self.cls_head_finetune = nn.Sequential(
                nn.Linear(1024, 512),
                nn.BatchNorm1d(512),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        self.build_loss_func()

    def build_loss_func(self):
        self.loss_ce = nn.CrossEntropyLoss()

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

        pts = pts[:, :, :3]
        pts = pts.transpose(1,2).contiguous()
        feature = self.dgcnn_encoder(pts)

        return feature

######################## copyed from crosspoint source code.
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = x.get_device()
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature

@MODELS.register_module()
class DGCNN_CrossPoint(nn.Module):
    def __init__(self, args):
        super(DGCNN_CrossPoint, self).__init__()
        # self.args = args
        self.k = 20 ### for fair comparison

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))


        self.inv_head = nn.Sequential(
            nn.Linear(1024 * 2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)
        #
        # feat = x
        # inv_feat = self.inv_head(feat)

        return x

    # encoder + one fc classifier


