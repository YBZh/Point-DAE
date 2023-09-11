import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from timm.models.layers import DropPath, trunc_normal_
import numpy as np
from .build import MODELS
from utils import misc
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from utils.logger import *
import random
from knn_cuda import KNN
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferDistanceL2_withnormal, \
    ChamferDistanceL2_withnormal_strict, ChamferDistanceL2_withnormal_strict_normalindex, ChamferDistanceL2_withnormal_normalindex, ChamferDistanceL2_withnormalL1
import ipdb
from pointnet2_ops import pointnet2_utils


class Decoder(nn.Module):
    def __init__(self, encoder_channel, num_fine):
        super().__init__()
        self.num_fine = num_fine
        self.grid_size = 2
        self.num_coarse = self.num_fine // 4
        assert num_fine % 4 == 0

        self.mlp = nn.Sequential(
            nn.Linear(encoder_channel, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_coarse)
        )
        self.final_conv = nn.Sequential(
            nn.Conv1d(encoder_channel + 3 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2)  # 1 2 S

    def forward(self, feature_global):
        '''
            feature_global : B G C
            -------
            coarse : B G M 3
            fine : B G N 3

        '''
        bs, g, c = feature_global.shape
        feature_global = feature_global.reshape(bs * g, c)

        coarse = self.mlp(feature_global).reshape(bs * g, self.num_coarse, 3)  # BG M 3

        point_feat = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # BG (M) S 3
        point_feat = point_feat.reshape(bs * g, self.num_fine, 3).transpose(2, 1)  # BG 3 N

        seed = self.folding_seed.unsqueeze(2).expand(bs * g, -1, self.num_coarse, -1)  # BG 2 M (S)
        seed = seed.reshape(bs * g, -1, self.num_fine).to(feature_global.device)  # BG 2 N

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_fine)  # BG 1024 N
        feat = torch.cat([feature_global, seed, point_feat], dim=1)  # BG C N

        center = coarse.unsqueeze(2).expand(-1, -1, self.grid_size ** 2, -1)  # BG (M) S 3
        center = center.reshape(bs * g, self.num_fine, 3).transpose(2, 1)  # BG 3 N

        fine = self.final_conv(feat) + center  # BG 3 N
        fine = fine.reshape(bs, g, 3, self.num_fine).transpose(-1, -2)
        coarse = coarse.reshape(bs, g, self.num_coarse, 3)
        return coarse, fine

knn = KNN(k=4, transpose_mode=False)

class DGCNN(nn.Module):
    def __init__(self, encoder_channel, output_channel):
        super().__init__()
        '''
        K has to be 16
        '''
        self.input_trans = nn.Conv1d(encoder_channel, 128, 1)

        self.layer1 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 256),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer2 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 512),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer3 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 512),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, 1024),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

        self.layer5 = nn.Sequential(nn.Conv1d(2304, output_channel, kernel_size=1, bias=False),
                                    nn.GroupNorm(4, output_channel),
                                    nn.LeakyReLU(negative_slope=0.2)
                                    )

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k):
        # coor: bs, 3, np, x: bs, c, np

        k = 4
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, f, coor):
        # f: B G C
        # coor: B G 3

        # bs 3 N   bs C N
        feature_list = []
        coor = coor.transpose(1, 2).contiguous()  # B 3 N
        f = f.transpose(1, 2).contiguous()  # B C N
        f = self.input_trans(f)  # B 128 N

        f = self.get_graph_feature(coor, f, coor, f)  # B 256 N k
        f = self.layer1(f)  # B 256 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 256 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)  # B 512 N k
        f = self.layer2(f)  # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)  # B 1024 N k
        f = self.layer3(f)  # B 512 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 512 N
        feature_list.append(f)

        f = self.get_graph_feature(coor, f, coor, f)  # B 1024 N k
        f = self.layer4(f)  # B 1024 N k
        f = f.max(dim=-1, keepdim=False)[0]  # B 1024 N
        feature_list.append(f)

        f = torch.cat(feature_list, dim=1)  # B 2304 N

        f = self.layer5(f)  # B C' N

        f = f.transpose(-1, -2)

        return f

class DiscreteVAE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.encoder_dims = config.encoder_dims
        self.tokens_dims = config.tokens_dims

        self.decoder_dims = config.decoder_dims
        self.num_tokens = config.num_tokens

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
        self.encoder = Encoder(encoder_channel=self.encoder_dims)
        self.dgcnn_1 = DGCNN(encoder_channel=self.encoder_dims, output_channel=self.num_tokens)
        self.codebook = nn.Parameter(torch.randn(self.num_tokens, self.tokens_dims))

        self.dgcnn_2 = DGCNN(encoder_channel=self.tokens_dims, output_channel=self.decoder_dims)
        self.decoder = Decoder(encoder_channel=self.decoder_dims, num_fine=self.group_size)

    def forward(self, inp, temperature=1., hard=False, **kwargs):
        neighborhood, center = self.group_divider(inp)
        logits = self.encoder(neighborhood)  # B G C
        logits = self.dgcnn_1(logits, center)  # B G N
        soft_one_hot = F.gumbel_softmax(logits, tau=temperature, dim=2, hard=hard)  # B G N
        sampled = torch.einsum('b g n, n c -> b g c', soft_one_hot, self.codebook)  # B G C
        feature = self.dgcnn_2(sampled, center)
        coarse, fine = self.decoder(feature)

        with torch.no_grad():
            whole_fine = (fine + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)
            whole_coarse = (coarse + center.unsqueeze(2)).reshape(inp.size(0), -1, 3)

        assert fine.size(2) == self.group_size
        ret = (whole_coarse, whole_fine, coarse, fine, neighborhood, logits)
        return ret

class Encoder(nn.Module):   ## Embedding module
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)


class Group(nn.Module):  # FPS + KNN
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3 or B N 6
            ---------------------------
            output: B G M 3 or B G M 6
            center : B G 3 or B G 6
        '''
        # for complexity calculation.
        # print(xyz.shape)
        # if xyz.shape[0] == 1:
        #     xyz = xyz[0]
        batch_size, num_points, _ = xyz.shape
        xyz_only = xyz[:, :, :3].clone().contiguous()
        attribute_only = xyz[:, :, 3:].clone().contiguous()
        # fps the centers out
        fps_idx, center = misc.fps(xyz_only, self.num_group) # B G 3
        center_attribute = pointnet2_utils.gather_operation(attribute_only.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()  ## B Gn 6
        # knn to get the neighborhood
        _, idx = self.knn(xyz_only, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        # neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        # neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 6).contiguous()

        neighborhood_xyz_only = xyz_only.view(batch_size * num_points, -1)[idx, :]
        neighborhood_xyz_only = neighborhood_xyz_only.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood_xyz_only = neighborhood_xyz_only - center.unsqueeze(2)
        attribute_dim = attribute_only.size(-1)
        neighborhood_attribute_only = attribute_only.view(batch_size * num_points, -1)[idx, :]
        neighborhood_attribute_only = neighborhood_attribute_only.view(batch_size, self.num_group, self.group_size, attribute_dim).contiguous()

        return neighborhood_xyz_only, neighborhood_attribute_only, center, center_attribute


## Transformers
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path = drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
                )
            for i in range(depth)])

    def forward(self, x, pos):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim=384, depth=4, num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, pos, return_token_num=-1):
        for _, block in enumerate(self.blocks):
            x = block(x + pos)
        if return_token_num == -1:
            x = self.head(self.norm(x))
        else:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x


# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio 
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoder(
            embed_dim = self.trans_dim,
            depth = self.depth,
            drop_path_rate = dpr,
            num_heads = self.num_heads,
        )

        self.norm = nn.LayerNorm(self.trans_dim)
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

    def _mask_center_block(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()
        # mask a continuous part
        mask_idx = []
        for points in center:
            # G 3
            points = points.unsqueeze(0)  # 1 G 3
            index = random.randint(0, points.size(1) - 1)
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1)  # 1 1 3 - 1 G 3 -> 1 G

            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0]  # G
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            mask = torch.zeros(len(idx))
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device)  # B G

        return bool_masked_pos

    def _mask_center_rand(self, center, noaug = False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        self.num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device) # B G

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)
        return x_vis, bool_masked_pos


# @MODELS.register_module()
# class MaskSurf_v2(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         print_log(f'[MaskSurf_v2] ', logger ='MaskSurf_v2')
#         self.config = config
#         self.all_patch = config.all_patch
#         self.trans_dim = config.transformer_config.trans_dim
#         self.MAE_encoder = MaskTransformer(config)
#         self.group_size = config.group_size
#         self.num_group = config.num_group
#         self.drop_path_rate = config.transformer_config.drop_path_rate
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
#         self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
#         # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
#         self.decoder_pos_embed = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.GELU(),
#             nn.Linear(128, self.trans_dim)
#         )
#         self.decoder_depth = config.transformer_config.decoder_depth
#         self.decoder_num_heads = config.transformer_config.decoder_num_heads
#         dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
#         self.MAE_decoder = TransformerDecoder(
#             embed_dim=self.trans_dim,
#             depth=self.decoder_depth,
#             drop_path_rate=dpr,
#             num_heads=self.decoder_num_heads,
#         )
#
#         print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
#         self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
#
#         self.coarse_pred = nn.Sequential(
#             nn.Linear(self.trans_dim, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 7 * self.num_group)
#         )
#
#         # prediction head
#         self.increase_dim = nn.Sequential(
#             # nn.Conv1d(self.trans_dim, 1024, 1),
#             # nn.BatchNorm1d(1024),
#             # nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(self.trans_dim, 7*self.group_size, 1)
#         )
#
#         # self.increase_dim2 = nn.Sequential(
#         #     # nn.Conv1d(self.trans_dim, 1024, 1),
#         #     # nn.BatchNorm1d(1024),
#         #     # nn.LeakyReLU(negative_slope=0.2),
#         #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
#         # )
#         trunc_normal_(self.mask_token, std=.02)
#         self.loss = config.loss
#         # loss
#         self.build_loss_func(self.loss)
#
#     def build_loss_func(self, loss_type):
#         if loss_type == "cdl1":
#             self.loss_func = ChamferDistanceL1().cuda()
#         elif loss_type =='cdl2':
#             self.loss_func = ChamferDistanceL2().cuda()
#         elif loss_type =='cdl2normal':
#             self.loss_func = ChamferDistanceL2_withnormal().cuda()
#         # elif loss_type == 'cdl2normall1':
#         #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
#         # elif loss_type == 'cdl2normal_normalindex':
#         #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
#         # elif loss_type =='cdl2normalstrict':
#         #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
#         # elif loss_type == 'cdl2normalstrict_normalindex':
#         #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
#         else:
#             raise NotImplementedError
#             # self.loss_func = emd().cuda()
#     def get_loss_value_weight(self):
#         return self.loss_concat, self.loss_weight_current
#
#     def load_model_from_ckpt(self, bert_ckpt_path):
#         if bert_ckpt_path is not None:
#             ckpt = torch.load(bert_ckpt_path)
#             base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#
#             for k in list(base_ckpt.keys()):
#                 if k.startswith('MAE_encoder') :
#                     base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
#                     del base_ckpt[k]
#                 elif k.startswith('base_model'):
#                     base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
#                     del base_ckpt[k]
#
#             incompatible = self.load_state_dict(base_ckpt, strict=False)
#
#             if incompatible.missing_keys:
#                 print_log('missing_keys', logger='Transformer')
#                 print_log(
#                     get_missing_parameters_message(incompatible.missing_keys),
#                     logger='Transformer'
#                 )
#             if incompatible.unexpected_keys:
#                 print_log('unexpected_keys', logger='Transformer')
#                 print_log(
#                     get_unexpected_parameters_message(incompatible.unexpected_keys),
#                     logger='Transformer'
#                 )
#
#             print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
#         else:
#             print_log('Training from scratch!!!', logger='Transformer')
#             self.apply(self._init_weights)
#
#     def forward(self, corrupted_pts, pts, vis = False, **kwargs):
#
#
#         # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
#         # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
#         # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
#         # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
#         # center = center_with_normal[:, :, :3].contiguous()
#         pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
#         position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
#         pts = torch.cat((pts, position.unsqueeze(-1)), -1)
#         ################ estimation position from pts.
#         neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
#
#         x_vis, mask = self.MAE_encoder(neighborhood, center)
#         B,_,C = x_vis.shape # B VIS C
#
#         global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
#         coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 7)  # B M C(3)
#         rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
#         rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
#         rebuild_coarse_position = coarse_surfel_cloud[:, :, 6:].contiguous()
#
#         gt_coarse_points = center
#         gt_coarse_normal = center_attribute[:, :, :3].contiguous()
#         gt_coarse_position = center_attribute[:, :, 3:].contiguous()
#
#         loss_coarse_xyz, loss_coarse_normal, loss_coarse_position = self.loss_func(rebuild_coarse_points, gt_coarse_points, \
#                                                rebuild_coarse_normal, gt_coarse_normal, torch.abs(rebuild_coarse_position), gt_coarse_position)
#
#         pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
#         pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
#
#         _,N,_ = pos_emd_mask.shape
#         mask_token = self.mask_token.expand(B, N, -1)
#         x_full = torch.cat([x_vis, mask_token], dim=1)
#         pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
#
#         if self.all_patch == 'True':
#             x_rec = self.MAE_decoder(x_full, pos_full)
#         else:
#             x_rec = self.MAE_decoder(x_full, pos_full, N)
#
#         B, M, C = x_rec.shape
#         rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 7)  # BM Gs 6
#         rebuild_points = rebuild_surfel[:, :, :3].contiguous()
#         rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
#         rebuild_position = rebuild_surfel[:, :, 6:].contiguous()
#         # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6
#
#
#         if self.all_patch == 'True':
#             gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
#                                    neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
#             gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
#                                    neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(B * M, -1, 4)
#         else:
#             gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
#             gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 4)  ## BM, Gs, 6
#             # gt_points = neighborhood[mask].reshape(B * M, -1, 3)
#
#         gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
#         gt_position = gt_attribute[:, :, 3:]  ## BM, Gs, 6
#         loss_xyz, loss_normal, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals, torch.abs(rebuild_position), gt_position)
#         loss_concat = torch.stack((loss_coarse_xyz, loss_coarse_normal, loss_coarse_position, loss_xyz, loss_normal, loss_position))
#         self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight),2)
#         self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[3].detach()
#         # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()
#
#         loss_point = loss_concat[0] * self.loss_weight_current[0] + loss_concat[3] * self.loss_weight_current[3]
#         loss_surfel = loss_concat[1] * self.loss_weight_current[1] + loss_concat[2] * self.loss_weight_current[2] +\
#                       loss_concat[4] * self.loss_weight_current[4] + loss_concat[5] * self.loss_weight_current[5]
#         self.loss_concat = loss_concat
#
#         if vis: #visualization
#             vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
#             full_vis = vis_points + center[~mask].unsqueeze(1)
#             full_rebuild = rebuild_points + center[mask].unsqueeze(1)
#             full = torch.cat([full_vis, full_rebuild], dim=0)
#             # full_points = torch.cat([rebuild_points,vis_points], dim=0)
#             full_center = torch.cat([center[mask], center[~mask]], dim=0)
#             # full = full_points + full_center.unsqueeze(1)
#             ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
#             ret1 = full.reshape(-1, 3).unsqueeze(0)
#             # return ret1, ret2
#             return ret1, ret2, full_center
#         else:
#             return loss_point, loss_surfel
#
#
# @MODELS.register_module()
# class MaskSurf_v2_onlynormal(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         print_log(f'[MaskSurf_v2] ', logger ='MaskSurf_v2')
#         self.config = config
#         self.all_patch = config.all_patch
#         self.trans_dim = config.transformer_config.trans_dim
#         self.MAE_encoder = MaskTransformer(config)
#         self.group_size = config.group_size
#         self.num_group = config.num_group
#         self.drop_path_rate = config.transformer_config.drop_path_rate
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
#         self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
#         # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
#         self.decoder_pos_embed = nn.Sequential(
#             nn.Linear(3, 128),
#             nn.GELU(),
#             nn.Linear(128, self.trans_dim)
#         )
#         self.decoder_depth = config.transformer_config.decoder_depth
#         self.decoder_num_heads = config.transformer_config.decoder_num_heads
#         dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
#         self.MAE_decoder = TransformerDecoder(
#             embed_dim=self.trans_dim,
#             depth=self.decoder_depth,
#             drop_path_rate=dpr,
#             num_heads=self.decoder_num_heads,
#         )
#
#         print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='Point_MAE')
#         self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
#
#         self.coarse_pred = nn.Sequential(
#             nn.Linear(self.trans_dim, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 1024),
#             nn.ReLU(inplace=True),
#             nn.Linear(1024, 7 * self.num_group)
#         )
#
#         # prediction head
#         self.increase_dim = nn.Sequential(
#             # nn.Conv1d(self.trans_dim, 1024, 1),
#             # nn.BatchNorm1d(1024),
#             # nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(self.trans_dim, 7*self.group_size, 1)
#         )
#
#         # self.increase_dim2 = nn.Sequential(
#         #     # nn.Conv1d(self.trans_dim, 1024, 1),
#         #     # nn.BatchNorm1d(1024),
#         #     # nn.LeakyReLU(negative_slope=0.2),
#         #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
#         # )
#         trunc_normal_(self.mask_token, std=.02)
#         self.loss = config.loss
#         # loss
#         self.build_loss_func(self.loss)
#
#     def build_loss_func(self, loss_type):
#         if loss_type == "cdl1":
#             self.loss_func = ChamferDistanceL1().cuda()
#         elif loss_type =='cdl2':
#             self.loss_func = ChamferDistanceL2().cuda()
#         elif loss_type =='cdl2normal':
#             self.loss_func = ChamferDistanceL2_withnormal().cuda()
#         # elif loss_type == 'cdl2normall1':
#         #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
#         # elif loss_type == 'cdl2normal_normalindex':
#         #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
#         # elif loss_type =='cdl2normalstrict':
#         #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
#         # elif loss_type == 'cdl2normalstrict_normalindex':
#         #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
#         else:
#             raise NotImplementedError
#             # self.loss_func = emd().cuda()
#     def get_loss_value_weight(self):
#         return self.loss_concat, self.loss_weight_current
#
#     def load_model_from_ckpt(self, bert_ckpt_path):
#         if bert_ckpt_path is not None:
#             ckpt = torch.load(bert_ckpt_path)
#             base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#
#             for k in list(base_ckpt.keys()):
#                 if k.startswith('MAE_encoder') :
#                     base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
#                     del base_ckpt[k]
#                 elif k.startswith('base_model'):
#                     base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
#                     del base_ckpt[k]
#
#             incompatible = self.load_state_dict(base_ckpt, strict=False)
#
#             if incompatible.missing_keys:
#                 print_log('missing_keys', logger='Transformer')
#                 print_log(
#                     get_missing_parameters_message(incompatible.missing_keys),
#                     logger='Transformer'
#                 )
#             if incompatible.unexpected_keys:
#                 print_log('unexpected_keys', logger='Transformer')
#                 print_log(
#                     get_unexpected_parameters_message(incompatible.unexpected_keys),
#                     logger='Transformer'
#                 )
#
#             print_log(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}', logger='Transformer')
#         else:
#             print_log('Training from scratch!!!', logger='Transformer')
#             self.apply(self._init_weights)
#
#     def forward(self, corrupted_pts, pts, vis = False, **kwargs):
#
#
#         # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
#         # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
#         # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
#         # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
#         # center = center_with_normal[:, :, :3].contiguous()
#         pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
#         position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
#         pts = torch.cat((pts, position.unsqueeze(-1)), -1)
#         ################ estimation position from pts.
#         neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
#
#         x_vis, mask = self.MAE_encoder(neighborhood, center)
#         B,_,C = x_vis.shape # B VIS C
#
#         global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
#         coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 7)  # B M C(3)
#         rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
#         rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
#         rebuild_coarse_position = coarse_surfel_cloud[:, :, 6:].contiguous()
#
#         gt_coarse_points = center
#         gt_coarse_normal = center_attribute[:, :, :3].contiguous()
#         gt_coarse_position = center_attribute[:, :, 3:].contiguous()
#
#         loss_coarse_xyz, loss_coarse_normal, loss_coarse_position = self.loss_func(rebuild_coarse_points, gt_coarse_points, \
#                                                rebuild_coarse_normal, gt_coarse_normal, rebuild_coarse_position, gt_coarse_position)
#
#         pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
#         pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
#
#         _,N,_ = pos_emd_mask.shape
#         mask_token = self.mask_token.expand(B, N, -1)
#         x_full = torch.cat([x_vis, mask_token], dim=1)
#         pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
#
#         if self.all_patch == 'True':
#             x_rec = self.MAE_decoder(x_full, pos_full)
#         else:
#             x_rec = self.MAE_decoder(x_full, pos_full, N)
#
#         B, M, C = x_rec.shape
#         rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 7)  # BM Gs 6
#         rebuild_points = rebuild_surfel[:, :, :3].contiguous()
#         rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
#         rebuild_position = rebuild_surfel[:, :, 6:].contiguous()
#         # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6
#
#
#         if self.all_patch == 'True':
#             gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
#                                    neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
#             gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
#                                    neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(B * M, -1, 4)
#         else:
#             gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
#             gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 4)  ## BM, Gs, 6
#             # gt_points = neighborhood[mask].reshape(B * M, -1, 3)
#
#         gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
#         gt_position = gt_attribute[:, :, 3:]  ## BM, Gs, 6
#         loss_xyz, loss_normal, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals, rebuild_position, gt_position)
#         loss_concat = torch.stack((loss_coarse_xyz, loss_coarse_normal, loss_coarse_position, loss_xyz, loss_normal, loss_position))
#         self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight),2)
#         self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[3].detach()
#         # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()
#
#         loss_point = loss_concat[0] * self.loss_weight_current[0] + loss_concat[3] * self.loss_weight_current[3]
#         loss_surfel = loss_concat[1] * self.loss_weight_current[1]  +\
#                       loss_concat[4] * self.loss_weight_current[4]
#         self.loss_concat = loss_concat
#
#         if vis: #visualization
#             vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
#             full_vis = vis_points + center[~mask].unsqueeze(1)
#             full_rebuild = rebuild_points + center[mask].unsqueeze(1)
#             full = torch.cat([full_vis, full_rebuild], dim=0)
#             # full_points = torch.cat([rebuild_points,vis_points], dim=0)
#             full_center = torch.cat([center[mask], center[~mask]], dim=0)
#             # full = full_points + full_center.unsqueeze(1)
#             ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
#             ret1 = full.reshape(-1, 3).unsqueeze(0)
#             # return ret1, ret2
#             return ret1, ret2, full_center
#         else:
#             return loss_point, loss_surfel
#
#


@MODELS.register_module()
class MaskSurf_v2_local_global_point(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskSurf_v2] ', logger='MaskSurf_v2')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
        # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7 * self.num_group)
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 7 * self.group_size, 1)
        )

        # self.increase_dim2 = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl2normal':
            self.loss_func = ChamferDistanceL2_withnormal().cuda()
        # elif loss_type == 'cdl2normall1':
        #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
        # elif loss_type == 'cdl2normal_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
        # elif loss_type =='cdl2normalstrict':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
        # elif loss_type == 'cdl2normalstrict_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def get_loss_value_weight(self):
        return self.loss_concat, self.loss_weight_current

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
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

    def forward(self, corrupted_pts, pts, vis=False, return_feat=False, **kwargs):

        # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
        # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
        # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
        # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
        # center = center_with_normal[:, :, :3].contiguous()
        if return_feat:
            pts = torch.cat((pts, pts), -1)
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
            return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
        else:
            pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
            position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
            pts = torch.cat((pts, position.unsqueeze(-1)), -1)
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
            coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 7)  # B M C(3)
            rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
            rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
            rebuild_coarse_position = coarse_surfel_cloud[:, :, 6:].contiguous()
            gt_coarse_points = center
            gt_coarse_normal = center_attribute[:, :, :3].contiguous()
            gt_coarse_position = center_attribute[:, :, 3:].contiguous()

            loss_coarse_xyz, loss_coarse_normal, loss_coarse_position = self.loss_func(rebuild_coarse_points,
                                                                                       gt_coarse_points, \
                                                                                       rebuild_coarse_normal,
                                                                                       gt_coarse_normal,
                                                                                       torch.abs(rebuild_coarse_position),
                                                                                       gt_coarse_position)

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 7)  # BM Gs 6
            rebuild_points = rebuild_surfel[:, :, :3].contiguous()
            rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
            rebuild_position = rebuild_surfel[:, :, 6:].contiguous()
            # print(rebuild_position)
            # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6

            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
                gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
                                          neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(
                    B * M, -1, 4)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
                gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 4)  ## BM, Gs, 6
                # gt_points = neighborhood[mask].reshape(B * M, -1, 3)

            gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
            gt_position = gt_attribute[:, :, 3:]  ## BM, Gs, 6
            loss_xyz, loss_normal, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals,
                                                                  torch.abs(rebuild_position), gt_position)
            loss_concat = torch.stack(
                (loss_coarse_xyz, loss_coarse_normal, loss_coarse_position, loss_xyz, loss_normal, loss_position))
            # print(loss_concat)
            # self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight), 2)
            self.loss_weight_current = 1.0 / loss_concat.detach()
            self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[3].detach()
            # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()

            loss_point = loss_concat[0] * self.loss_weight_current[0] + loss_concat[3] * self.loss_weight_current[3]
            # loss_surfel = loss_concat[1] * self.loss_weight_current[1] + loss_concat[2] * self.loss_weight_current[2] + \
            #               loss_concat[4] * self.loss_weight_current[4] + loss_concat[5] * self.loss_weight_current[5]
            self.loss_concat = loss_concat

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                return loss_point, torch.zeros(1).to(loss_point.device)

@MODELS.register_module()
class MaskSurf_v2_local_point_only(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskSurf_v2] ', logger='MaskSurf_v2')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
        # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7 * self.num_group)
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 7 * self.group_size, 1)
        )

        # self.increase_dim2 = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl2normal':
            self.loss_func = ChamferDistanceL2_withnormal().cuda()
        # elif loss_type == 'cdl2normall1':
        #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
        # elif loss_type == 'cdl2normal_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
        # elif loss_type =='cdl2normalstrict':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
        # elif loss_type == 'cdl2normalstrict_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def get_loss_value_weight(self):
        return self.loss_concat, self.loss_weight_current

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
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

    def forward(self, corrupted_pts, pts, vis=False, return_feat = False, **kwargs):

        # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
        # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
        # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
        # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
        # center = center_with_normal[:, :, :3].contiguous()
        if return_feat:
            pts = torch.cat((pts, pts), -1)
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
            return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
        else:
            pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
            position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
            pts = torch.cat((pts, position.unsqueeze(-1)), -1)
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug=False)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
            coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 7)  # B M C(3)
            rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
            rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
            rebuild_coarse_position = coarse_surfel_cloud[:, :, 6:].contiguous()

            gt_coarse_points = center
            gt_coarse_normal = center_attribute[:, :, :3].contiguous()
            gt_coarse_position = center_attribute[:, :, 3:].contiguous()

            loss_coarse_xyz, loss_coarse_normal, loss_coarse_position = self.loss_func(rebuild_coarse_points,
                                                                                       gt_coarse_points, \
                                                                                       rebuild_coarse_normal,
                                                                                       gt_coarse_normal,
                                                                                       torch.abs(rebuild_coarse_position),
                                                                                       gt_coarse_position)

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 7)  # BM Gs 6
            rebuild_points = rebuild_surfel[:, :, :3].contiguous()
            rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
            rebuild_position = rebuild_surfel[:, :, 6:].contiguous()
            # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6

            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
                gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
                                          neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(
                    B * M, -1, 4)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
                gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 4)  ## BM, Gs, 6
                # gt_points = neighborhood[mask].reshape(B * M, -1, 3)

            gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
            gt_position = gt_attribute[:, :, 3:]  ## BM, Gs, 6
            loss_xyz, loss_normal, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals,
                                                                  torch.abs(rebuild_position), gt_position)
            # loss_concat = torch.stack(
            #     (loss_coarse_xyz, loss_coarse_normal, loss_coarse_position, loss_xyz, loss_normal, loss_position))
            # # self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight), 2)
            # self.loss_weight_current = 1.0 / loss_concat.detach()
            # self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[3].detach()
            # # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()
            #
            # loss_point = loss_concat[3] * self.loss_weight_current[3]
            # self.loss_concat = loss_concat
            loss_point = loss_xyz

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                return loss_point, torch.zeros(1).to(loss_point.device)

@MODELS.register_module()
class MaskSurf_v2_local_point_normal(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskSurf_v2] ', logger='MaskSurf_v2')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
        # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7 * self.num_group)
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 7 * self.group_size, 1)
        )

        # self.increase_dim2 = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl2normal':
            self.loss_func = ChamferDistanceL2_withnormal().cuda()
        # elif loss_type == 'cdl2normall1':
        #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
        # elif loss_type == 'cdl2normal_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
        # elif loss_type =='cdl2normalstrict':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
        # elif loss_type == 'cdl2normalstrict_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def get_loss_value_weight(self):
        return self.loss_concat, self.loss_weight_current

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
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

    def forward(self, corrupted_pts, pts, vis=False, return_feat=False, **kwargs):

        # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
        # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
        # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
        # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
        # center = center_with_normal[:, :, :3].contiguous()
        if return_feat:
            pts = torch.cat((pts, pts), -1)
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
            return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
        else:
            pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
            position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
            pts = torch.cat((pts, position.unsqueeze(-1)), -1)
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
            coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 7)  # B M C(3)
            rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
            rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
            rebuild_coarse_position = coarse_surfel_cloud[:, :, 6:].contiguous()

            gt_coarse_points = center
            gt_coarse_normal = center_attribute[:, :, :3].contiguous()
            gt_coarse_position = center_attribute[:, :, 3:].contiguous()

            loss_coarse_xyz, loss_coarse_normal, loss_coarse_position = self.loss_func(rebuild_coarse_points,
                                                                                       gt_coarse_points, \
                                                                                       rebuild_coarse_normal,
                                                                                       gt_coarse_normal,
                                                                                       torch.abs(rebuild_coarse_position),
                                                                                       gt_coarse_position)

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 7)  # BM Gs 6
            rebuild_points = rebuild_surfel[:, :, :3].contiguous()
            rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
            rebuild_position = rebuild_surfel[:, :, 6:].contiguous()
            # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6

            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
                gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
                                          neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(
                    B * M, -1, 4)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
                gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 4)  ## BM, Gs, 6
                # gt_points = neighborhood[mask].reshape(B * M, -1, 3)

            gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
            gt_position = gt_attribute[:, :, 3:]  ## BM, Gs, 6
            loss_xyz, loss_normal, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals,
                                                                  torch.abs(rebuild_position), gt_position)
            # loss_concat = torch.stack(
            #     (loss_coarse_xyz, loss_coarse_normal, loss_coarse_position, loss_xyz, loss_normal, loss_position))
            # # self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight), 2)
            # self.loss_weight_current = 1.0 / loss_concat.detach()
            # self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[3].detach()
            # # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()
            # loss_point =  loss_concat[3] * self.loss_weight_current[3]
            # loss_surfel = loss_concat[4] * self.loss_weight_current[4]
            # self.loss_concat = loss_concat
            loss_point = loss_xyz
            loss_surfel = loss_normal

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                return loss_point, loss_surfel

@MODELS.register_module()
class MaskSurf_v2_local_point_position(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskSurf_v2] ', logger='MaskSurf_v2')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
        # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7 * self.num_group)
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 7 * self.group_size, 1)
        )

        # self.increase_dim2 = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl2normal':
            self.loss_func = ChamferDistanceL2_withnormal().cuda()
        # elif loss_type == 'cdl2normall1':
        #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
        # elif loss_type == 'cdl2normal_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
        # elif loss_type =='cdl2normalstrict':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
        # elif loss_type == 'cdl2normalstrict_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def get_loss_value_weight(self):
        return self.loss_concat, self.loss_weight_current

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
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

    def forward(self, corrupted_pts, pts, vis=False, return_feat=False, **kwargs):

        # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
        # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
        # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
        # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
        # center = center_with_normal[:, :, :3].contiguous()
        if return_feat:
            pts = torch.cat((pts, pts), -1)
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
            return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
        else:
            pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
            position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
            pts = torch.cat((pts, position.unsqueeze(-1)), -1)
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
            coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 7)  # B M C(3)
            rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
            rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
            rebuild_coarse_position = coarse_surfel_cloud[:, :, 6:].contiguous()

            gt_coarse_points = center
            gt_coarse_normal = center_attribute[:, :, :3].contiguous()
            gt_coarse_position = center_attribute[:, :, 3:].contiguous()

            loss_coarse_xyz, loss_coarse_normal, loss_coarse_position = self.loss_func(rebuild_coarse_points,
                                                                                       gt_coarse_points, \
                                                                                       rebuild_coarse_normal,
                                                                                       gt_coarse_normal,
                                                                                       torch.abs(rebuild_coarse_position),
                                                                                       gt_coarse_position)

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 7)  # BM Gs 6
            rebuild_points = rebuild_surfel[:, :, :3].contiguous()
            rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
            rebuild_position = rebuild_surfel[:, :, 6:].contiguous()
            # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6

            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
                gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
                                          neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(
                    B * M, -1, 4)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
                gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 4)  ## BM, Gs, 6
                # gt_points = neighborhood[mask].reshape(B * M, -1, 3)

            gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
            gt_position = gt_attribute[:, :, 3:]  ## BM, Gs, 6
            loss_xyz, loss_normal, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals,
                                                                  torch.abs(rebuild_position), gt_position)
            # loss_concat = torch.stack(
            #     (loss_coarse_xyz, loss_coarse_normal, loss_coarse_position, loss_xyz, loss_normal, loss_position))
            # # self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight), 2)
            # self.loss_weight_current = 1.0 / loss_concat.detach()
            # self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[3].detach()
            # # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()
            #
            # loss_point = loss_concat[3] * self.loss_weight_current[3]
            # loss_surfel = loss_concat[5] * self.loss_weight_current[5]
            # self.loss_concat = loss_concat
            loss_point = loss_xyz
            loss_surfel = loss_position

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                return loss_point, loss_surfel

@MODELS.register_module()
class MaskSurf_v2_local_point_normal_position(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskSurf_v2] ', logger='MaskSurf_v2')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.weight_dis_vs_normal = config.weight_dis_vs_normal
        self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
        # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 7 * self.num_group)
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 7 * self.group_size, 1)
        )

        # self.increase_dim2 = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl2normal':
            self.loss_func = ChamferDistanceL2_withnormal().cuda()
        # elif loss_type == 'cdl2normall1':
        #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
        # elif loss_type == 'cdl2normal_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
        # elif loss_type =='cdl2normalstrict':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
        # elif loss_type == 'cdl2normalstrict_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def get_loss_value_weight(self):
        return self.loss_concat, self.loss_weight_current

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
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

    def forward(self, corrupted_pts, pts, vis=False, return_feat=False, **kwargs):

        # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
        # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
        # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
        # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
        # center = center_with_normal[:, :, :3].contiguous()
        if return_feat:
            pts = torch.cat((pts, pts), -1)
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
            return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
        else:
            pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
            position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
            pts = torch.cat((pts, position.unsqueeze(-1)), -1)
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
            coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 7)  # B M C(3)
            rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
            rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
            rebuild_coarse_position = coarse_surfel_cloud[:, :, 6:].contiguous()

            gt_coarse_points = center
            gt_coarse_normal = center_attribute[:, :, :3].contiguous()
            gt_coarse_position = center_attribute[:, :, 3:].contiguous()

            loss_coarse_xyz, loss_coarse_normal, loss_coarse_position = self.loss_func(rebuild_coarse_points,
                                                                                       gt_coarse_points, \
                                                                                       rebuild_coarse_normal,
                                                                                       gt_coarse_normal,
                                                                                       torch.abs(rebuild_coarse_position),
                                                                                       gt_coarse_position)

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 7)  # BM Gs 6
            rebuild_points = rebuild_surfel[:, :, :3].contiguous()
            rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
            rebuild_position = rebuild_surfel[:, :, 6:].contiguous()
            # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6

            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
                gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
                                          neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(
                    B * M, -1, 4)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
                gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 4)  ## BM, Gs, 6
                # gt_points = neighborhood[mask].reshape(B * M, -1, 3)

            gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
            gt_position = gt_attribute[:, :, 3:]  ## BM, Gs, 6
            loss_xyz, loss_normal, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals,
                                                                  torch.abs(rebuild_position), gt_position)
            loss_concat = torch.stack(
                (loss_coarse_xyz, loss_coarse_normal, loss_coarse_position, loss_xyz, loss_normal, loss_position))
            # # self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight), 2)
            # self.loss_weight_current = 1.0 / loss_concat.detach()
            # self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[3].detach()
            # # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()
            #
            # loss_point =  loss_concat[3] * self.loss_weight_current[3]
            # loss_surfel = loss_concat[4] * self.loss_weight_current[4] + loss_concat[5] * self.loss_weight_current[5]
            self.loss_concat = loss_concat
            self.loss_weight_current = loss_concat
            loss_point = loss_xyz
            loss_surfel = loss_normal + loss_position * self.weight_dis_vs_normal

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                return loss_point, loss_surfel


@MODELS.register_module()
class MaskSurf_v2_token_dis(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskSurf_v2] ', logger='MaskSurf_v2')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
        # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 8192, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

        self.dvae = DiscreteVAE(config.dvae_config)
        self._prepare_dvae()

        for param in self.dvae.parameters():
            param.requires_grad = False

    def _prepare_dvae(self):
        dvae_ckpt = self.config.dvae_config.ckpt
        ckpt = torch.load(dvae_ckpt, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.dvae.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for dvae from {dvae_ckpt}', logger='Point_BERT')

    def build_loss_func(self, loss_type):
        self.loss_func = nn.CrossEntropyLoss()


    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
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

    def forward(self, corrupted_pts, pts, vis=False, return_feat = False, **kwargs):

        if return_feat:
            pts = torch.cat((pts, pts), -1)
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
            return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
        else:
            pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
            position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
            pts = torch.cat((pts, position.unsqueeze(-1)), -1)
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug=False)
            B, _, C = x_vis.shape  # B VIS C

            # global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
            # coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 7)  # B M C(3)
            # rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
            # rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
            # rebuild_coarse_position = coarse_surfel_cloud[:, :, 6:].contiguous()
            #
            # gt_coarse_points = center
            # gt_coarse_normal = center_attribute[:, :, :3].contiguous()
            # gt_coarse_position = center_attribute[:, :, 3:].contiguous()
            #
            # loss_coarse_xyz, loss_coarse_normal, loss_coarse_position = self.loss_func(rebuild_coarse_points,
            #                                                                            gt_coarse_points, \
            #                                                                            rebuild_coarse_normal,
            #                                                                            gt_coarse_normal,
            #                                                                            torch.abs(rebuild_coarse_position),
            #                                                                            gt_coarse_position)

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_token_label = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1)  # BM Gs 6
            # ipdb.set_trace()
            if self.all_patch == 'True':
                with torch.no_grad():
                    gt_logits = self.dvae.encoder(neighborhood)
                    gt_logits = self.dvae.dgcnn_1(gt_logits, center)  # B G N
                    dvae_label = gt_logits.argmax(-1).long()  # B G
                # gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                #                        neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
                # gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
                #                           neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(
                #     B * M, -1, 4)
            else:
                with torch.no_grad():
                    gt_logits = self.dvae.encoder(neighborhood)
                    gt_logits = self.dvae.dgcnn_1(gt_logits, center)  # B G N
                    dvae_label = gt_logits.argmax(-1).long()  # B G
                    dvae_label = dvae_label[mask]
            loss_point = self.loss_func(rebuild_token_label,dvae_label)
                # gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
                # gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 4)  ## BM, Gs, 6
                # gt_points = neighborhood[mask].reshape(B * M, -1, 3)

            # gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
            # gt_position = gt_attribute[:, :, 3:]  ## BM, Gs, 6
            # loss_xyz, loss_normal, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals,
            #                                                       torch.abs(rebuild_position), gt_position)
            # loss_concat = torch.stack(
            #     (loss_coarse_xyz, loss_coarse_normal, loss_coarse_position, loss_xyz, loss_normal, loss_position))
            # # self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight), 2)
            # self.loss_weight_current = 1.0 / loss_concat.detach()
            # self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[3].detach()
            # # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()
            #
            # loss_point = loss_concat[3] * self.loss_weight_current[3]
            # self.loss_concat = loss_concat

            if vis:  # visualization
                raise NotImplementedError
            else:
                return loss_point, torch.zeros(1).to(loss_point.device)



@MODELS.register_module()
class MaskSurf_v2_local_point_normal_position_curve(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskSurf_v2] ', logger='MaskSurf_v2')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
        # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 8 * self.num_group)
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 8 * self.group_size, 1)
        )

        # self.increase_dim2 = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl2normal':
            self.loss_func = ChamferDistanceL2_withnormal().cuda()
        # elif loss_type == 'cdl2normall1':
        #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
        # elif loss_type == 'cdl2normal_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
        # elif loss_type =='cdl2normalstrict':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
        # elif loss_type == 'cdl2normalstrict_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def get_loss_value_weight(self):
        return self.loss_concat, self.loss_weight_current

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
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

    def forward(self, corrupted_pts, pts, vis=False, return_feat=False, **kwargs):

        # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
        # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
        # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
        # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
        # center = center_with_normal[:, :, :3].contiguous()
        if return_feat:
            pts = torch.cat((pts, pts), -1)
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
            return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
        else:
            pts = pts[:, :, :7].contiguous()  #### xyz & estimated normal & curve. curve always be positive.
            position = torch.abs((pts[:, :, :3] * pts[:, :, 3:6]).sum(-1))
            pts = torch.cat((pts, position.unsqueeze(-1)), -1) ## xyz, abc, c, p
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
            coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 8)  # B M C(3)
            rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
            rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
            rebuild_coarse_curve = coarse_surfel_cloud[:, :, 6:7].contiguous()
            rebuild_coarse_position = coarse_surfel_cloud[:, :, 7:].contiguous()

            gt_coarse_points = center
            gt_coarse_normal = center_attribute[:, :, :3].contiguous()
            gt_coarse_curve = center_attribute[:, :, 3:4].contiguous()
            gt_coarse_position = center_attribute[:, :, 4:].contiguous()

            loss_coarse_xyz, loss_coarse_normal, loss_coarse_curve, loss_coarse_position = self.loss_func(rebuild_coarse_points,
                                                                                       gt_coarse_points, \
                                                                                       rebuild_coarse_normal,
                                                                                       gt_coarse_normal,
                                                                                       torch.abs(rebuild_coarse_curve),
                                                                                       gt_coarse_curve,
                                                                                       torch.abs(rebuild_coarse_position),
                                                                                       gt_coarse_position
                                                                                       )

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 8)  # BM Gs 6
            rebuild_points = rebuild_surfel[:, :, :3].contiguous()
            rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
            rebuild_curve = rebuild_surfel[:, :, 6:7].contiguous()
            rebuild_position = rebuild_surfel[:, :, 7:].contiguous()
            # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6

            if self.all_patch == 'True':
                raise NotImplementedError
                # gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                #                        neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
                # gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
                #                           neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(
                #     B * M, -1, 4)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
                gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 5)  ## BM, Gs, 6

            gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
            gt_curve = gt_attribute[:, :, 3:4]  ## BM, Gs, 6
            gt_position = gt_attribute[:, :, 4:]  ## BM, Gs, 6
            loss_xyz, loss_normal, loss_curve, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals,
                                                                  torch.abs(rebuild_curve), gt_curve,
                                                                  torch.abs(rebuild_position), gt_position)
            loss_concat = torch.stack(
                (loss_coarse_xyz, loss_coarse_normal, loss_coarse_curve, loss_coarse_position, loss_xyz, loss_normal, loss_curve, loss_position))
            # self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight), 2)
            self.loss_weight_current = 1.0 / loss_concat.detach()
            self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[4].detach()
            # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()

            loss_point =  loss_concat[4] * self.loss_weight_current[4]
            loss_surfel = loss_concat[5] * self.loss_weight_current[5] + loss_concat[6] * self.loss_weight_current[6] \
                          + loss_concat[7] * self.loss_weight_current[7]
            self.loss_concat = loss_concat

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                return loss_point, loss_surfel * 0.33


@MODELS.register_module()
class MaskSurf_v2_local_point_curve(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskSurf_v2] ', logger='MaskSurf_v2')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
        # self.loss_weight = nn.Parameter(torch.FloatTensor([-0.9, 0.8, 1.6, -1.8, 0.8, 1.6]))  ## log_sigma
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[Point_MAE] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='Point_MAE')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 8 * self.num_group)
        )

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 8 * self.group_size, 1)
        )

        # self.increase_dim2 = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type == 'cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        elif loss_type == 'cdl2normal':
            self.loss_func = ChamferDistanceL2_withnormal().cuda()
        # elif loss_type == 'cdl2normall1':
        #     self.loss_func = ChamferDistanceL2_withnormalL1().cuda()
        # elif loss_type == 'cdl2normal_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_normalindex().cuda()
        # elif loss_type =='cdl2normalstrict':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict().cuda()
        # elif loss_type == 'cdl2normalstrict_normalindex':
        #     self.loss_func = ChamferDistanceL2_withnormal_strict_normalindex().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def get_loss_value_weight(self):
        return self.loss_concat, self.loss_weight_current

    def load_model_from_ckpt(self, bert_ckpt_path):
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}

            for k in list(base_ckpt.keys()):
                if k.startswith('MAE_encoder'):
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

    def forward(self, corrupted_pts, pts, vis=False, return_feat=False, **kwargs):

        # print_log(f'loss weight{self.loss_weight[0].item()}, {self.loss_weight[1].item()}, {self.loss_weight[2].item()} \
        # {self.loss_weight[3].item()}, {self.loss_weight[4].item()}, {self.loss_weight[5].item()}...', logger='Point_MAE')
        # neighborhood_withnormal, center_with_normal = self.group_divider(pts)
        # neighborhood = neighborhood_withnormal[:, :, :, :3].contiguous()
        # center = center_with_normal[:, :, :3].contiguous()
        if return_feat:
            pts = torch.cat((pts, pts), -1)
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
            return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
        else:
            pts = pts[:, :, :7].contiguous()  #### xyz & estimated normal & curve. curve always be positive.
            position = torch.abs((pts[:, :, :3] * pts[:, :, 3:6]).sum(-1))
            pts = torch.cat((pts, position.unsqueeze(-1)), -1) ## xyz, abc, c, p
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
            coarse_surfel_cloud = self.coarse_pred(global_feature).reshape(B, -1, 8)  # B M C(3)
            rebuild_coarse_points = coarse_surfel_cloud[:, :, :3].contiguous()
            rebuild_coarse_normal = coarse_surfel_cloud[:, :, 3:6].contiguous()
            rebuild_coarse_curve = coarse_surfel_cloud[:, :, 6:7].contiguous()
            rebuild_coarse_position = coarse_surfel_cloud[:, :, 7:].contiguous()

            gt_coarse_points = center
            gt_coarse_normal = center_attribute[:, :, :3].contiguous()
            gt_coarse_curve = center_attribute[:, :, 3:4].contiguous()
            gt_coarse_position = center_attribute[:, :, 4:].contiguous()

            loss_coarse_xyz, loss_coarse_normal, loss_coarse_curve, loss_coarse_position = self.loss_func(rebuild_coarse_points,
                                                                                       gt_coarse_points, \
                                                                                       rebuild_coarse_normal,
                                                                                       gt_coarse_normal,
                                                                                       torch.abs(rebuild_coarse_curve),
                                                                                       gt_coarse_curve,
                                                                                       torch.abs(rebuild_coarse_position),
                                                                                       gt_coarse_position
                                                                                       )

            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_surfel = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 8)  # BM Gs 6
            rebuild_points = rebuild_surfel[:, :, :3].contiguous()
            rebuild_normal = rebuild_surfel[:, :, 3:6].contiguous()
            rebuild_curve = rebuild_surfel[:, :, 6:7].contiguous()
            rebuild_position = rebuild_surfel[:, :, 7:].contiguous()
            # rebuild_normal = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # BM Gs 6

            if self.all_patch == 'True':
                raise NotImplementedError
                # gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                #                        neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1, 3)
                # gt_attribute = torch.cat((neighborhood_attribute[~mask].reshape(B, -1, self.group_size, 4),
                #                           neighborhood_attribute[mask].reshape(B, -1, self.group_size, 4)), dim=1).reshape(
                #     B * M, -1, 4)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)  ## BM, Gs, 6
                gt_attribute = neighborhood_attribute[mask].reshape(B * M, -1, 5)  ## BM, Gs, 6

            gt_normals = gt_attribute[:, :, :3]  ## BM, Gs, 6
            gt_curve = gt_attribute[:, :, 3:4]  ## BM, Gs, 6
            gt_position = gt_attribute[:, :, 4:]  ## BM, Gs, 6
            loss_xyz, loss_normal, loss_curve, loss_position = self.loss_func(rebuild_points, gt_points, rebuild_normal, gt_normals,
                                                                  torch.abs(rebuild_curve), gt_curve,
                                                                  torch.abs(rebuild_position), gt_position)
            loss_concat = torch.stack(
                (loss_coarse_xyz, loss_coarse_normal, loss_coarse_curve, loss_coarse_position, loss_xyz, loss_normal, loss_curve, loss_position))
            # self.loss_weight_current = 0.5 / torch.pow(torch.exp(self.loss_weight), 2)
            self.loss_weight_current = 1.0 / loss_concat.detach()
            self.loss_weight_current = self.loss_weight_current / self.loss_weight_current[4].detach()
            # loss = (loss_concat * self.loss_weight_current).sum() + self.loss_weight.sum()

            loss_point =  loss_concat[4] * self.loss_weight_current[4]
            loss_surfel = loss_concat[7] * self.loss_weight_current[7]
            self.loss_concat = loss_concat

            if vis:  # visualization
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                # full_points = torch.cat([rebuild_points,vis_points], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                # full = full_points + full_center.unsqueeze(1)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                ret1 = full.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return ret1, ret2, full_center
            else:
                return loss_point, loss_surfel



