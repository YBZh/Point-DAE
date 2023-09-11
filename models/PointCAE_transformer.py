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
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL2, ChamferDistanceL2_corase2fine
import ipdb
from datasets.corrupt_util_tensor import corrupt_data
import itertools
from .detr.build import build_encoder as build_encoder_3detr, build_preencoder as build_preencoder_3detr
import pointnet2_utils

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

class DummyGroup(Group):
    def forward(self, xyz):
        center = xyz.clone()
        neighborhood = torch.zeros_like(xyz).unsqueeze(2)
        return neighborhood, center
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


class TransformerEncoderOnePE(nn.Module):
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
            for i in range(depth)])

    def forward(self, x, pos):
        x = x + pos
        for _, block in enumerate(self.blocks):
            x = block(x)
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

class TransformerDecoderOnePE(nn.Module):
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
        x = x + pos
        for _, block in enumerate(self.blocks):
            x = block(x)
        if return_token_num == -1:
            x = self.head(self.norm(x))
        else:
            x = self.head(self.norm(x[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return x

class TransformerDecoderOnePEMultiX(nn.Module):
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
            pos = block(x+pos)
        if return_token_num == -1:
            pos = self.head(self.norm(pos))
        else:
            pos = self.head(self.norm(pos[:, -return_token_num:]))  # only return the mask tokens predict pixel
        return pos

# Pretrain model
class MaskTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        self.mask_ratio = config.transformer_config.mask_ratio
        self.rand_ratio = config.transformer_config.rand_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth 
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads 
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.num_group = config.num_group
        self.group_size = config.group_size

        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        if self.enc_arch == '3detr':
            self.encoder = build_preencoder_3detr(num_group=self.num_group, group_size=self.group_size, dim=self.encoder_dims)
        else:
            self.encoder = Encoder(encoder_channel = self.encoder_dims)

        self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        if self.enc_arch == '3detr':
            self.blocks = build_encoder_3detr(
                ndim=self.trans_dim,
                nhead=self.num_heads,
                nlayers=self.depth
            )
        else:
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

        if self.rand_ratio == 'True':
            self.mask_ratio = torch.FloatTensor(1).uniform_(0.5, 0.8)
            self.mask_ratio = self.mask_ratio.item()
        # print(self.mask_ratio)
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

    def forward(self, neighborhood, center_init, noaug = False):
        # ipdb.set_trace()
        if self.enc_arch == '3detr':
            #### the center is the vanilla input, B * 20K * 3
            #### since the point grouping is conducted here, therefore, the grouped cluster should be returned.
            pre_enc_xyz, group_input_tokens, pre_enc_inds = self.encoder(center_init)
            group_input_tokens = group_input_tokens.permute(0, 2, 1)
            ### size to check:
            ### group center: pre_enc_xyz, B * 2048 * 3
            ### group feature: group_input_tokens, B * 2048 * d
            ### index of group center: pre_enc_xyz, B * 2048
            center = pre_enc_xyz
        else:
            group_input_tokens = self.encoder(neighborhood) #  B G C
            center = center_init
            # group_input_tokens = self.encoder(neighborhood)  #  B G C
        # generate mask
        if self.mask_type == 'rand':
            bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        else:
            raise NotImplementedError
            # bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens[~bool_masked_pos].reshape(batch_size, -1, C)  ## B * visual group size * d
        # add pos embedding
        # mask pos center
        masked_center = center[~bool_masked_pos].reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # return x_vis, bool_masked_pos, pre_enc_inds  ## this is ok, but ththe following block has problem
        # transformer
        # ipdb.set_trace()
        if self.enc_arch == '3detr':
            ### with pe, but 3detr has no pe in the encoder.
            # x_vis = self.blocks(x_vis.transpose(0, 1), pos=pos.transpose(0, 1))[1].transpose(0, 1)
            ### no pe, same to 3detr backbone.
            x_vis = self.blocks(x_vis.transpose(0, 1))[1].transpose(0, 1)
            ## B * vis_group_size* D;  B*all_group_size; B*all_group_size
            return x_vis, bool_masked_pos, pre_enc_inds ## additional group center and neighborhood.
        else:
            x_vis = self.blocks(x_vis, pos)
            x_vis = self.norm(x_vis)

            return x_vis, bool_masked_pos


# Pretrain model, Not mask input.
class NormalTransformer(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        # self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        # self.mask_type = config.transformer_config.mask_type

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

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        # if self.mask_type == 'rand':
        #     bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        # else:
        #     bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens.reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center.reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis

# Pretrain model, Not mask input.
class NormalTransformerOnePE(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        # define the transformer argparse
        # self.mask_ratio = config.transformer_config.mask_ratio
        self.trans_dim = config.transformer_config.trans_dim
        self.depth = config.transformer_config.depth
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.num_heads = config.transformer_config.num_heads
        print_log(f'[args] {config.transformer_config}', logger = 'Transformer')
        # embedding
        self.encoder_dims =  config.transformer_config.encoder_dims
        self.encoder = Encoder(encoder_channel = self.encoder_dims)

        # self.mask_type = config.transformer_config.mask_type

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim),
        )

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.depth)]
        self.blocks = TransformerEncoderOnePE(
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

    def forward(self, neighborhood, center, noaug = False):
        # generate mask
        # if self.mask_type == 'rand':
        #     bool_masked_pos = self._mask_center_rand(center, noaug = noaug) # B G
        # else:
        #     bool_masked_pos = self._mask_center_block(center, noaug = noaug)

        group_input_tokens = self.encoder(neighborhood)  #  B G C

        batch_size, seq_len, C = group_input_tokens.size()

        x_vis = group_input_tokens.reshape(batch_size, -1, C)
        # add pos embedding
        # mask pos center
        masked_center = center.reshape(batch_size, -1, 3)
        pos = self.pos_embed(masked_center)

        # transformer
        x_vis = self.blocks(x_vis, pos)
        x_vis = self.norm(x_vis)

        return x_vis

### 给定 GT PE, 来重建 其对应位置的 normalized local patch. 这里还是在重建 local structure.
### 如果输入的是
@MODELS.register_module()
class PointCAE_transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer] ', logger ='PointCAE_transformer')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.all_patch = config.all_patch
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood = neighborhood - center.unsqueeze(2)
        transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            # x_rec = self.MAE_decoder(x_full, pos_full, N)
            if self.all_patch == 'True':
                x_rec = self.MAE_decoder(x_full, pos_full)
            else:
                x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1,3)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            # gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)

        else:
            x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)

            # _, N, _ = pos_emd_mask.shape
            # mask_token = self.mask_token.expand(B, N, -1)
            x_full = x_vis
            pos_full = pos_emd_vis

            x_rec = self.MAE_decoder(x_full, pos_full)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = neighborhood.reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)

        return loss1, torch.zeros(1).to(loss1.device)

### 给定 GT PE, 来重建 其对应位置的 normalized local patch. 这里还是在重建 local structure.
### 这里的local structure 重建采用的是 folding.
@MODELS.register_module()
class PointCAE_transformer_folding(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer] ', logger ='PointCAE_transformer')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.all_patch = config.all_patch
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # # prediction head
        # self.increase_dim = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        self.folding1 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.meshgrid = [[-0.3, 0.3, 6], [-0.3, 0.3, 6]]
        trunc_normal_(self.mask_token, std=.02)
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

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood = neighborhood - center.unsqueeze(2)
        transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
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
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            x_rec = x_rec.reshape(B * M, C)
            x_rec = x_rec.unsqueeze(-1).repeat(1, 1, 36)  # (batch_size*group number, feat_dims, num_points)
            points = self.build_grid(x_rec.shape[0]).transpose(1,
                                                               2)  # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
            if x_rec.get_device() != -1:
                points = points.cuda(x_rec.get_device())
            cat1 = torch.cat((x_rec, points),
                             dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
            cat2 = torch.cat((x_rec, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
            folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)
            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1,3)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss1 = self.loss_func(folding_result2, gt_points)

        else:
            x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)

            # _, N, _ = pos_emd_mask.shape
            # mask_token = self.mask_token.expand(B, N, -1)
            x_full = x_vis
            pos_full = pos_emd_vis

            x_rec = self.MAE_decoder(x_full, pos_full)

            B, M, C = x_rec.shape
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            x_rec = x_rec.reshape(B * M, C)
            x_rec = x_rec.unsqueeze(-1).repeat(1, 1, 36)  # (batch_size*group number, feat_dims, num_points)
            points = self.build_grid(x_rec.shape[0]).transpose(1,
                                                               2)  # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
            if x_rec.get_device() != -1:
                points = points.cuda(x_rec.get_device())
            cat1 = torch.cat((x_rec, points),
                             dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
            cat2 = torch.cat((x_rec, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
            folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)

            gt_points = neighborhood.reshape(B * M, -1, 3)
            loss1 = self.loss_func(folding_result2, gt_points)

        return loss1, torch.zeros(1).to(loss1.device)

### 给定 GT PE, 来重建 其对应位置的 normalized local patch. 这里还是在重建 local structure.
### 这里的local structure 重建采用的是 folding, global 重建用的是FC
@MODELS.register_module()
class PointCAE_transformer_fc_global_folding_local(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer] ', logger ='PointCAE_transformer')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.all_patch = config.all_patch
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # # prediction head
        # self.increase_dim = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * 64)
        )

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.meshgrid = [[-0.3, 0.3, 6], [-0.3, 0.3, 6]]
        trunc_normal_(self.mask_token, std=.02)
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

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, corrupted_pts, pts, vis = False, return_feat = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood = neighborhood - center.unsqueeze(2)
        transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            if return_feat:
                return global_feature
            else:
                coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
                gt_points_center = center

                ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
                # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
                # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
                ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
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
                # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
                x_rec = x_rec.reshape(B * M, C)
                x_rec = x_rec.unsqueeze(-1).repeat(1, 1, 36)  # (batch_size*group number, feat_dims, num_points)
                points = self.build_grid(x_rec.shape[0]).transpose(1,
                                                                   2)  # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
                if x_rec.get_device() != -1:
                    points = points.cuda(x_rec.get_device())
                cat1 = torch.cat((x_rec, points),
                                 dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
                folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
                cat2 = torch.cat((x_rec, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
                folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
                folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)
                if self.all_patch == 'True':
                    gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                           neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1,3)
                else:
                    gt_points = neighborhood[mask].reshape(B * M, -1, 3)
                loss1 = self.loss_func(folding_result2, gt_points)
                loss2 = self.loss_func(coarse_point_cloud, gt_points_center)

                if vis: #visualization
                    ####### reconstructed masked patch and visual patch.
                    vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                    full_vis = vis_points + center[~mask].unsqueeze(1)
                    full_rebuild = folding_result2 + center[mask].unsqueeze(1)
                    # ipdb.set_trace() ## 可能要从full_rebuild 中随机选32 points for the consistency.
                    full_rebuild = full_rebuild[:, :32, :]
                    full = torch.cat([full_vis, full_rebuild], dim=0)
                    full_center = torch.cat([center[mask], center[~mask]], dim=0)
                    ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                    full = full.reshape(-1, 3).unsqueeze(0) ## reconstruction with local patches.
                    ######  corrupted and visual patches.
                    vis_points_corrupted = transformed_neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                    full_vis_corrupted = vis_points_corrupted + transformed_center[~mask].unsqueeze(1)
                    full_vis_corrupted = full_vis_corrupted.reshape(-1, 3).unsqueeze(0)
                    # return ret1, ret2
                    return full_vis_corrupted, coarse_point_cloud, full, pts
                    # return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
                else:
                    return loss1, loss2

        else:
            x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            if return_feat:
                return global_feature
            else:
                coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
                gt_points_center = center
                ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
                # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
                # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
                ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
                pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)

                # _, N, _ = pos_emd_mask.shape
                # mask_token = self.mask_token.expand(B, N, -1)
                x_full = x_vis
                pos_full = pos_emd_vis

                x_rec = self.MAE_decoder(x_full, pos_full)

                B, M, C = x_rec.shape
                # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
                x_rec = x_rec.reshape(B * M, C)
                x_rec = x_rec.unsqueeze(-1).repeat(1, 1, 36)  # (batch_size*group number, feat_dims, num_points)
                points = self.build_grid(x_rec.shape[0]).transpose(1,
                                                                   2)  # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
                if x_rec.get_device() != -1:
                    points = points.cuda(x_rec.get_device())
                cat1 = torch.cat((x_rec, points),
                                 dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
                folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
                cat2 = torch.cat((x_rec, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
                folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
                folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)

                gt_points = neighborhood.reshape(B * M, -1, 3)
                loss1 = self.loss_func(folding_result2, gt_points)
                loss2 = self.loss_func(coarse_point_cloud, gt_points_center)

                if vis: #visualization
                    ####### reconstructed masked patch and visual patch.
                    # folding_result2 = folding_result2[:, :32]
                    full_rebuild = folding_result2 + center.unsqueeze(1)
                    full = full_rebuild.reshape(-1, 3).unsqueeze(0) ## reconstruction with local patches.
                    ######  corrupted and visual patches.
                    vis_points_corrupted = transformed_neighborhood.reshape(B * self.num_group, -1, 3)
                    full_vis_corrupted = vis_points_corrupted + transformed_center.unsqueeze(1)
                    full_vis_corrupted = full_vis_corrupted.reshape(-1, 3).unsqueeze(0)
                    # return ret1, ret2
                    return full_vis_corrupted, coarse_point_cloud, full, pts
                    # return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
                else:
                    return loss1, loss2

### 给定 GT PE, 来重建 其对应位置的 normalized local patch. 这里还是在重建 local structure.
### 这里的local structure 重建采用的是 folding, global 重建用的是folding
@MODELS.register_module()
class PointCAE_transformer_folding_global_folding_local(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer] ', logger ='PointCAE_transformer')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.all_patch = config.all_patch
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # # prediction head
        # self.increase_dim = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        # self.coarse_pred = nn.Sequential(
        #     nn.Linear(self.trans_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * 64)
        # )

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.meshgrid = [[-0.3, 0.3, 6], [-0.3, 0.3, 6]]
        self.meshgrid_global = [[-0.3, 0.3, 8], [-0.3, 0.3, 8]]
        self.folding1_global = nn.Sequential(
            nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.folding2_global = nn.Sequential(
            nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )

        trunc_normal_(self.mask_token, std=.02)
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

    def build_grid_global(self, batch_size):
        x = np.linspace(*self.meshgrid_global[0])
        y = np.linspace(*self.meshgrid_global[1])
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


    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood = neighborhood - center.unsqueeze(2)
        transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, num_vis, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            global_feature = global_feature.unsqueeze(-1).repeat(1, 1, 64)  # (batch_size, feat_dims, num_points)
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            points = self.build_grid_global(global_feature.shape[0]).transpose(1,2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if global_feature.get_device() != -1:
                points = points.cuda(global_feature.get_device())
            cat1 = torch.cat((global_feature, points),
                             dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1_global(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((global_feature, folding_result1), dim=1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2_global(cat2)  # (batch_size, 3, num_points)
            coarse_point_cloud = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)
            gt_points_center = center

            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
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
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            x_rec = x_rec.reshape(B * M, C)
            x_rec = x_rec.unsqueeze(-1).repeat(1, 1, 36)  # (batch_size*group number, feat_dims, num_points)
            points = self.build_grid(x_rec.shape[0]).transpose(1,
                                                               2)  # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
            if x_rec.get_device() != -1:
                points = points.cuda(x_rec.get_device())
            cat1 = torch.cat((x_rec, points),
                             dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
            cat2 = torch.cat((x_rec, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
            folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)
            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1,3)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss1 = self.loss_func(folding_result2, gt_points)
            loss2 = self.loss_func(coarse_point_cloud, gt_points_center)

        else:
            x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            global_feature = global_feature.unsqueeze(-1).repeat(1, 1, 64)  # (batch_size, feat_dims, num_points)
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            points = self.build_grid_global(global_feature.shape[0]).transpose(1,2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if global_feature.get_device() != -1:
                points = points.cuda(global_feature.get_device())
            cat1 = torch.cat((global_feature, points),
                             dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((global_feature, folding_result1), dim=1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
            coarse_point_cloud = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)
            gt_points_center = center
            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)

            # _, N, _ = pos_emd_mask.shape
            # mask_token = self.mask_token.expand(B, N, -1)
            x_full = x_vis
            pos_full = pos_emd_vis

            x_rec = self.MAE_decoder(x_full, pos_full)

            B, M, C = x_rec.shape
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            x_rec = x_rec.reshape(B * M, C)
            x_rec = x_rec.unsqueeze(-1).repeat(1, 1, 36)  # (batch_size*group number, feat_dims, num_points)
            points = self.build_grid(x_rec.shape[0]).transpose(1,
                                                               2)  # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
            if x_rec.get_device() != -1:
                points = points.cuda(x_rec.get_device())
            cat1 = torch.cat((x_rec, points),
                             dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
            cat2 = torch.cat((x_rec, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
            folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)

            gt_points = neighborhood.reshape(B * M, -1, 3)
            loss1 = self.loss_func(folding_result2, gt_points)
            loss2 = self.loss_func(coarse_point_cloud, gt_points_center)

        return loss1, loss2

### 给定 GT PE, 来重建 其对应位置的 normalized local patch. 这里还是在重建 local structure.
### 这里的local structure 重建采用的是 folding, global 重建用的是folding
@MODELS.register_module()
class PointCAE_transformer_folding_global_fc_local(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer] ', logger ='PointCAE_transformer')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.all_patch = config.all_patch
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )
        # self.coarse_pred = nn.Sequential(
        #     nn.Linear(self.trans_dim, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * 64)
        # )

        # self.folding1 = nn.Sequential(
        #     nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(self.trans_dim, self.trans_dim, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(self.trans_dim, 3, 1),
        # )
        # self.folding2 = nn.Sequential(
        #     nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(self.trans_dim, self.trans_dim, 1),
        #     nn.ReLU(),
        #     nn.Conv1d(self.trans_dim, 3, 1),
        # )
        self.meshgrid = [[-0.3, 0.3, 6], [-0.3, 0.3, 6]]
        self.meshgrid_global = [[-0.3, 0.3, 8], [-0.3, 0.3, 8]]
        self.folding1_global = nn.Sequential(
            nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.folding2_global = nn.Sequential(
            nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )

        trunc_normal_(self.mask_token, std=.02)
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

    def build_grid_global(self, batch_size):
        x = np.linspace(*self.meshgrid_global[0])
        y = np.linspace(*self.meshgrid_global[1])
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


    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood = neighborhood - center.unsqueeze(2)
        transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, num_vis, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            global_feature = global_feature.unsqueeze(-1).repeat(1, 1, 64)  # (batch_size, feat_dims, num_points)
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            points = self.build_grid_global(global_feature.shape[0]).transpose(1,2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if global_feature.get_device() != -1:
                points = points.cuda(global_feature.get_device())
            cat1 = torch.cat((global_feature, points),
                             dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1_global(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((global_feature, folding_result1), dim=1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2_global(cat2)  # (batch_size, 3, num_points)
            coarse_point_cloud = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)
            gt_points_center = center

            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
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
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            # x_rec = x_rec.reshape(B * M, C)
            # x_rec = x_rec.unsqueeze(-1).repeat(1, 1, 36)  # (batch_size*group number, feat_dims, num_points)
            # points = self.build_grid(x_rec.shape[0]).transpose(1,
            #                                                    2)  # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
            # if x_rec.get_device() != -1:
            #     points = points.cuda(x_rec.get_device())
            # cat1 = torch.cat((x_rec, points),
            #                  dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
            # folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
            # cat2 = torch.cat((x_rec, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
            # folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
            # folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)
            if self.all_patch == 'True':
                gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
                                       neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1,3)
            else:
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)
            loss2 = self.loss_func(coarse_point_cloud, gt_points_center)

        else:
            x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            global_feature = global_feature.unsqueeze(-1).repeat(1, 1, 64)  # (batch_size, feat_dims, num_points)
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            points = self.build_grid_global(global_feature.shape[0]).transpose(1,2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if global_feature.get_device() != -1:
                points = points.cuda(global_feature.get_device())
            cat1 = torch.cat((global_feature, points),
                             dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1_global(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((global_feature, folding_result1), dim=1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2_global(cat2)  # (batch_size, 3, num_points)
            coarse_point_cloud = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)
            gt_points_center = center
            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)

            # _, N, _ = pos_emd_mask.shape
            # mask_token = self.mask_token.expand(B, N, -1)
            x_full = x_vis
            pos_full = pos_emd_vis

            x_rec = self.MAE_decoder(x_full, pos_full)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            # x_rec = x_rec.reshape(B * M, C)
            # x_rec = x_rec.unsqueeze(-1).repeat(1, 1, 36)  # (batch_size*group number, feat_dims, num_points)
            # points = self.build_grid(x_rec.shape[0]).transpose(1,
            #                                                    2)  # (batch_size*group number, 2, 36) or (batch_size*group number, 3, 36)
            # if x_rec.get_device() != -1:
            #     points = points.cuda(x_rec.get_device())
            # cat1 = torch.cat((x_rec, points),
            #                  dim=1)  # (batch_size*group number, feat_dims+2, num_points) or (batch_size*group number, feat_dims+3, num_points)
            # folding_result1 = self.folding1(cat1)  # (batch_size*group number, 3, num_points)
            # cat2 = torch.cat((x_rec, folding_result1), dim=1)  # (batch_size*group number, 515, num_points)
            # folding_result2 = self.folding2(cat2)  # (batch_size*group number, 3, num_points)
            # folding_result2 = folding_result2.transpose(1, 2)  # (batch_size*group number, num_points ,3)

            gt_points = neighborhood.reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)
            loss2 = self.loss_func(coarse_point_cloud, gt_points_center)

        return loss1, loss2



### 给定 GT PE, 来重建 其对应位置的 normalized local patch. 这里还是在重建 local structure.
## old name: PointCAE_transformer_with_fc_center_p
@MODELS.register_module()
class PointCAE_transformer_fc_global_fc_local(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer] ', logger ='PointCAE_transformer')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * self.num_group)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood = neighborhood - center.unsqueeze(2)
        transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            gt_points_center = center

            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)

            _, N, _ = pos_emd_mask.shape
            mask_token = self.mask_token.expand(B, N, -1)
            x_full = torch.cat([x_vis, mask_token], dim=1)
            pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

            x_rec = self.MAE_decoder(x_full, pos_full, N)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)
            loss2 = self.loss_func(coarse_point_cloud, gt_points_center)
            if vis: #visualization
                ####### reconstructed masked patch and visual patch.
                vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis = vis_points + center[~mask].unsqueeze(1)
                full_rebuild = rebuild_points + center[mask].unsqueeze(1)
                full = torch.cat([full_vis, full_rebuild], dim=0)
                full_center = torch.cat([center[mask], center[~mask]], dim=0)
                ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                full = full.reshape(-1, 3).unsqueeze(0) ## reconstruction with local patches.
                ######  corrupted and visual patches.
                vis_points_corrupted = transformed_neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                full_vis_corrupted = vis_points_corrupted + transformed_center[~mask].unsqueeze(1)
                full_vis_corrupted = full_vis_corrupted.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return full_vis_corrupted, coarse_point_cloud, full, pts
                # return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
            else:
                return loss1, loss2

        else:
            x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            gt_points_center = center

            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)

            # _, N, _ = pos_emd_mask.shape
            # mask_token = self.mask_token.expand(B, N, -1)
            x_full = x_vis
            pos_full = pos_emd_vis

            x_rec = self.MAE_decoder(x_full, pos_full)

            B, M, C = x_rec.shape
            rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024

            gt_points = neighborhood.reshape(B * M, -1, 3)
            loss1 = self.loss_func(rebuild_points, gt_points)

            loss2 = self.loss_func(coarse_point_cloud, gt_points_center)
            if vis: #visualization
                ####### reconstructed masked patch and visual patch.
                # folding_result2 = folding_result2[:, :32]
                # ipdb.set_trace()
                full_rebuild = rebuild_points + center[0].unsqueeze(1)
                full = full_rebuild.reshape(-1, 3).unsqueeze(0) ## reconstruction with local patches.
                ######  corrupted and visual patches.
                vis_points_corrupted = transformed_neighborhood.reshape(B * self.num_group, -1, 3)
                full_vis_corrupted = vis_points_corrupted + transformed_center[0].unsqueeze(1)
                full_vis_corrupted = full_vis_corrupted.reshape(-1, 3).unsqueeze(0)
                # return ret1, ret2
                return full_vis_corrupted, coarse_point_cloud, full, pts
                # return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
            else:
                return loss1, loss2

### 给定 GT PE, 来重建 其对应位置的 normalized local patch. 这里还是在重建 local structure.
@MODELS.register_module()
class PointCAE_transformer_fc_global_fc_local_3detr(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_fc_global_fc_local_3detr] ', logger ='PointCAE_transformer_fc_global_fc_local_3detr')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            raise NotImplementedError
            # self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        self.group_divider = (DummyGroup if self.enc_arch == '3detr' else Group)(num_group = self.num_group, group_size = self.group_size)
        # prediction head
        self.grouper = pointnet2_utils.QueryAndGroup(0.2, self.group_size,
                                                     use_xyz=True, ret_grouped_xyz=True, normalize_xyz=True,
                                                     sample_uniformly=False, ret_unique_cnt=False)
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3 * self.num_group)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, corrupted_pts, pts, vis = False, return_feat = False, **kwargs):
        pts = pts[:, :, :3].contiguous()  ## [-0.91, 0.96]
        # ipdb.set_trace()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3
        # if self.enc_arch == '3detr':
        #     #### for 3detr, neighborhood means nothing and center is the vanilla pts, since the group and forward is combined within the encoder.
        #     # we apply the affine transformation here, and re-get the UN-Affined point center and their neighbor with the center index.
        #     transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        # else:
        # neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        # neighborhood = neighborhood - center.unsqueeze(2)
        # transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            if self.enc_arch == '3detr':
                x_vis, mask, pre_enc_inds = self.MAE_encoder(transformed_neighborhood, transformed_center)
                ## B * vis_group_size* D;  B*all_group_size; B*all_group_size
                ########## get the Un-transformed neighborhood and center with the pre_enc_inds here, they will be used as the reconstruction target.
                all_points = center
                xyz_flipped = all_points.transpose(1, 2).contiguous()
                new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pre_enc_inds.detach()).transpose(1, 2).contiguous()  ## B*all_group_size*3
                grouped_features, grouped_xyz = self.grouper(all_points, new_xyz, None)  # (B, C, group_num, group_size)
                ### the new_xyz and grouped_features here well match the center and neighbor in the encoder.
                ### the grouped_xyz are centerlized, and then each group is rescale to [-1,1].
                # ipdb.set_trace()
                center_real = new_xyz
                neighborhood = grouped_xyz.transpose(1,2).transpose(2,3)  ### unnormalized patches [-1,1]
            else:
                raise NotImplementedError
                # x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
                # center_real = center.clone()
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            if return_feat:
                return global_feature
            else:
                coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
                gt_points_center = center_real

                ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
                # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
                # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
                ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
                pos_emd_vis = self.decoder_pos_embed(center_real[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embed(center_real[mask]).reshape(B, -1, C)

                _, N, _ = pos_emd_mask.shape
                mask_token = self.mask_token.expand(B, N, -1)
                x_full = torch.cat([x_vis, mask_token], dim=1)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                x_rec = self.MAE_decoder(x_full, pos_full, N)

                B, M, C = x_rec.shape
                rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)

                loss_fine = self.loss_func(rebuild_points, gt_points)
                loss_coarse = self.loss_func(coarse_point_cloud, gt_points_center)
                if vis: #visualization
                    ####### reconstructed masked patch and visual patch.
                    vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                    full_vis = vis_points + center_real[~mask].unsqueeze(1)
                    full_rebuild = rebuild_points + center_real[mask].unsqueeze(1)
                    full = torch.cat([full_vis, full_rebuild], dim=0)
                    full_center = torch.cat([center_real[mask], center_real[~mask]], dim=0)
                    ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                    full = full.reshape(-1, 3).unsqueeze(0) ## reconstruction with local patches.
                    ######  corrupted and visual patches.
                    vis_points_corrupted = transformed_neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                    full_vis_corrupted = vis_points_corrupted + transformed_center[~mask].unsqueeze(1)
                    full_vis_corrupted = full_vis_corrupted.reshape(-1, 3).unsqueeze(0)
                    # return ret1, ret2
                    return full_vis_corrupted, coarse_point_cloud, full, pts
                    # return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
                else:
                    # return torch.mean(x_vis), torch.mean(x_vis)
                    return loss_fine, loss_coarse

        else:
            raise NotImplementedError
            # x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            # B, _, C = x_vis.shape  # B VIS C
            #
            # global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            #
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            # gt_points_center = center
            #
            # ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            # ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            # pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)
            #
            # # _, N, _ = pos_emd_mask.shape
            # # mask_token = self.mask_token.expand(B, N, -1)
            # x_full = x_vis
            # pos_full = pos_emd_vis
            #
            # x_rec = self.MAE_decoder(x_full, pos_full)
            #
            # B, M, C = x_rec.shape
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            #
            # gt_points = neighborhood.reshape(B * M, -1, 3)
            # loss1 = self.loss_func(rebuild_points, gt_points)
            #
            # loss2 = self.loss_func(coarse_point_cloud, gt_points_center)
            # if vis: #visualization
            #     ####### reconstructed masked patch and visual patch.
            #     # folding_result2 = folding_result2[:, :32]
            #     # ipdb.set_trace()
            #     full_rebuild = rebuild_points + center[0].unsqueeze(1)
            #     full = full_rebuild.reshape(-1, 3).unsqueeze(0) ## reconstruction with local patches.
            #     ######  corrupted and visual patches.
            #     vis_points_corrupted = transformed_neighborhood.reshape(B * self.num_group, -1, 3)
            #     full_vis_corrupted = vis_points_corrupted + transformed_center[0].unsqueeze(1)
            #     full_vis_corrupted = full_vis_corrupted.reshape(-1, 3).unsqueeze(0)
            #     # return ret1, ret2
            #     return full_vis_corrupted, coarse_point_cloud, full, pts
            #     # return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
            # else:
            #     return loss1, loss2

### 给定 GT PE, 来重建 其对应位置的 normalized local patch. 这里还是在重建 local structure.
@MODELS.register_module()
class PointCAE_transformer_fc_global_fc_local_3detr_encodernope(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_fc_global_fc_local_3detr_encodernope] ', logger ='PointCAE_transformer_fc_global_fc_local_3detr_encodernope')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            raise NotImplementedError
            # self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        # self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        self.enc_arch = config.transformer_config.get('enc_arch', 'PointViT')
        self.group_divider = (DummyGroup if self.enc_arch == '3detr' else Group)(num_group = self.num_group, group_size = self.group_size)
        # prediction head
        self.grouper = pointnet2_utils.QueryAndGroup(0.2, self.group_size,
                                                     use_xyz=True, ret_grouped_xyz=True, normalize_xyz=True,
                                                     sample_uniformly=False, ret_unique_cnt=False)
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 3 * self.num_group)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, corrupted_pts, pts, vis = False, return_feat = False, **kwargs):
        pts = pts[:, :, :3].contiguous()  ## [-0.91, 0.96]
        # ipdb.set_trace()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3
        # if self.enc_arch == '3detr':
        #     #### for 3detr, neighborhood means nothing and center is the vanilla pts, since the group and forward is combined within the encoder.
        #     # we apply the affine transformation here, and re-get the UN-Affined point center and their neighbor with the center index.
        #     transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        # else:
        # neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        # neighborhood = neighborhood - center.unsqueeze(2)
        # transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            if self.enc_arch == '3detr':
                x_vis, mask, pre_enc_inds = self.MAE_encoder(transformed_neighborhood, transformed_center)
                ## B * vis_group_size* D;  B*all_group_size; B*all_group_size
                ########## get the Un-transformed neighborhood and center with the pre_enc_inds here, they will be used as the reconstruction target.
                all_points = center
                xyz_flipped = all_points.transpose(1, 2).contiguous()
                new_xyz = pointnet2_utils.gather_operation(xyz_flipped, pre_enc_inds.detach()).transpose(1, 2).contiguous()  ## B*all_group_size*3
                grouped_features, grouped_xyz = self.grouper(all_points, new_xyz, None)  # (B, C, group_num, group_size)
                ### the new_xyz and grouped_features here well match the center and neighbor in the encoder.
                ### the grouped_xyz are centerlized, and then each group is rescale to [-1,1].
                # ipdb.set_trace()
                center_real = new_xyz
                neighborhood = grouped_xyz.transpose(1,2).transpose(2,3)  ### unnormalized patches [-1,1]
            else:
                raise NotImplementedError
                # x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
                # center_real = center.clone()
            B, _, C = x_vis.shape  # B VIS C

            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            if return_feat:
                return global_feature
            else:
                coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
                gt_points_center = center_real

                ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
                # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
                # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
                ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
                pos_emd_vis = self.decoder_pos_embed(center_real[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embed(center_real[mask]).reshape(B, -1, C)

                _, N, _ = pos_emd_mask.shape
                mask_token = self.mask_token.expand(B, N, -1)
                x_full = torch.cat([x_vis, mask_token], dim=1)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                x_rec = self.MAE_decoder(x_full, pos_full, N)

                B, M, C = x_rec.shape
                rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
                gt_points = neighborhood[mask].reshape(B * M, -1, 3)

                loss_fine = self.loss_func(rebuild_points, gt_points)
                loss_coarse = self.loss_func(coarse_point_cloud, gt_points_center)
                if vis: #visualization
                    ####### reconstructed masked patch and visual patch.
                    vis_points = neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                    full_vis = vis_points + center_real[~mask].unsqueeze(1)
                    full_rebuild = rebuild_points + center_real[mask].unsqueeze(1)
                    full = torch.cat([full_vis, full_rebuild], dim=0)
                    full_center = torch.cat([center_real[mask], center_real[~mask]], dim=0)
                    ret2 = full_vis.reshape(-1, 3).unsqueeze(0)
                    full = full.reshape(-1, 3).unsqueeze(0) ## reconstruction with local patches.
                    ######  corrupted and visual patches.
                    vis_points_corrupted = transformed_neighborhood[~mask].reshape(B * (self.num_group - M), -1, 3)
                    full_vis_corrupted = vis_points_corrupted + transformed_center[~mask].unsqueeze(1)
                    full_vis_corrupted = full_vis_corrupted.reshape(-1, 3).unsqueeze(0)
                    # return ret1, ret2
                    return full_vis_corrupted, coarse_point_cloud, full, pts
                    # return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
                else:
                    # return torch.mean(x_vis), torch.mean(x_vis)
                    return loss_fine, loss_coarse

        else:
            raise NotImplementedError
            # x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            # B, _, C = x_vis.shape  # B VIS C
            #
            # global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1) # B C
            #
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            # gt_points_center = center
            #
            # ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            # ### pos 是 transformed 之前的 GT, 也就是 reconstruction 的中心。
            # pos_emd_vis = self.decoder_pos_embed(center).reshape(B, -1, C)
            #
            # # _, N, _ = pos_emd_mask.shape
            # # mask_token = self.mask_token.expand(B, N, -1)
            # x_full = x_vis
            # pos_full = pos_emd_vis
            #
            # x_rec = self.MAE_decoder(x_full, pos_full)
            #
            # B, M, C = x_rec.shape
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            #
            # gt_points = neighborhood.reshape(B * M, -1, 3)
            # loss1 = self.loss_func(rebuild_points, gt_points)
            #
            # loss2 = self.loss_func(coarse_point_cloud, gt_points_center)
            # if vis: #visualization
            #     ####### reconstructed masked patch and visual patch.
            #     # folding_result2 = folding_result2[:, :32]
            #     # ipdb.set_trace()
            #     full_rebuild = rebuild_points + center[0].unsqueeze(1)
            #     full = full_rebuild.reshape(-1, 3).unsqueeze(0) ## reconstruction with local patches.
            #     ######  corrupted and visual patches.
            #     vis_points_corrupted = transformed_neighborhood.reshape(B * self.num_group, -1, 3)
            #     full_vis_corrupted = vis_points_corrupted + transformed_center[0].unsqueeze(1)
            #     full_vis_corrupted = full_vis_corrupted.reshape(-1, 3).unsqueeze(0)
            #     # return ret1, ret2
            #     return full_vis_corrupted, coarse_point_cloud, full, pts
            #     # return corrupted_pts.transpose(1, 2).contiguous(), coarse, fine, pts
            # else:
            #     return loss1, loss2




### 给定 transformed PE, 来重建 其对应位置的 patch-wise center. 先观察patch center 的重建，因为这可能是最重要的。
### 每个patch 只有一个center. 这里重复apply Affined PE to encoder.
@MODELS.register_module()
class PointCAE_transformer_patch_center_only(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer] ', logger ='PointCAE_transformer')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.all_patch = config.all_patch
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        # self.decoder_pos_embed = nn.Sequential(
        #     nn.Linear(3, 128),
        #     nn.GELU(),
        #     nn.Linear(128, self.trans_dim)
        # )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        # self.MAE_decoder = TransformerDecoder(
        #     embed_dim=self.trans_dim,
        #     depth=self.decoder_depth,
        #     drop_path_rate=dpr,
        #     num_heads=self.decoder_num_heads,
        # )

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        # prediction head for patch detail.
        # self.increase_dim = nn.Sequential(
        #     # nn.Conv1d(self.trans_dim, 1024, 1),
        #     # nn.BatchNorm1d(1024),
        #     # nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        # )
        # prediction head for patch center
        self.increase_dim2 = nn.Sequential(
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3, 1)
        )

        # trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood = neighborhood - center.unsqueeze(2)
        transformed_neighborhood = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            raise NotImplementedError
            # x_vis, mask = self.MAE_encoder(transformed_neighborhood, transformed_center)
            # B, _, C = x_vis.shape  # B VIS C
            #
            # ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            # ### pos 是 transformed 之后的 GT,
            # pos_emd_vis = self.decoder_pos_embed(transformed_center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(transformed_center[mask]).reshape(B, -1, C)
            #
            # _, N, _ = pos_emd_mask.shape
            # mask_token = self.mask_token.expand(B, N, -1)
            # x_full = torch.cat([x_vis, mask_token], dim=1)
            # pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
            #
            # if self.all_patch == 'True':
            #     x_rec = self.MAE_decoder(x_full, pos_full)
            # else:
            #     x_rec = self.MAE_decoder(x_full, pos_full, N)
            #
            # B, M, C = x_rec.shape
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M group_size 3
            # rebuild_centers = self.increase_dim2(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, 3)  # B*M 1
            #
            # if self.all_patch == 'True':
            #     gt_points = torch.cat((neighborhood[~mask].reshape(B, -1, self.group_size, 3),
            #                            neighborhood[mask].reshape(B, -1, self.group_size, 3)), dim=1).reshape(B * M, -1,3)
            #     gt_centers = torch.cat((center[~mask].reshape(B, -1, 3),
            #                            center[mask].reshape(B, -1, 3)), dim=1).reshape(B * M, 3)
            # else:
            #     gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            #     gt_centers = center[mask].reshape(B * M, 3)
            # # gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            # loss1 = self.loss_func(rebuild_points, gt_points)
            # loss2 = (rebuild_centers-gt_centers).pow(2).sum(1).mean()
        else:
            x_vis = self.MAE_encoder(transformed_neighborhood, transformed_center)
            B, M, C = x_vis.shape  # B VIS C
            ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            ### pos 是 transformed 之后的 。
            # pos_emd_vis = self.decoder_pos_embed(transformed_center).reshape(B, -1, C)
            #
            # # _, N, _ = pos_emd_mask.shape
            # # mask_token = self.mask_token.expand(B, N, -1)
            # x_full = x_vis
            # pos_full = pos_emd_vis
            #
            # x_rec = self.MAE_decoder(x_full, pos_full)
            #
            # B, M, C = x_rec.shape
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            rebuild_centers = self.increase_dim2(x_vis.transpose(1, 2)).transpose(1, 2).reshape(B * M, 3)  # B*M 1

            # gt_points = neighborhood.reshape(B * M, -1, 3)
            gt_centers = center.reshape(B * M, 3)
            # loss1 = self.loss_func(rebuild_points, gt_points)
            loss2 = (rebuild_centers-gt_centers).pow(2).sum(1).mean()

        return loss2, torch.zeros(1).to(loss2.device)


### encoder做法不变，完全推翻decoder 做法; 采用 FC reconstruction 的decoder 作为默认decoder. 只重建global center, 看一下重建global 的效果。
@MODELS.register_module()
class PointCAE_transformer_fc_center(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_fc_center] ', logger ='PointCAE_transformer_fc_center')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoderOnePEMultiX(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)


        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(self.trans_dim, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(1024, 1024, 1)
        # )

        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * 64)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood_normalized + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood_normalized = neighborhood - center.unsqueeze(2)
        transformed_neighborhood_normalized = transformed_neighborhood - transformed_center.unsqueeze(2)
        if 'Drop-Patch' in self.corrupt_type:
            x_vis, _ = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C
            # global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B C N
            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0]  # B C

            coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            gt_points = center
            loss1 = self.loss_func(coarse_point_cloud, gt_points)
        else:
            x_vis = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C
            # global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B C N
            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0]  # B C

            coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            gt_points = center
            loss1 = self.loss_func(coarse_point_cloud, gt_points)

        return loss1, torch.zeros(1).to(loss1.device)

### encoder做法不变，完全推翻decoder 做法; 采用 FC reconstruction 的decoder 作为默认decoder. 只重建global center, 看一下重建global 的效果。
@MODELS.register_module()
class PointCAE_transformer_fold_center(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_fold_center] ', logger ='PointCAE_transformer_fold_center')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.meshgrid = [[-0.3, 0.3, 8], [-0.3, 0.3, 8]]
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoderOnePEMultiX(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(self.trans_dim, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(1024, 1024, 1)
        # )
        # self.coarse_pred = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * 64)
        # )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood_normalized + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood_normalized = neighborhood - center.unsqueeze(2)
        transformed_neighborhood_normalized = transformed_neighborhood - transformed_center.unsqueeze(2)
        if 'Drop-Patch' in self.corrupt_type:
            x_vis, _ = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C
            global_feature = x_vis.transpose(1, 2)  # B C N
            x_vis = torch.max(global_feature, dim=-1)[0]  # B C
            x_vis = x_vis.unsqueeze(-1).repeat(1, 1, 64)  # (batch_size, feat_dims, num_points)
            points = self.build_grid(x_vis.shape[0]).transpose(1,2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if x_vis.get_device() != -1:
                points = points.cuda(x_vis.get_device())
            cat1 = torch.cat((x_vis, points),dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((x_vis, folding_result1), dim=1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
            folding_result2 = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)

            # global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B 1024 N
            # global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024
            #
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            gt_points = center
            loss1 = self.loss_func(folding_result2, gt_points)
        else:
            x_vis = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C
            global_feature = x_vis.transpose(1, 2)  # B C N
            x_vis = torch.max(global_feature, dim=-1)[0]  # B C
            x_vis = x_vis.unsqueeze(-1).repeat(1, 1, 64)  # (batch_size, feat_dims, num_points)
            points = self.build_grid(x_vis.shape[0]).transpose(1,2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if x_vis.get_device() != -1:
                points = points.cuda(x_vis.get_device())
            cat1 = torch.cat((x_vis, points),dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((x_vis, folding_result1), dim=1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
            folding_result2 = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)

            # global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B 1024 N
            # global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024
            #
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            gt_points = center
            loss1 = self.loss_func(folding_result2, gt_points)

        return loss1, torch.zeros(1).to(loss1.device)

### encoder做法不变，完全推翻decoder 做法; 采用 FC reconstruction 的decoder 作为默认decoder.
# 先重建global center, 然后把global center 作为transformer decoder 的输入，来重建detail.
@MODELS.register_module()
class PointCAE_transformer_v6_corase2fine_transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_v5] ', logger ='PointCAE_transformer_v5')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
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

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)


        self.increase_dim = nn.Sequential(
            nn.Conv1d(self.trans_dim, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, 1024, 1)
        )
        self.coarse_pred = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * 64)
        )
        # prediction head
        self.increase_dim_fine = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2_corase2fine().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood_normalized + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood_normalized = neighborhood - center.unsqueeze(2)
        transformed_neighborhood_normalized = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            raise NotImplementedError
            # x_vis, mask = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            # B, _, C = x_vis.shape  # B VIS C
            #
            # ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            # ### pos 是 transformed 之后的 GT,
            # pos_emd_vis = self.decoder_pos_embed(transformed_center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(transformed_center[mask]).reshape(B, -1, C)
            #
            # _, N, _ = pos_emd_mask.shape
            # mask_token = self.mask_token.expand(B, N, -1)
            # x_full = torch.cat([x_vis, mask_token], dim=1)
            # pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
            #
            # x_rec = self.MAE_decoder(x_full, pos_full, N)
            #
            # B, M, C = x_rec.shape
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            #
            # gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            # loss1 = self.loss_func(rebuild_points, gt_points)

        else:
            x_vis = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C
            global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B 1024 N
            global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024

            coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3); B * 64 * 3
            ## coarse_point_cloud is the predicted patch centers.
            pos_emd_vis = self.decoder_pos_embed(coarse_point_cloud).reshape(B, -1, C)
            x_full = x_vis
            pos_full = pos_emd_vis
            x_rec = self.MAE_decoder(x_full, pos_full)
            B, M, C = x_rec.shape
            ## increase_dim_fine is not defined.
            rebuild_points = self.increase_dim_fine(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, M, -1, 3)  # B M 1024
            gt_points_fine = neighborhood_normalized.reshape(B, M, -1, 3)  ### normalized.

            gt_points_coarse = center
            loss1, loss2 = self.loss_func(coarse_point_cloud, gt_points_coarse, rebuild_points, gt_points_fine)

        return loss1, loss2



### encoder做法不变，完全推翻decoder 做法; 采用 FC reconstruction 的decoder 作为默认decoder.
# 先重建global center, 然后把global center 作为transformer decoder 的输入，来重建detail.
@MODELS.register_module()
class PointCAE_transformer_v6_folding_corase2fine_transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_v5] ', logger ='PointCAE_transformer_v5')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.meshgrid = [[-0.3, 0.3, 8], [-0.3, 0.3, 8]]
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoder(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(self.trans_dim, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(1024, 1024, 1)
        # )
        # self.coarse_pred = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * 64)
        # )
        # prediction head
        self.increase_dim_fine = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 3*self.group_size, 1)
        )
        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2_corase2fine().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood_normalized + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood_normalized = neighborhood - center.unsqueeze(2)
        transformed_neighborhood_normalized = transformed_neighborhood - transformed_center.unsqueeze(2)

        ## input for encoder.
        if 'Drop-Patch' in self.corrupt_type:
            raise NotImplementedError
            # x_vis, mask = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            # B, _, C = x_vis.shape  # B VIS C
            #
            # ## positional embedding for decoder. 我需要把真实的center 的位置给decoder 嘛？ 还是说，给一个transform 之前的也行？ 要给一个GT 中的位置，否则这个问题太难了。
            # # pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
            # # pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
            # ### pos 是 transformed 之后的 GT,
            # pos_emd_vis = self.decoder_pos_embed(transformed_center[~mask]).reshape(B, -1, C)
            # pos_emd_mask = self.decoder_pos_embed(transformed_center[mask]).reshape(B, -1, C)
            #
            # _, N, _ = pos_emd_mask.shape
            # mask_token = self.mask_token.expand(B, N, -1)
            # x_full = torch.cat([x_vis, mask_token], dim=1)
            # pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
            #
            # x_rec = self.MAE_decoder(x_full, pos_full, N)
            #
            # B, M, C = x_rec.shape
            # rebuild_points = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1, 3)  # B M 1024
            #
            # gt_points = neighborhood[mask].reshape(B * M, -1, 3)
            # loss1 = self.loss_func(rebuild_points, gt_points)

        else:
            x_vis = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C
            global_feature = x_vis.transpose(1, 2)  # B C N
            global_feature = torch.max(global_feature, dim=-1)[0]  # B C
            global_feature = global_feature.unsqueeze(-1).repeat(1, 1, 64)  # (batch_size, feat_dims, num_points)
            points = self.build_grid(global_feature.shape[0]).transpose(1,
                                                               2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if global_feature.get_device() != -1:
                points = points.cuda(global_feature.get_device())
            cat1 = torch.cat((global_feature, points),
                             dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((global_feature, folding_result1), dim=1)  # (batch_size, 515, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
            coarse_point_cloud = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)

            # global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B 1024 N
            # global_feature = torch.max(global_feature, dim=-1)[0]  # B 1024
            #
            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3); B * 64 * 3
            ## coarse_point_cloud is the predicted patch centers.
            pos_emd_vis = self.decoder_pos_embed(coarse_point_cloud).reshape(B, -1, C)
            x_full = x_vis
            pos_full = pos_emd_vis
            x_rec = self.MAE_decoder(x_full, pos_full)
            B, M, C = x_rec.shape
            ## increase_dim_fine is not defined.
            rebuild_points = self.increase_dim_fine(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B, M, -1, 3)  # B M 1024
            gt_points_fine = neighborhood_normalized.reshape(B, M, -1, 3)  ### normalized.

            gt_points_coarse = center
            loss1, loss2 = self.loss_func(coarse_point_cloud, gt_points_coarse, rebuild_points, gt_points_fine)

        return loss1, loss2

### encoder做法不变，完全推翻decoder 做法; 采用 FC reconstruction 的decoder 作为默认decoder. 重建 1024 point, 看一下重建global + detail 的效果。
@MODELS.register_module()
class PointCAE_transformer_fc_all(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_fc_all] ', logger ='PointCAE_transformer_fc_all')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )

        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoderOnePEMultiX(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)


        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(self.trans_dim, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(1024, 1024, 1)
        # )
        self.coarse_pred = nn.Sequential(
            nn.Linear(self.trans_dim, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * 1024)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()


    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood_normalized + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood_normalized = neighborhood - center.unsqueeze(2)
        transformed_neighborhood_normalized = transformed_neighborhood - transformed_center.unsqueeze(2)

        if 'Drop-Patch' in self.corrupt_type:
            x_vis, _ = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C

            # global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B C N
            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0]  # B C

            coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            gt_points = pts

            loss1 = self.loss_func(coarse_point_cloud, gt_points)
        else:
            x_vis = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C

            # global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B C N
            global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0]  # B C

            coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            gt_points = pts

            loss1 = self.loss_func(coarse_point_cloud, gt_points)

        return loss1, torch.zeros(1).to(loss1.device)



### encoder做法不变，完全推翻decoder 做法; 采用 FC reconstruction 的decoder 作为默认decoder. 重建 1024 point, 看一下重建global + detail 的效果。
@MODELS.register_module()
class PointCAE_transformer_fold_all(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_fold_all] ', logger ='PointCAE_transformer_fold_all')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )
        self.meshgrid = [[-0.3, 0.3, 32], [-0.3, 0.3, 32]]
        self.decoder_depth = config.transformer_config.decoder_depth
        self.decoder_num_heads = config.transformer_config.decoder_num_heads
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        self.MAE_decoder = TransformerDecoderOnePEMultiX(
            embed_dim=self.trans_dim,
            depth=self.decoder_depth,
            drop_path_rate=dpr,
            num_heads=self.decoder_num_heads,
        )

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        self.folding1 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 2, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        self.folding2 = nn.Sequential(
            nn.Conv1d(self.trans_dim + 3, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, self.trans_dim, 1),
            nn.ReLU(),
            nn.Conv1d(self.trans_dim, 3, 1),
        )
        # self.increase_dim = nn.Sequential(
        #     nn.Conv1d(self.trans_dim, 1024, 1),
        #     nn.BatchNorm1d(1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Conv1d(1024, 1024, 1)
        # )
        # self.coarse_pred = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * 64)
        # )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        if loss_type == "cdl1":
            self.loss_func = ChamferDistanceL1().cuda()
        elif loss_type =='cdl2':
            self.loss_func = ChamferDistanceL2().cuda()
        else:
            raise NotImplementedError
            # self.loss_func = emd().cuda()

    def build_grid(self, batch_size):
        x = np.linspace(*self.meshgrid[0])
        y = np.linspace(*self.meshgrid[1])
        points = np.array(list(itertools.product(x, y)))
        points = np.repeat(points[np.newaxis, ...], repeats=batch_size, axis=0)
        points = torch.tensor(points)
        return points.float()

    def forward(self, corrupted_pts, pts, vis = False, **kwargs):
        pts = pts[:, :, :3].contiguous()
        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood_normalized + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood_normalized = neighborhood - center.unsqueeze(2)
        transformed_neighborhood_normalized = transformed_neighborhood - transformed_center.unsqueeze(2)

        if 'Drop-Patch' in self.corrupt_type:
            x_vis, _ = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C
            global_feature = x_vis.transpose(1, 2)  # B C N
            x_vis = torch.max(global_feature, dim=-1)[0]  # B C
            x_vis = x_vis.unsqueeze(-1).repeat(1, 1, 1024)  # (batch_size, feat_dims, num_points)
            points = self.build_grid(x_vis.shape[0]).transpose(1,2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if x_vis.get_device() != -1:
                points = points.cuda(x_vis.get_device())
            cat1 = torch.cat((x_vis, points),dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((x_vis, folding_result1), dim=1)  # (batch_size, feat_dims+3, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
            folding_result2 = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)

            gt_points = pts
            loss1 = self.loss_func(folding_result2, gt_points)
        else:
            x_vis = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C
            global_feature = x_vis.transpose(1, 2)  # B C N
            x_vis = torch.max(global_feature, dim=-1)[0]  # B C
            x_vis = x_vis.unsqueeze(-1).repeat(1, 1, 1024)  # (batch_size, feat_dims, num_points)
            points = self.build_grid(x_vis.shape[0]).transpose(1,
                                                               2)  # (batch_size, 2, num_points) or (batch_size, 3, num_points)
            if x_vis.get_device() != -1:
                points = points.cuda(x_vis.get_device())
            cat1 = torch.cat((x_vis, points),
                             dim=1)  # (batch_size, feat_dims+2, num_points) or (batch_size, feat_dims+3, num_points)
            folding_result1 = self.folding1(cat1)  # (batch_size, 3, num_points)
            cat2 = torch.cat((x_vis, folding_result1), dim=1)  # (batch_size, feat_dims+3, num_points)
            folding_result2 = self.folding2(cat2)  # (batch_size, 3, num_points)
            folding_result2 = folding_result2.transpose(1, 2)  # (batch_size, num_points ,3)

            gt_points = pts
            loss1 = self.loss_func(folding_result2, gt_points)

        return loss1, torch.zeros(1).to(loss1.device)


### encoder做法不变，完全推翻decoder 做法; 采用 FC reconstruction 的decoder 作为默认decoder. 重建 1024 point, 看一下重建global + detail 的效果。
@MODELS.register_module()
class PointCAE_transformer_supervised(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[PointCAE_transformer_supervised] ', logger ='PointCAE_transformer_supervised')
        self.config = config
        self.trans_dim = config.transformer_config.trans_dim
        if 'Drop-Patch' in config.corrupt_type:
            self.MAE_encoder = MaskTransformer(config)
        else:
            self.MAE_encoder = NormalTransformer(config)
        self.group_size = config.group_size
        self.corrupt_type = config.corrupt_type
        self.num_group = config.num_group
        self.drop_path_rate = config.transformer_config.drop_path_rate
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
        self.cls_dim = config.transformer_config.cls_dim
        # self.decoder_pos_embed = nn.Sequential(
        #     nn.Linear(3, 128),
        #     nn.GELU(),
        #     nn.Linear(128, self.trans_dim)
        # )

        # self.decoder_depth = config.transformer_config.decoder_depth
        # self.decoder_num_heads = config.transformer_config.decoder_num_heads
        # dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.decoder_depth)]
        # self.MAE_decoder = TransformerDecoderOnePEMultiX(
        #     embed_dim=self.trans_dim,
        #     depth=self.decoder_depth,
        #     drop_path_rate=dpr,
        #     num_heads=self.decoder_num_heads,
        # )

        print_log(f'[PointCAE_transformer] divide point cloud into G{self.num_group} x S{self.group_size} points ...', logger ='PointCAE_transformer')
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)

        self.cls_head_finetune = nn.Sequential(
                nn.Linear(self.trans_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(256, self.cls_dim)
            )

        # trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)

    def build_loss_func(self, loss_type):
        self.loss_ce = nn.CrossEntropyLoss()

    def get_loss_acc(self, ret, gt):
        loss = self.loss_ce(ret, gt.long())
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
        neighborhood_normalized, center = self.group_divider(pts)
        ## neighborhood: 128 * 64 * 32 * 3
        ## center: 128 * 64 * 3.
        neighborhood = neighborhood_normalized + center.unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhood, transformed_center = corrupt_data(neighborhood, center, type=self.corrupt_type)
        neighborhood_normalized = neighborhood - center.unsqueeze(2)
        transformed_neighborhood_normalized = transformed_neighborhood - transformed_center.unsqueeze(2)

        if 'Drop-Patch' in self.corrupt_type:
            x_vis, _ = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C

            concat_f = x_vis.max(1)[0] + x_vis.mean(1)
            # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)  ## B 2N
            ret = self.cls_head_finetune(concat_f)

            # # global_feature = self.increase_dim(x_vis.transpose(1, 2))  # B C N
            # global_feature = torch.max(x_vis.transpose(1, 2), dim=-1)[0]  # B C

            # coarse_point_cloud = self.coarse_pred(global_feature).reshape(B, -1, 3)  # B M C(3)
            # gt_points = pts
            #
            # loss1 = self.loss_func(coarse_point_cloud, gt_points)
        else:
            x_vis = self.MAE_encoder(transformed_neighborhood_normalized, transformed_center)
            B, _, C = x_vis.shape  # B N C

            concat_f = x_vis.max(1)[0] + x_vis.mean(1)
            # concat_f = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)  ## B 2N
            ret = self.cls_head_finetune(concat_f)

        return ret


