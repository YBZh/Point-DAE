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
from tools import builder


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


@MODELS.register_module()
class MaskFeat_dgcnn(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[MaskFeat_dgcnn] ', logger='MaskFeat_dgcnn')
        self.config = config
        self.all_patch = config.all_patch
        self.trans_dim = config.transformer_config.trans_dim
        self.MAE_encoder = MaskTransformer(config)
        self.group_size = config.group_size
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

        print_log(f'[MaskFeat_dgcnn] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
                  logger='MaskFeat_dgcnn')
        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

        # prediction head
        self.increase_dim = nn.Sequential(
            # nn.Conv1d(self.trans_dim, 1024, 1),
            # nn.BatchNorm1d(1024),
            # nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(self.trans_dim, 1024, 1)
        )

        trunc_normal_(self.mask_token, std=.02)
        self.loss = config.loss
        # loss
        self.build_loss_func(self.loss)
        # ipdb.set_trace()
        self.teacher_model = builder.model_builder(config.teacher_config)
        self._prepare_teacher_model()
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False

    def _prepare_teacher_model(self):
        # ipdb.set_trace()
        teacher_model_ckpt = self.config.teacher_config.ckpt
        ckpt = torch.load(teacher_model_ckpt, map_location='cpu')
        if self.config.teacher_config.NAME == 'DGCNN_CrossPoint':
            base_ckpt = ckpt
        else:
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
        self.teacher_model.load_state_dict(base_ckpt, strict=True)
        print_log(f'[dVAE] Successful Loading the ckpt for teacher_model from {teacher_model_ckpt}', logger='Point_BERT')

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
            pts = pts[:, :, :3].contiguous()  #### xyz & estimated normal
            ### pts: B*N*3
            #
            if self.config.teacher_config.NAME == 'DGCNN_CrossPoint':
                with torch.no_grad():
                    x_all = self.teacher_model(pts.transpose(1,2).contiguous()) ### B * D * N
            else:
                with torch.no_grad():
                    x_all = self.teacher_model.dgcnn_encoder(pts.transpose(1,2).contiguous()) ### B * D * N
            # ipdb.set_trace()
            pts = torch.cat((pts, x_all.transpose(1,2).contiguous()), -1) ### B * N * (3+D)
            ################ estimation position from pts.
            neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
            x_vis, mask = self.MAE_encoder(neighborhood, center, noaug=False)

            if self.all_patch == 'True':
                raise NotImplementedError
                # with torch.no_grad():
                #     x_all, _ = self.teacher_model.dgcnn_encoder(pts[:, :, :3].contiguous(), noaug=True)
            else:
                x_masked_gt = neighborhood_attribute[mask].max(1)[0]

            B, _, C = x_vis.shape  # B VIS C

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
            x_masked_gt = torch.nn.functional.normalize(x_masked_gt, dim=1)
            rebuild_token_label = torch.nn.functional.normalize(rebuild_token_label, dim=1)
            loss_point = (x_masked_gt - rebuild_token_label).pow(2).sum(1).mean()

            if vis:  # visualization
                raise NotImplementedError
            else:
                return loss_point, torch.zeros(1).to(loss_point.device)


# @MODELS.register_module()
# class MaskFeat_transformer_supervised(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         print_log(f'[MaskFeat_transformer_supervised] ', logger='MaskFeat_transformer_supervised')
#         self.config = config
#         self.all_patch = config.all_patch
#         self.trans_dim = config.transformer_config.trans_dim
#         self.MAE_encoder = MaskTransformer(config)
#         self.group_size = config.group_size
#         self.num_group = config.num_group
#         self.drop_path_rate = config.transformer_config.drop_path_rate
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim))
#         # self.loss_weight = nn.Parameter(torch.zeros(6))  ## log_sigma
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
#         print_log(f'[MaskFeat_transformer_supervised] divide point cloud into G{self.num_group} x S{self.group_size} points ...',
#                   logger='MaskFeat_transformer_supervised')
#         self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)
#
#         # prediction head
#         self.increase_dim = nn.Sequential(
#             # nn.Conv1d(self.trans_dim, 1024, 1),
#             # nn.BatchNorm1d(1024),
#             # nn.LeakyReLU(negative_slope=0.2),
#             nn.Conv1d(self.trans_dim, self.trans_dim, 1)
#         )
#
#         trunc_normal_(self.mask_token, std=.02)
#         self.loss = config.loss
#         # loss
#         self.build_loss_func(self.loss)
#         # ipdb.set_trace()
#         # self.supervised = PointTransformer(config.supervised_config)
#         self.supervised = builder.model_builder(config.supervised_config)
#         self._prepare_supervised()
#
#         for param in self.supervised.parameters():
#             param.requires_grad = False
#
#     def _prepare_supervised(self):
#         supervised_ckpt = self.config.supervised_config.ckpt
#         ckpt = torch.load(supervised_ckpt, map_location='cpu')
#         base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#         # ipdb.set_trace()
#         # self.supervised.state_dict().keys()
#         self.supervised.load_state_dict(base_ckpt, strict=False)
#         print_log(f'[dVAE] Successful Loading the ckpt for supervised from {supervised_ckpt}', logger='Point_BERT')
#
#     def build_loss_func(self, loss_type):
#         self.loss_func = nn.CrossEntropyLoss()
#
#     def load_model_from_ckpt(self, bert_ckpt_path):
#         if bert_ckpt_path is not None:
#             ckpt = torch.load(bert_ckpt_path)
#             base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
#
#             for k in list(base_ckpt.keys()):
#                 if k.startswith('MAE_encoder'):
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
#     def forward(self, corrupted_pts, pts, vis=False, return_feat = False, **kwargs):
#
#         if return_feat:
#             pts = torch.cat((pts, pts), -1)
#             neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
#             x_vis, mask = self.MAE_encoder(neighborhood, center, noaug = True)
#             return torch.max(x_vis.transpose(1, 2), dim=-1)[0] + x_vis.mean(1)  # B C
#         else:
#             pts = pts[:, :, :6].contiguous()  #### xyz & estimated normal
#             position = torch.abs((pts[:, :, :3] * pts[:, :, 3:]).sum(-1))
#             pts = torch.cat((pts, position.unsqueeze(-1)), -1)
#             ################ estimation position from pts.
#             neighborhood, neighborhood_attribute, center, center_attribute = self.group_divider(pts)
#             x_vis, mask = self.MAE_encoder(neighborhood, center, noaug=False)
#             B, _, C = x_vis.shape  # B VIS C
#
#             pos_emd_vis = self.decoder_pos_embed(center[~mask]).reshape(B, -1, C)
#             pos_emd_mask = self.decoder_pos_embed(center[mask]).reshape(B, -1, C)
#
#             _, N, _ = pos_emd_mask.shape
#             mask_token = self.mask_token.expand(B, N, -1)
#             x_full = torch.cat([x_vis, mask_token], dim=1)
#             pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)
#
#             if self.all_patch == 'True':
#                 x_rec = self.MAE_decoder(x_full, pos_full)
#             else:
#                 x_rec = self.MAE_decoder(x_full, pos_full, N)
#
#             B, M, C = x_rec.shape
#             rebuild_token_label = self.increase_dim(x_rec.transpose(1, 2)).transpose(1, 2).reshape(B * M, -1)  # BM Gs 6
#             # ipdb.set_trace()
#             if self.all_patch == 'True':
#                 raise NotImplementedError
#             else:
#                 with torch.no_grad():
#                     x_all = self.supervised.MAE_encoder(neighborhood, center, noaug = True)
#                     x_masked_gt = x_all[mask]
#                 # with torch.no_grad():
#                 #     group_input_tokens_sup = self.supervised.encoder(neighborhood)
#                 #     pos_sup = self.supervised.pos_embed(center)
#                 #     encoded_tokens_sup = self.supervised.blocks(group_input_tokens_sup, pos_sup)
#                 #     encoded_tokens_sup = self.supervised.norm(encoded_tokens_sup)
#                 #     x_masked_gt = encoded_tokens_sup[mask]
#
#             x_masked_gt = torch.nn.functional.normalize(x_masked_gt, dim=1)
#             rebuild_token_label = torch.nn.functional.normalize(rebuild_token_label, dim=1)
#             loss_point = (x_masked_gt - rebuild_token_label).pow(2).sum(1).mean()
#
#             if vis:  # visualization
#                 raise NotImplementedError
#             else:
#                 return loss_point, torch.zeros(1).to(loss_point.device)
#




