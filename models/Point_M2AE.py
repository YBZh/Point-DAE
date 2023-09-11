import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models.layers import trunc_normal_
from .build import MODELS
import random
from extensions.chamfer_dist import ChamferDistanceL2
from utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message

from utils.logger import *
from .Point_M2AE_modules import *
from datasets.corrupt_util_tensor import corrupt_data
import ipdb

# Hierarchical Encoder
class H_Encoder(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

        self.mask_ratio = config.mask_ratio
        self.encoder_depths = config.encoder_depths
        self.encoder_dims = config.encoder_dims
        self.local_radius = config.local_radius

        # token merging and positional embeddings
        self.token_embed = nn.ModuleList()
        self.encoder_pos_embeds = nn.ModuleList()
        for i in range(len(self.encoder_dims)):
            if i == 0:
                self.token_embed.append(Token_Embed(in_c=3, out_c=self.encoder_dims[i]))
            else:
                self.token_embed.append(Token_Embed(in_c=self.encoder_dims[i - 1], out_c=self.encoder_dims[i]))

            self.encoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.encoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.encoder_dims[i], self.encoder_dims[i]),
            ))

        # encoder blocks
        self.encoder_blocks = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.encoder_depths))]
        for i in range(len(self.encoder_depths)):
            self.encoder_blocks.append(Encoder_Block(
                embed_dim=self.encoder_dims[i],
                depth=self.encoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.encoder_depths[i]],
                num_heads=config.num_heads,
            ))
            depth_count += self.encoder_depths[i]

        self.encoder_norms = nn.ModuleList()
        for i in range(len(self.encoder_depths)):
            self.encoder_norms.append(nn.LayerNorm(self.encoder_dims[i]))

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

    def rand_mask(self, center):
        B, G, _ = center.shape
        self.num_mask = int(self.mask_ratio * G)
        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G - self.num_mask),
                np.ones(self.num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)
        return overall_mask.to(center.device)  # B G

    def local_att_mask(self, xyz, radius, dist=None):
        with torch.no_grad():
            if dist is None or dist.shape[1] != xyz.shape[1]:
                dist = torch.cdist(xyz, xyz, p=2)
            # entries that are True in the mask do not contribute to self-attention
            # so points outside the radius are not considered
            mask = dist >= radius
        return mask, dist

    def forward(self, neighborhoods, centers, idxs, eval=False):
        # generate mask at the highest level
        bool_masked_pos = []
        if eval:
            # no mask
            B, G, _ = centers[-1].shape
            bool_masked_pos.append(torch.zeros(B, G).bool().cuda())
        else:
            # mask_index: 1, mask; 0, vis
            bool_masked_pos.append(self.rand_mask(centers[-1]))

        # Multi-scale Masking by back-propagation
        for i in range(len(neighborhoods) - 1, 0, -1):
            b, g, k, _ = neighborhoods[i].shape
            idx = idxs[i].reshape(b * g, -1)
            idx_masked = ~(bool_masked_pos[-1].reshape(-1).unsqueeze(-1)) * idx
            idx_masked = idx_masked.reshape(-1).long()
            masked_pos = torch.ones(b * centers[i - 1].shape[1]).cuda().scatter(0, idx_masked, 0).bool()
            bool_masked_pos.append(masked_pos.reshape(b, centers[i - 1].shape[1]))

        # hierarchical encoding
        bool_masked_pos.reverse()
        x_vis_list = []
        mask_vis_list = []
        xyz_dist = None
        for i in range(len(centers)):
            # 1st-layer encoder, conduct token embedding
            if i == 0:
                group_input_tokens = self.token_embed[i](neighborhoods[0])
            # intermediate layers, conduct token merging
            else:
                b, g1, _ = x_vis.shape
                b, g2, k2, _ = neighborhoods[i].shape
                x_vis_neighborhoods = x_vis.reshape(b * g1, -1)[idxs[i], :].reshape(b, g2, k2, -1)
                group_input_tokens = self.token_embed[i](x_vis_neighborhoods)

            # visible_index
            bool_vis_pos = ~(bool_masked_pos[i])
            batch_size, seq_len, C = group_input_tokens.size()

            # Due to Multi-scale Masking different, samples of a batch have varying numbers of visible tokens
            # find the longest visible sequence in the batch
            vis_tokens_len = bool_vis_pos.long().sum(dim=1)
            max_tokens_len = torch.max(vis_tokens_len)
            # use the longest length (max_tokens_len) to construct tensors
            x_vis = torch.zeros(batch_size, max_tokens_len, C).cuda()
            masked_center = torch.zeros(batch_size, max_tokens_len, 3).cuda()
            mask_vis = torch.ones(batch_size, max_tokens_len, max_tokens_len).cuda()

            for bz in range(batch_size):
                # inject valid visible tokens
                vis_tokens = group_input_tokens[bz][bool_vis_pos[bz]]
                x_vis[bz][0: vis_tokens_len[bz]] = vis_tokens
                # inject valid visible centers
                vis_centers = centers[i][bz][bool_vis_pos[bz]]
                masked_center[bz][0: vis_tokens_len[bz]] = vis_centers
                # the mask for valid visible tokens/centers
                mask_vis[bz][0: vis_tokens_len[bz], 0: vis_tokens_len[bz]] = 0

            if self.local_radius[i] > 0:
                mask_radius, xyz_dist = self.local_att_mask(masked_center, self.local_radius[i], xyz_dist)
                # disabled for pre-training, this step would not change mask_vis by *
                mask_vis_att = mask_radius * mask_vis
            else:
                mask_vis_att = mask_vis

            pos = self.encoder_pos_embeds[i](masked_center)

            x_vis = self.encoder_blocks[i](x_vis, pos, mask_vis_att)
            x_vis_list.append(x_vis)
            mask_vis_list.append(~(mask_vis[:, :, 0].bool()))

            if i == len(centers) - 1:
                pass
            else:
                group_input_tokens[bool_vis_pos] = x_vis[~(mask_vis[:, :, 0].bool())]
                x_vis = group_input_tokens

        for i in range(len(x_vis_list)):
            x_vis_list[i] = self.encoder_norms[i](x_vis_list[i])

        return x_vis_list, mask_vis_list, bool_masked_pos


@MODELS.register_module()
class Point_M2AE(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE]', logger='Point_M2AE')
        self.config = config

        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)

        # hierarchical decoder
        self.decoder_depths = config.decoder_depths
        self.decoder_dims = config.decoder_dims
        self.decoder_up_blocks = config.decoder_up_blocks

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        for i in range(0, len(self.decoder_dims)):
            # decoder block
            self.h_decoder.append(Decoder_Block(
                embed_dim=self.decoder_dims[i],
                depth=self.decoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
                num_heads=config.num_heads,
            ))
            depth_count += self.decoder_depths[i]
            # decoder's positional embeddings
            self.decoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.decoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.decoder_dims[i], self.decoder_dims[i])
            ))
            # token propagation
            if i > 0:
                self.token_prop.append(PointNetFeaturePropagation(
                    self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
                    blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
                ))
        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])

        # prediction head
        self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)
        # loss
        self.rec_loss = ChamferDistanceL2().cuda()

    # def forward(self, pts, eval=False, **kwargs):
    #     # multi-scale representations of point clouds
    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        pts = pts[:, :, :3].contiguous()  # 128*2048*3
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
                # neighborhood: 128 * 512 * 16 * 3 (batch/G number/G size/xyz)
                # center: 128*512*3
                # idx: 1048576 (大概是 128*512*16)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
                # neighborhood: 128*256*8*3(batch/G number/G size/xyz)
                # idx: 262144 (128*256*8)
                ## -------------------------------------------------------------
                # neighborhood: 128*64*8*3(batch/G number/G size/xyz)
                # idx:  (128*64*8)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        x_vis_list, mask_vis_list, masks = self.h_encoder(neighborhoods, centers, idxs)
        ## x_vis_list: [128*378*96, 128*103*192, 128*13*384]
        ## mask_vis_list: [128*378, 128*103, 128*13] True or False.
        ## masks: [128*512, 128*256, 128*64] True or False
        # hierarchical decoder
        centers.reverse() ## list [0,1,2] --> list [2,1,0]
        neighborhoods.reverse()
        x_vis_list.reverse()
        masks.reverse()

        for i in range(len(self.decoder_dims)):
            center = centers[i]
            # 1st-layer decoder, concatenate visible and masked tokens
            if i == 0:
                x_full, mask = x_vis_list[i], masks[i] # 128*13*384, 128*64
                B, _, C = x_full.shape
                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)

                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                _, N, _ = pos_emd_mask.shape  ## 51
                mask_token = self.mask_token.expand(B, N, -1)  ## B, N, 384
                x_full = torch.cat([x_full, mask_token], dim=1)  ## 128*64*384

            else:
                x_vis = x_vis_list[i]    # 128, 103, 192
                bool_vis_pos = ~masks[i] # 128, 256
                mask_vis = mask_vis_list[i] # 128, 103
                B, N, _ = center.shape
                _, _, C = x_vis.shape
                x_full_en = torch.zeros(B, N, C).cuda()
                x_full_en[bool_vis_pos] = x_vis[mask_vis] ## 128, 256, 192

                # token propagation
                if i == 1:
                    x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)
                    ## center: 128*256*3, center_0:128*64*3
                    ## x_full: 128*256*192
                else:
                    x_full = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full) ## 128*64*384

        # reconstruction
        x_full = self.decoder_norm(x_full)
        B, N, C = x_full.shape
        x_rec = x_full[masks[-2]].reshape(-1, C)
        L, _ = x_rec.shape

        rec_points = self.rec_head(x_rec.unsqueeze(-1)).reshape(L, -1, 3)
        gt_points = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)

        # CD loss
        loss = self.rec_loss(rec_points, gt_points)
        return loss, torch.zeros(1).to(loss.device)

## three level global shape reconstruction. Not reasonable, since the early feature are not shape aware.
@MODELS.register_module()
class Point_M2AE_with_fc_center_p(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE_with_fc_center_p]', logger='Point_M2AE_with_fc_center_p')
        self.config = config

        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)
        self.coarse_pred_0 = nn.Sequential(
            nn.Linear(config.encoder_dims[2], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * config.num_groups[2])
        )

        self.coarse_pred_1 = nn.Sequential(
            nn.Linear(config.encoder_dims[1], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * config.num_groups[1])
        )

        self.coarse_pred_2 = nn.Sequential(
            nn.Linear(config.encoder_dims[0], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * config.num_groups[0])
        )

        # hierarchical decoder
        self.decoder_depths = config.decoder_depths
        self.decoder_dims = config.decoder_dims
        self.decoder_up_blocks = config.decoder_up_blocks

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        for i in range(0, len(self.decoder_dims)):
            # decoder block
            self.h_decoder.append(Decoder_Block(
                embed_dim=self.decoder_dims[i],
                depth=self.decoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
                num_heads=config.num_heads,
            ))
            depth_count += self.decoder_depths[i]
            # decoder's positional embeddings
            self.decoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.decoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.decoder_dims[i], self.decoder_dims[i])
            ))
            # token propagation
            if i > 0:
                self.token_prop.append(PointNetFeaturePropagation(
                    self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
                    blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
                ))
        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])

        # prediction head
        self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)
        # loss
        self.rec_loss = ChamferDistanceL2().cuda()

    # def forward(self, pts, eval=False, **kwargs):
    #     # multi-scale representations of point clouds
    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        pts = pts[:, :, :3].contiguous()  # 128*2048*3
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
                # neighborhood: 128 * 512 * 16 * 3 (batch/G number/G size/xyz)
                # center: 128*512*3
                # idx: 1048576 (大概是 128*512*16)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
                # neighborhood: 128*256*8*3(batch/G number/G size/xyz)
                # idx: 262144 (128*256*8)
                ## -------------------------------------------------------------
                # neighborhood: 128*64*8*3(batch/G number/G size/xyz)
                # idx:  (128*64*8)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        x_vis_list, mask_vis_list, masks = self.h_encoder(neighborhoods, centers, idxs)
        ## x_vis_list: [128*378*96, 128*103*192, 128*13*384]
        ## mask_vis_list: [128*378, 128*103, 128*13] True or False.
        ## masks: [128*512, 128*256, 128*64] True or False
        # hierarchical decoder
        centers.reverse() ## list [0,1,2] --> list [2,1,0]
        neighborhoods.reverse()
        x_vis_list.reverse()
        masks.reverse()

        B = pts.size(0)
        global_feature_0 = torch.max(x_vis_list[0].transpose(1, 2), dim=-1)[0] + x_vis_list[0].mean(1)  # B C
        coarse_point_cloud_0 = self.coarse_pred_0(global_feature_0).reshape(B, -1, 3)  # B M C(3)
        gt_points_center_0 = centers[0]

        global_feature_1 = torch.max(x_vis_list[1].transpose(1, 2), dim=-1)[0] + x_vis_list[1].mean(1)  # B C
        coarse_point_cloud_1 = self.coarse_pred_1(global_feature_1).reshape(B, -1, 3)  # B M C(3)
        gt_points_center_1 = centers[1]

        global_feature_2 = torch.max(x_vis_list[2].transpose(1, 2), dim=-1)[0] + x_vis_list[2].mean(1)  # B C
        coarse_point_cloud_2 = self.coarse_pred_2(global_feature_2).reshape(B, -1, 3)  # B M C(3)
        gt_points_center_2 = centers[2]

        loss_2 = self.rec_loss(coarse_point_cloud_0, gt_points_center_0) + \
                 self.rec_loss(coarse_point_cloud_1, gt_points_center_1) + \
                 self.rec_loss(coarse_point_cloud_2, gt_points_center_2)

        for i in range(len(self.decoder_dims)):
            center = centers[i]
            # 1st-layer decoder, concatenate visible and masked tokens
            if i == 0:
                x_full, mask = x_vis_list[i], masks[i] # 128*13*384, 128*64
                B, _, C = x_full.shape
                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)

                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                _, N, _ = pos_emd_mask.shape  ## 51
                mask_token = self.mask_token.expand(B, N, -1)  ## B, N, 384
                x_full = torch.cat([x_full, mask_token], dim=1)  ## 128*64*384

            else:
                x_vis = x_vis_list[i]    # 128, 103, 192
                bool_vis_pos = ~masks[i] # 128, 256
                mask_vis = mask_vis_list[i] # 128, 103
                B, N, _ = center.shape
                _, _, C = x_vis.shape
                x_full_en = torch.zeros(B, N, C).cuda()
                x_full_en[bool_vis_pos] = x_vis[mask_vis] ## 128, 256, 192

                # token propagation
                if i == 1:
                    x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)
                    ## center: 128*256*3, center_0:128*64*3
                    ## x_full: 128*256*192
                else:
                    x_full = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full) ## 128*64*384

        # reconstruction
        x_full = self.decoder_norm(x_full)
        B, N, C = x_full.shape
        x_rec = x_full[masks[-2]].reshape(-1, C)
        L, _ = x_rec.shape

        rec_points = self.rec_head(x_rec.unsqueeze(-1)).reshape(L, -1, 3)
        gt_points = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)

        # CD loss
        loss = self.rec_loss(rec_points, gt_points)
        return loss, loss_2

## only one level global shape reconstruction at the end of the encoder; this should work.
@MODELS.register_module()
class Point_M2AE_with_fc_center_p_v1(nn.Module):
    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE_with_fc_center_p]', logger='Point_M2AE_with_fc_center_p')
        self.config = config

        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)
        self.coarse_pred_0 = nn.Sequential(
            nn.Linear(config.encoder_dims[2], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * config.num_groups[2])
        )

        # self.coarse_pred_1 = nn.Sequential(
        #     nn.Linear(config.encoder_dims[1], 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * config.num_groups[1])
        # )
        #
        # self.coarse_pred_2 = nn.Sequential(
        #     nn.Linear(config.encoder_dims[0], 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * config.num_groups[0])
        # )

        # hierarchical decoder
        self.decoder_depths = config.decoder_depths
        self.decoder_dims = config.decoder_dims
        self.decoder_up_blocks = config.decoder_up_blocks

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        for i in range(0, len(self.decoder_dims)):
            # decoder block
            self.h_decoder.append(Decoder_Block(
                embed_dim=self.decoder_dims[i],
                depth=self.decoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
                num_heads=config.num_heads,
            ))
            depth_count += self.decoder_depths[i]
            # decoder's positional embeddings
            self.decoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.decoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.decoder_dims[i], self.decoder_dims[i])
            ))
            # token propagation
            if i > 0:
                self.token_prop.append(PointNetFeaturePropagation(
                    self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
                    blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
                ))
        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])

        # prediction head
        self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)
        # loss
        self.rec_loss = ChamferDistanceL2().cuda()

    # def forward(self, pts, eval=False, **kwargs):
    #     # multi-scale representations of point clouds
    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        pts = pts[:, :, :3].contiguous()  # 128*2048*3
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
                # neighborhood: 128 * 512 * 16 * 3 (batch/G number/G size/xyz)
                # center: 128*512*3
                # idx: 1048576 (大概是 128*512*16)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
                # neighborhood: 128*256*8*3(batch/G number/G size/xyz)
                # idx: 262144 (128*256*8)
                ## -------------------------------------------------------------
                # neighborhood: 128*64*8*3(batch/G number/G size/xyz)
                # idx:  (128*64*8)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        x_vis_list, mask_vis_list, masks = self.h_encoder(neighborhoods, centers, idxs)
        ## x_vis_list: [128*378*96, 128*103*192, 128*13*384]
        ## mask_vis_list: [128*378, 128*103, 128*13] True or False.
        ## masks: [128*512, 128*256, 128*64] True or False
        # hierarchical decoder
        centers.reverse() ## list [0,1,2] --> list [2,1,0]
        neighborhoods.reverse()
        x_vis_list.reverse()
        masks.reverse()

        B = pts.size(0)
        global_feature_0 = torch.max(x_vis_list[0].transpose(1, 2), dim=-1)[0] + x_vis_list[0].mean(1)  # B C
        coarse_point_cloud_0 = self.coarse_pred_0(global_feature_0).reshape(B, -1, 3)  # B M C(3)
        gt_points_center_0 = centers[0]

        # global_feature_1 = torch.max(x_vis_list[1].transpose(1, 2), dim=-1)[0] + x_vis_list[1].mean(1)  # B C
        # coarse_point_cloud_1 = self.coarse_pred_1(global_feature_1).reshape(B, -1, 3)  # B M C(3)
        # gt_points_center_1 = centers[1]
        #
        # global_feature_2 = torch.max(x_vis_list[2].transpose(1, 2), dim=-1)[0] + x_vis_list[2].mean(1)  # B C
        # coarse_point_cloud_2 = self.coarse_pred_2(global_feature_2).reshape(B, -1, 3)  # B M C(3)
        # gt_points_center_2 = centers[2]

        loss_2 = self.rec_loss(coarse_point_cloud_0, gt_points_center_0)

        for i in range(len(self.decoder_dims)):
            center = centers[i]
            # 1st-layer decoder, concatenate visible and masked tokens
            if i == 0:
                x_full, mask = x_vis_list[i], masks[i] # 128*13*384, 128*64
                B, _, C = x_full.shape
                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)

                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                _, N, _ = pos_emd_mask.shape  ## 51
                mask_token = self.mask_token.expand(B, N, -1)  ## B, N, 384
                x_full = torch.cat([x_full, mask_token], dim=1)  ## 128*64*384

            else:
                x_vis = x_vis_list[i]    # 128, 103, 192
                bool_vis_pos = ~masks[i] # 128, 256
                mask_vis = mask_vis_list[i] # 128, 103
                B, N, _ = center.shape
                _, _, C = x_vis.shape
                x_full_en = torch.zeros(B, N, C).cuda()
                x_full_en[bool_vis_pos] = x_vis[mask_vis] ## 128, 256, 192

                # token propagation
                if i == 1:
                    x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)
                    ## center: 128*256*3, center_0:128*64*3
                    ## x_full: 128*256*192
                else:
                    x_full = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full) ## 128*64*384

        # reconstruction
        x_full = self.decoder_norm(x_full)
        B, N, C = x_full.shape
        x_rec = x_full[masks[-2]].reshape(-1, C)
        L, _ = x_rec.shape

        rec_points = self.rec_head(x_rec.unsqueeze(-1)).reshape(L, -1, 3)
        gt_points = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)

        # CD loss
        loss = self.rec_loss(rec_points, gt_points)
        return loss, loss_2


########################### M2AE + Affine Transformation.
@MODELS.register_module()
class Point_MDAE_with_fc_center_p(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MDAE]', logger='Point_MDAE')
        self.config = config

        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.corrupt_type = config.corrupt_type
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)

        self.coarse_pred_0 = nn.Sequential(
            nn.Linear(config.encoder_dims[2], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * config.num_groups[2])
        )

        self.coarse_pred_1 = nn.Sequential(
            nn.Linear(config.encoder_dims[1], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * config.num_groups[1])
        )

        self.coarse_pred_2 = nn.Sequential(
            nn.Linear(config.encoder_dims[0], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * config.num_groups[0])
        )
        # hierarchical decoder
        self.decoder_depths = config.decoder_depths
        self.decoder_dims = config.decoder_dims
        self.decoder_up_blocks = config.decoder_up_blocks

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        for i in range(0, len(self.decoder_dims)):
            # decoder block
            self.h_decoder.append(Decoder_Block(
                embed_dim=self.decoder_dims[i],
                depth=self.decoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
                num_heads=config.num_heads,
            ))
            depth_count += self.decoder_depths[i]
            # decoder's positional embeddings
            self.decoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.decoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.decoder_dims[i], self.decoder_dims[i])
            ))
            # token propagation
            if i > 0:
                self.token_prop.append(PointNetFeaturePropagation(
                    self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
                    blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
                ))
        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])

        # prediction head
        self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)
        # loss
        self.rec_loss = ChamferDistanceL2().cuda()

    # def forward(self, pts, eval=False, **kwargs):
    #     # multi-scale representations of point clouds
    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        pts = pts[:, :, :3].contiguous()  # 128*2048*3
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
                # neighborhood: 128 * 512 * 16 * 3 (batch/G number/G size/xyz)
                # center: 128*512*3
                # idx: 1048576 (大概是 128*512*16)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
                # neighborhood: 128*256*8*3(batch/G number/G size/xyz)
                # idx: 262144 (128*256*8)
                ## -------------------------------------------------------------
                # neighborhood: 128*64*8*3(batch/G number/G size/xyz)
                # idx:  (128*64*8)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        for i in range(len(self.group_dividers)):
            neighborhoods[i] = neighborhoods[i] + centers[i].unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhoods, transformed_centers = corrupt_data(neighborhoods, centers, type=self.corrupt_type)
        for i in range(len(self.group_dividers)):
            neighborhoods[i] = neighborhoods[i] - centers[i].unsqueeze(2)
            transformed_neighborhoods[i] = transformed_neighborhoods[i] - transformed_centers[i].unsqueeze(2)

        # hierarchical encoder
        x_vis_list, mask_vis_list, masks = self.h_encoder(transformed_neighborhoods, transformed_centers, idxs)
        ## x_vis_list: [128*378*96, 128*103*192, 128*13*384]   ---> 三个global shape 重建loss. 因为这里有三块。
        ## mask_vis_list: [128*378, 128*103, 128*13] True or False.
        ## masks: [128*512, 128*256, 128*64] True or False
        # hierarchical decoder
        centers.reverse() ## list [0,1,2] --> list [2,1,0]
        neighborhoods.reverse()
        # ipdb.set_trace()
        transformed_centers.reverse()
        transformed_neighborhoods.reverse()
        x_vis_list.reverse()
        masks.reverse()

        B = pts.size(0)
        global_feature_0 = torch.max(x_vis_list[0].transpose(1, 2), dim=-1)[0] + x_vis_list[0].mean(1)  # B C
        coarse_point_cloud_0 = self.coarse_pred_0(global_feature_0).reshape(B, -1, 3)  # B M C(3)
        gt_points_center_0 = centers[0]

        global_feature_1 = torch.max(x_vis_list[1].transpose(1, 2), dim=-1)[0] + x_vis_list[1].mean(1)  # B C
        coarse_point_cloud_1 = self.coarse_pred_1(global_feature_1).reshape(B, -1, 3)  # B M C(3)
        gt_points_center_1 = centers[1]

        global_feature_2 = torch.max(x_vis_list[2].transpose(1, 2), dim=-1)[0] + x_vis_list[2].mean(1)  # B C
        coarse_point_cloud_2 = self.coarse_pred_2(global_feature_2).reshape(B, -1, 3)  # B M C(3)
        gt_points_center_2 = centers[2]

        loss_2 = self.rec_loss(coarse_point_cloud_0, gt_points_center_0) + \
                 self.rec_loss(coarse_point_cloud_1, gt_points_center_1) + \
                 self.rec_loss(coarse_point_cloud_2, gt_points_center_2)

        for i in range(len(self.decoder_dims)):
            center = centers[i]
            # 1st-layer decoder, concatenate visible and masked tokens
            if i == 0:
                x_full, mask = x_vis_list[i], masks[i] # 128*13*384, 128*64
                B, _, C = x_full.shape
                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)

                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                _, N, _ = pos_emd_mask.shape  ## 51
                mask_token = self.mask_token.expand(B, N, -1)  ## B, N, 384
                x_full = torch.cat([x_full, mask_token], dim=1)  ## 128*64*384

            else:
                x_vis = x_vis_list[i]    # 128, 103, 192
                bool_vis_pos = ~masks[i] # 128, 256
                mask_vis = mask_vis_list[i] # 128, 103
                B, N, _ = center.shape
                _, _, C = x_vis.shape
                x_full_en = torch.zeros(B, N, C).cuda()
                x_full_en[bool_vis_pos] = x_vis[mask_vis] ## 128, 256, 192

                # token propagation
                if i == 1:
                    x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)
                    ## center: 128*256*3, center_0:128*64*3
                    ## x_full: 128*256*192
                else:
                    x_full = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full) ## 128*64*384

        # reconstruction
        x_full = self.decoder_norm(x_full)
        B, N, C = x_full.shape
        x_rec = x_full[masks[-2]].reshape(-1, C)
        L, _ = x_rec.shape

        rec_points = self.rec_head(x_rec.unsqueeze(-1)).reshape(L, -1, 3)
        gt_points = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)

        # CD loss
        loss = self.rec_loss(rec_points, gt_points)
        return loss, loss_2

########################### M2AE + Affine Transformation.
@MODELS.register_module()
class Point_MDAE_with_fc_center_p_v1(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_MDAE]', logger='Point_MDAE')
        self.config = config

        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.corrupt_type = config.corrupt_type
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)

        self.coarse_pred_0 = nn.Sequential(
            nn.Linear(config.encoder_dims[2], 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3 * config.num_groups[2])
        )

        # self.coarse_pred_1 = nn.Sequential(
        #     nn.Linear(config.encoder_dims[1], 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * config.num_groups[1])
        # )
        #
        # self.coarse_pred_2 = nn.Sequential(
        #     nn.Linear(config.encoder_dims[0], 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 3 * config.num_groups[0])
        # )
        # hierarchical decoder
        self.decoder_depths = config.decoder_depths
        self.decoder_dims = config.decoder_dims
        self.decoder_up_blocks = config.decoder_up_blocks

        self.mask_token = nn.Parameter(torch.zeros(1, self.decoder_dims[0]))
        trunc_normal_(self.mask_token, std=.02)

        self.h_decoder = nn.ModuleList()
        self.decoder_pos_embeds = nn.ModuleList()
        self.token_prop = nn.ModuleList()

        depth_count = 0
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, sum(self.decoder_depths))]
        for i in range(0, len(self.decoder_dims)):
            # decoder block
            self.h_decoder.append(Decoder_Block(
                embed_dim=self.decoder_dims[i],
                depth=self.decoder_depths[i],
                drop_path_rate=dpr[depth_count: depth_count + self.decoder_depths[i]],
                num_heads=config.num_heads,
            ))
            depth_count += self.decoder_depths[i]
            # decoder's positional embeddings
            self.decoder_pos_embeds.append(nn.Sequential(
                nn.Linear(3, self.decoder_dims[i]),
                nn.GELU(),
                nn.Linear(self.decoder_dims[i], self.decoder_dims[i])
            ))
            # token propagation
            if i > 0:
                self.token_prop.append(PointNetFeaturePropagation(
                    self.decoder_dims[i] + self.decoder_dims[i - 1], self.decoder_dims[i],
                    blocks=self.decoder_up_blocks[i - 1], groups=1, res_expansion=1.0, bias=True
                ))
        self.decoder_norm = nn.LayerNorm(self.decoder_dims[-1])

        # prediction head
        self.rec_head = nn.Conv1d(self.decoder_dims[-1], 3 * self.group_sizes[0], 1)
        # loss
        self.rec_loss = ChamferDistanceL2().cuda()

    # def forward(self, pts, eval=False, **kwargs):
    #     # multi-scale representations of point clouds
    def forward(self, corrupted_pts, pts, vis=False, **kwargs):
        pts = pts[:, :, :3].contiguous()  # 128*2048*3
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
                # neighborhood: 128 * 512 * 16 * 3 (batch/G number/G size/xyz)
                # center: 128*512*3
                # idx: 1048576 (大概是 128*512*16)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
                # neighborhood: 128*256*8*3(batch/G number/G size/xyz)
                # idx: 262144 (128*256*8)
                ## -------------------------------------------------------------
                # neighborhood: 128*64*8*3(batch/G number/G size/xyz)
                # idx:  (128*64*8)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        for i in range(len(self.group_dividers)):
            neighborhoods[i] = neighborhoods[i] + centers[i].unsqueeze(2)
        ## 在这里把neighborhood 和 center 做一个一模一样的 affine transformation, after transformation --- predict --> before transformation.
        transformed_neighborhoods, transformed_centers = corrupt_data(neighborhoods, centers, type=self.corrupt_type)
        for i in range(len(self.group_dividers)):
            neighborhoods[i] = neighborhoods[i] - centers[i].unsqueeze(2)
            transformed_neighborhoods[i] = transformed_neighborhoods[i] - transformed_centers[i].unsqueeze(2)

        # hierarchical encoder
        x_vis_list, mask_vis_list, masks = self.h_encoder(transformed_neighborhoods, transformed_centers, idxs)
        ## x_vis_list: [128*378*96, 128*103*192, 128*13*384]   ---> 三个global shape 重建loss. 因为这里有三块。
        ## mask_vis_list: [128*378, 128*103, 128*13] True or False.
        ## masks: [128*512, 128*256, 128*64] True or False
        # hierarchical decoder
        centers.reverse() ## list [0,1,2] --> list [2,1,0]
        neighborhoods.reverse()
        # ipdb.set_trace()
        transformed_centers.reverse()
        transformed_neighborhoods.reverse()
        x_vis_list.reverse()
        masks.reverse()

        B = pts.size(0)
        global_feature_0 = torch.max(x_vis_list[0].transpose(1, 2), dim=-1)[0] + x_vis_list[0].mean(1)  # B C
        coarse_point_cloud_0 = self.coarse_pred_0(global_feature_0).reshape(B, -1, 3)  # B M C(3)
        gt_points_center_0 = centers[0]

        # global_feature_1 = torch.max(x_vis_list[1].transpose(1, 2), dim=-1)[0] + x_vis_list[1].mean(1)  # B C
        # coarse_point_cloud_1 = self.coarse_pred_1(global_feature_1).reshape(B, -1, 3)  # B M C(3)
        # gt_points_center_1 = centers[1]
        #
        # global_feature_2 = torch.max(x_vis_list[2].transpose(1, 2), dim=-1)[0] + x_vis_list[2].mean(1)  # B C
        # coarse_point_cloud_2 = self.coarse_pred_2(global_feature_2).reshape(B, -1, 3)  # B M C(3)
        # gt_points_center_2 = centers[2]

        loss_2 = self.rec_loss(coarse_point_cloud_0, gt_points_center_0)

        for i in range(len(self.decoder_dims)):
            center = centers[i]
            # 1st-layer decoder, concatenate visible and masked tokens
            if i == 0:
                x_full, mask = x_vis_list[i], masks[i] # 128*13*384, 128*64
                B, _, C = x_full.shape
                center_0 = torch.cat((center[~mask].reshape(B, -1, 3), center[mask].reshape(B, -1, 3)), dim=1)

                pos_emd_vis = self.decoder_pos_embeds[i](center[~mask]).reshape(B, -1, C)
                pos_emd_mask = self.decoder_pos_embeds[i](center[mask]).reshape(B, -1, C)
                pos_full = torch.cat([pos_emd_vis, pos_emd_mask], dim=1)

                _, N, _ = pos_emd_mask.shape  ## 51
                mask_token = self.mask_token.expand(B, N, -1)  ## B, N, 384
                x_full = torch.cat([x_full, mask_token], dim=1)  ## 128*64*384

            else:
                x_vis = x_vis_list[i]    # 128, 103, 192
                bool_vis_pos = ~masks[i] # 128, 256
                mask_vis = mask_vis_list[i] # 128, 103
                B, N, _ = center.shape
                _, _, C = x_vis.shape
                x_full_en = torch.zeros(B, N, C).cuda()
                x_full_en[bool_vis_pos] = x_vis[mask_vis] ## 128, 256, 192

                # token propagation
                if i == 1:
                    x_full = self.token_prop[i - 1](center, center_0, x_full_en, x_full)
                    ## center: 128*256*3, center_0:128*64*3
                    ## x_full: 128*256*192
                else:
                    x_full = self.token_prop[i - 1](center, centers[i - 1], x_full_en, x_full)
                pos_full = self.decoder_pos_embeds[i](center)

            x_full = self.h_decoder[i](x_full, pos_full) ## 128*64*384

        # reconstruction
        x_full = self.decoder_norm(x_full)
        B, N, C = x_full.shape
        x_rec = x_full[masks[-2]].reshape(-1, C)
        L, _ = x_rec.shape

        rec_points = self.rec_head(x_rec.unsqueeze(-1)).reshape(L, -1, 3)
        gt_points = neighborhoods[-2][masks[-2]].reshape(L, -1, 3)

        # CD loss
        loss = self.rec_loss(rec_points, gt_points)
        return loss, loss_2



@MODELS.register_module()
class Point_M2AE_SVMFeature(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE_SVMFeature]', logger='Point_M2AE_SVMFeature')
        self.config = config

        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.group_dividers = nn.ModuleList()
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)
        self.build_loss_func()

        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
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
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        # for linear svm
        x_vis_list, mask_vis_list, _ = self.h_encoder(neighborhoods, centers, idxs, eval=True)
        x_vis = x_vis_list[-1]
        return x_vis.mean(1) + x_vis.max(1)[0]

@MODELS.register_module()
class Point_M2AE_Finetune(nn.Module):

    def __init__(self, config):
        super().__init__()
        print_log(f'[Point_M2AE_Finetune]', logger='Point_M2AE_Finetune')
        self.config = config

        # tokenizers
        self.group_sizes = config.group_sizes
        self.num_groups = config.num_groups
        self.group_dividers = nn.ModuleList()
        self.trans_dim = config.encoder_dims[-1]
        self.cls_dim = config.cls_dim
        for i in range(len(self.group_sizes)):
            self.group_dividers.append(Group(num_group=self.num_groups[i], group_size=self.group_sizes[i]))

        # hierarchical encoder
        self.h_encoder = H_Encoder(config)
        self.build_loss_func()

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
        # trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.cls_pos, std=.02)

    def build_loss_func(self):
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
        neighborhoods, centers, idxs = [], [], []
        for i in range(len(self.group_dividers)):
            if i == 0:
                neighborhood, center, idx = self.group_dividers[i](pts)
            else:
                neighborhood, center, idx = self.group_dividers[i](center)
            neighborhoods.append(neighborhood)
            centers.append(center)
            idxs.append(idx)  # neighbor indices

        # hierarchical encoder
        # for linear svm
        x_vis_list, mask_vis_list, _ = self.h_encoder(neighborhoods, centers, idxs, eval=True)
        x_vis = x_vis_list[-1]
        x_vis = x_vis.mean(1) + x_vis.max(1)[0]

        return self.cls_head_finetune(x_vis)
