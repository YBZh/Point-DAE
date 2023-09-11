#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/AnTao97/dgcnn.pytorch/blob/master/model.py
#  Ref: https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/part_seg/train_multi_gpu.py

import pdb, torch, torch.nn as nn, torch.nn.functional as F
from models.pointnet_not_util import PointNetEncoderNoT_partseg
from logger import get_missing_parameters_message, get_unexpected_parameters_message
import ipdb

class get_model(nn.Module):
    def __init__(self, cls_dim):
        super(get_model, self).__init__()
        # self.config = config
        # self.cls_dim = config.cls_dim
        # self.smoothing = config.smoothloss
        self.part_num = cls_dim
        self.pointnet_encoder = PointNetEncoderNoT_partseg(global_feat=True, channel=3)
        # self.cls_head_finetune = nn.Sequential(
        #         nn.Linear(2048, 512),
        #         nn.BatchNorm1d(512),
        #         nn.ReLU(inplace=True),
        #         nn.Dropout(0.3),
        #         nn.Linear(512, 256),
        #         nn.BatchNorm1d(256),
        #         nn.ReLU(inplace=True),
        #         nn.Linear(256, self.cls_dim)
        #     )
        self.convs1 = nn.Conv1d(4944, 256, 1)
        self.convs2 = nn.Conv1d(256, 256, 1)
        self.convs3 = nn.Conv1d(256, 128, 1)
        self.convs4 = nn.Conv1d(128, self.part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)
        # self.build_loss_func()

    def load_model_from_ckpt(self, bert_ckpt_path):
        # ipdb.set_trace()
        if bert_ckpt_path is not None:
            ckpt = torch.load(bert_ckpt_path)
            base_ckpt = {k.replace("module.", ""): v for k, v in ckpt['base_model'].items()}
            print(ckpt['base_model'].keys())
            # for k in list(base_ckpt.keys()):
            #     if k.startswith('MAE_encoder'):
            #         base_ckpt[k[len('MAE_encoder.'):]] = base_ckpt[k]
            #         del base_ckpt[k]
            #     elif k.startswith('base_model'):
            #         base_ckpt[k[len('base_model.'):]] = base_ckpt[k]
            #         del base_ckpt[k]

            incompatible = self.load_state_dict(base_ckpt, strict=False)

            if incompatible.missing_keys:
                print('missing_keys')
                print(
                        get_missing_parameters_message(incompatible.missing_keys)
                    )
            if incompatible.unexpected_keys:
                print('unexpected_keys')
                print(
                        get_unexpected_parameters_message(incompatible.unexpected_keys)

                    )

            print(f'[Transformer] Successful Loading the ckpt from {bert_ckpt_path}')

    def forward(self, pts, label):
        B, D, N = pts.size()
        # pts = pts[:, :, :3]
        # pts = pts.transpose(1, 2).contiguous()
        out1, out2, out3, out4, out5 = self.pointnet_encoder(pts)
        out_max = torch.max(out5, 2, keepdim=False)[0]
        out_max = torch.cat([out_max, label.squeeze(1)], 1)
        # ipdb.set_trace()
        expand = out_max.view(-1, 2048 + 16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)

        net = F.relu(self.bns1(self.convs1(concat)))
        net = F.relu(self.bns2(self.convs2(net)))
        net = F.relu(self.bns3(self.convs3(net)))
        net = self.convs4(net).transpose(2, 1).contiguous()

        # net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        net = net.view(B, N, self.part_num)  # [B, N, 50]

        # ret = self.cls_head_finetune(feature)

        return net


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    @staticmethod
    def cal_loss(pred, gold, smoothing=False):
        """Calculate cross entropy loss, apply label smoothing if needed."""

        gold = gold.contiguous().view(-1)

        if smoothing:
            eps = 0.2
            n_class = pred.size()[1]
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()  # ~ F.nll_loss(log_prb, gold)
        else:
            loss = F.cross_entropy(pred, gold, reduction='mean')

        return loss

    def forward(self, pred, target):

        return self.cal_loss(pred, target, smoothing=False)


