#  Copyright (c) 2020. Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/AnTao97/dgcnn.pytorch/blob/master/model.py
#  Ref: https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/part_seg/train_multi_gpu.py

import pdb, torch, torch.nn as nn, torch.nn.functional as F
from models.dgcnn_util import dgcnn_encoder_partseg
from logger import get_missing_parameters_message, get_unexpected_parameters_message
import ipdb

class get_model(nn.Module):
    def __init__(self, cls_dim):
        super(get_model, self).__init__()
        self.dgcnn_encoder = dgcnn_encoder_partseg(channel=3)

        # self.k = args.k
        self.part_num = cls_dim
        # self.transform_net = T_Net(channel=num_channel)
        # self.bn1 = nn.BatchNorm2d(64)
        # self.bn2 = nn.BatchNorm2d(64)
        # self.bn3 = nn.BatchNorm2d(64)
        # self.bn4 = nn.BatchNorm2d(64)
        # self.bn5 = nn.BatchNorm2d(64)
        # self.bn6 = nn.BatchNorm1d(args.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        # self.conv1 = nn.Sequential(nn.Conv2d(num_channel*2, 64, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
        #                            self.bn3,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
        #                            self.bn4,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
        #                            self.bn5,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv6 = nn.Sequential(nn.Conv1d(192, args.emb_dims, kernel_size=1, bias=False),
        #                            self.bn6,
        #                            nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.part_num, kernel_size=1, bias=False)

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

    def forward(self, x, l):
        B, D, N = x.size()

        # x0 = get_graph_feature(x, k=self.k)
        # t = self.transform_net(x0)
        # x = x.transpose(2, 1)
        # if D > 3:
        #     x, feature = x.split(3, dim=2)
        # x = torch.bmm(x, t)
        # if D > 3:
        #     x = torch.cat([x, feature], dim=2)
        # x = x.transpose(2, 1)

        # x = get_graph_feature(x, k=self.k)
        # x = self.conv1(x)
        # x = self.conv2(x)
        # x1 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = get_graph_feature(x1, k=self.k)
        # x = self.conv3(x)
        # x = self.conv4(x)
        # x2 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = get_graph_feature(x2, k=self.k)
        # x = self.conv5(x)
        # x3 = x.max(dim=-1, keepdim=False)[0]
        #
        # x = torch.cat((x1, x2, x3), dim=1)
        #
        # x = self.conv6(x)
        # x = x.max(dim=-1, keepdim=True)[0]
        x1, x2, x3, x = self.dgcnn_encoder(x)
        l = l.view(B, -1, 1)
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, N)

        x = torch.cat((x, x1, x2, x3), dim=1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x.permute(0, 2, 1).contiguous()


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    @staticmethod
    def cal_loss(pred, gold, smoothing=True):
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


