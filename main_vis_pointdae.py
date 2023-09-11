### prepare data for surfel cloud visualization.

# from tools import run_net
from tools import test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *

import cv2
import numpy as np


def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)

    base_model = builder.model_builder(config.model)
    # base_model.load_model_from_ckpt(args.ckpts)
    builder.load_model(base_model, args.ckpts, logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP
    if args.distributed:
        raise NotImplementedError()

    test(base_model, test_dataloader, args, config, logger=logger)


# visualization
def test(base_model, test_dataloader, args, config, logger=None):
    base_model.eval()  # set model to eval mode
    useful_cate = [
        # "02691156",  # plane
        "04379243",  # table
        # "03790512",  # motorbike
        # "03948459",  # pistol
        # "03642806",  # laptop
        # "03467517",  # guitar
        # "03261776",  # earphone
        # "03001627",  # chair
        # "02958343",  # car
        # "04090263",  # rifle
        # "03759954",  # microphone
    ]
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, corrupted_data, clean_data) in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            if taxonomy_ids[0] not in useful_cate:
                continue
            # if taxonomy_ids[0] == "02691156":
            #     a, b = 90, 135
            # elif taxonomy_ids[0] == "04379243":
            #     a, b = 30, 30
            # elif taxonomy_ids[0] == "03642806":
            #     a, b = 30, -45
            # elif taxonomy_ids[0] == "03467517":
            #     a, b = 0, 90
            # elif taxonomy_ids[0] == "03261776":
            #     a, b = 0, 75
            # elif taxonomy_ids[0] == "03001627":
            #     a, b = 30, -45
            # else:
            #     a, b = 0, 0
            print(idx)
            dataset_name = config.dataset.test._base_.NAME
            exp_name = args.exp_name
            if dataset_name == 'ShapeNet':
                corrupted_data = corrupted_data.cuda()
                clean_data = clean_data.cuda()
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            corrupted_pts, coarse, fine, pts = base_model(corrupted_data, clean_data, vis=True)

            output = {
                'input_point': corrupted_pts.cpu(),
                'coarse': coarse.cpu(),
                'fine': fine.cpu(),
                'gt': pts.cpu(),
            }

            data_path = f'./vis/{exp_name}/{taxonomy_ids[0]}_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            filename = 'pointdae.pth.tar'
            dir_save_file = os.path.join(data_path, filename)
            torch.save(output, dir_save_file)

            if idx > 234:
                break


def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger = logger)
    if args.model_name != 'none':
        config.model.NAME = args.model_name
    if args.total_bs != -1:
        config.total_bs = args.total_bs
    # ipdb.set_trace()
    if args.finetune_model or args.scratch_model:
        pass
    else: ## only for pre-training.
        if len(config.model['corrupt_type']) == 0:
            config.model['corrupt_type'] = config.dataset['train']['others']['corrupt_type']
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        config.dataset.val.others.bs = 1
        config.dataset.test.others.bs = 1
    else:
        config.dataset.train.others.bs = config.total_bs
        config.dataset.val.others.bs = 1
        config.dataset.test.others.bs = 1
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed + args.local_rank, deterministic=args.deterministic) # seed + rank, for augmentation
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    test_net(args, config)



    # # run
    # if args.test:
    #     test_net(args, config)
    # else:
    #     # run_net(args, config, train_writer, val_writer)
    #     raise NotImplementedError


if __name__ == '__main__':
    main()
