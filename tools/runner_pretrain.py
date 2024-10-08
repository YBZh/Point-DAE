import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
import math
# from torchvision import transforms
# from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
## for complexity
from thop import profile, clever_format
from ptflops import get_model_complexity_info

import ipdb



class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]

def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    (_, extra_train_dataloader)  = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get('extra_train') else (None, None)
    # build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)

    # from IPython import embed; embed()
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    print(args.start_ckpts)
    # resume ckpts
    if args.resume:
        print('start the resume process')
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        builder.load_model(base_model, args.start_ckpts, logger = logger)


    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # if args.resume:
    #     builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    base_model.zero_grad()
    # torch.autograd.set_detect_anomaly(True)
    reset_optimizer = True
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])
        losses_normal = AverageMeter(['Loss'])

        num_iter = 0
        if config.loss_type == 'xyznormal_gradual':
            gradual_weight = float(epoch) / float(config.max_epoch)
        elif config.loss_type == 'xyznormal_warm':
            if float(epoch) / float(config.max_epoch) < 1.0/3.0:
                gradual_weight = float(epoch) / float(config.max_epoch) * 3
            else:
                gradual_weight = 1.0
            # gradual_weight = float(epoch) / float(config.max_epoch)
        else:
            gradual_weight = 0
        print(gradual_weight)
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        for idx, (taxonomy_ids, model_ids, corrupted_data, clean_data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME

            points = corrupted_data.cuda()
            reconstructed_gt = clean_data.cuda()
            # if dataset_name == 'ShapeNet':
            #     points = corrupted_data.cuda()
            #     reconstructed_gt = clean_data.cuda()
            # elif dataset_name == 'ScanNet':
            #     points = corrupted_data.cuda()
            #     reconstructed_gt = clean_data.cuda()
            # # elif dataset_name == 'ModelNet':
            #     # points = corrupted_data[0].cuda()
            #     # points = misc.fps(points, npoints)
            # else:
            #     raise NotImplementedError(f'Train phase do not support {dataset_name}')

            data_time.update(time.time() - batch_start_time)
            assert clean_data.size(1) == npoints
            assert points.size(1) <= npoints
            # points = train_transforms(points)
            loss_xyz, loss_normal = base_model(points, reconstructed_gt)
            # data_time.update(time.time() - batch_start_time)

            ###################### calculating complexity
            # macs, params = get_model_complexity_info(base_model, (2, 1024, 6), as_strings=False,
            #                                          print_per_layer_stat=True, verbose=False)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # ipdb.set_trace()

            if config.loss_type == 'xyz':
                loss = loss_xyz
            elif config.loss_type == 'normal':
                loss = float(config.normal_weight) * loss_normal
            elif config.loss_type == 'xyznormal':
                loss = loss_xyz + float(config.normal_weight) * loss_normal
            elif config.loss_type == 'xyznormal_gradual':
                loss = loss_xyz + float(config.normal_weight) * loss_normal * gradual_weight
            elif config.loss_type == 'xyznormal_warm':
                loss = loss_xyz + float(config.normal_weight) * loss_normal * gradual_weight
            # elif config.loss_type == 'xyznormal_xyzfirst':
            #     if epoch < 300:
            #         loss = loss_xyz
            #     else:
            #         loss = loss_xyz + float(config.normal_weight) * loss_normal
            # elif config.loss_type == 'xyznormal_xyzfirst_gradual':
            #     if epoch < 300:
            #         loss = loss_xyz
            #     else:
            #         if reset_optimizer:
            #             reset_optimizer = False
            #             optimizer, scheduler = builder.build_opti_sche(base_model, config)
            #         gradual_weight = float(epoch-299) / float(config.max_epoch-299)
            #         loss = loss_xyz + float(config.normal_weight) * loss_normal * gradual_weight
            else:
                raise NotImplementedError
            try:
                loss.backward()
                # print("Using one GPU")
            except:
                loss = loss.mean()
                loss.backward()
                # print("Using multi GPUs")

            # forward
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                losses.update([loss_xyz.item()*1000])
                losses_normal.update([loss_normal.item()*1000])
            else:
                try:
                    losses.update([loss_xyz.mean().item() * 1000])
                    losses_normal.update([loss_normal.mean().item() * 1000])
                    # print("Using one GPU")
                except:
                    losses.update([loss_xyz.mean().item() * 1000])
                    losses_normal.update([loss_normal.mean().item() * 1000])



            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()


            if idx % 50 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Lossxyz = %s Lossnormal = %s lr = %.6f' %
                            (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                            ['%.4f' % l for l in losses.val()], ['%.4f' % l for l in losses_normal.val()], optimizer.param_groups[0]['lr']), logger = logger)
            # dict(base_model.module.named_parameters())['loss_weight'] =  dict(base_model.module.named_parameters())['loss_weight'] / dict(base_model.module.named_parameters())['loss_weight'].sum()
            # ipdb.set_trace()
            # print(0.5 / torch.pow(torch.exp(dict(base_model.module.named_parameters())['loss_weight']),2))
            # print(base_model.module.get_loss_value())
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s Lossnormal = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], ['%.4f' % l for l in losses_normal.avg()],
             optimizer.param_groups[0]['lr']), logger = logger)
        if 'loss_weight' in dict(base_model.module.named_parameters()).keys():
            ### record loss weight & loss value.
            # loss_weight_current = 0.5 / torch.pow(torch.exp(dict(base_model.module.named_parameters())['loss_weight']),2)
            loss_current, loss_weight_current = base_model.module.get_loss_value_weight()

            if  loss_current.size(0) == 6:
                # loss_weight = dict(base_model.module.named_parameters())['loss_weight'] / dict(base_model.module.named_parameters())['loss_weight'].sum()
                print_log('[Training] EPOCH: %d loss_weight = %.3f, %.3f, %.3f, %.3f, %.3f, %.3f' %
                    (epoch, loss_weight_current[0].item(), loss_weight_current[1].item(),loss_weight_current[2].item(),\
                     loss_weight_current[3].item(),loss_weight_current[4].item(),loss_weight_current[5].item()), logger = logger)
                print_log('[Training] EPOCH: %d loss_value = %.3f, %.3f, %.3f, %.3f, %.3f, %.3f' %
                    (epoch, loss_current[0].item(), loss_current[1].item(),loss_current[2].item(),\
                     loss_current[3].item(),loss_current[4].item(),loss_current[5].item()), logger = logger)
            else:
                # loss_weight = dict(base_model.module.named_parameters())['loss_weight'] / dict(base_model.module.named_parameters())['loss_weight'].sum()
                print_log('[Training] EPOCH: %d loss_weight = %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f' %
                    (epoch, loss_weight_current[0].item(), loss_weight_current[1].item(),loss_weight_current[2].item(),\
                     loss_weight_current[3].item(),loss_weight_current[4].item(),loss_weight_current[5].item(),loss_weight_current[6].item(),loss_weight_current[7].item()), logger = logger)
                print_log('[Training] EPOCH: %d loss_value = %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f, %.3f' %
                    (epoch, loss_current[0].item(), loss_current[1].item(),loss_current[2].item(),\
                     loss_current[3].item(),loss_current[4].item(),loss_current[5].item(),loss_current[6].item(),loss_current[7].item()), logger = logger)

        if epoch % args.val_freq == 0: ## and epoch != 0:
            # Validate the current model
            metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=logger)

            # Save ckeckpoints
            if metrics.better_than(best_metrics):
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)
        # if epoch % math.ceil(config.max_epoch * 0.05) == 0 and epoch >= (config.max_epoch * 0.85):
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args,
        #                             logger=logger)
        # if (config.max_epoch - epoch) < 10:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger = None):
    print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_features = []
    test_label = []

    train_features = []
    train_label = []
    npoints = config.dataset.extra_train.others.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(extra_train_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            _, points = misc.fps(points, npoints)  ## points contain 1024 points, thus this command not work.

            assert points.size(1) == npoints
            feature = base_model(points, points, vis=False, return_feat = True)
            target = label.view(-1)
            # ipdb.set_trace()
            train_features.append(feature.detach())
            train_label.append(target.detach())

        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            _, points = misc.fps(points, npoints)
            assert points.size(1) == npoints
            feature = base_model(points, points, vis=False, return_feat=True)
            target = label.view(-1)

            test_features.append(feature.detach())
            test_label.append(target.detach())


        train_features = torch.cat(train_features, dim=0)
        train_label = torch.cat(train_label, dim=0)
        test_features = torch.cat(test_features, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            train_features = dist_utils.gather_tensor(train_features, args)
            train_label = dist_utils.gather_tensor(train_label, args)
            test_features = dist_utils.gather_tensor(test_features, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        svm_acc = evaluate_svm(train_features.data.cpu().numpy(), train_label.data.cpu().numpy(), test_features.data.cpu().numpy(), test_label.data.cpu().numpy())

        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch,svm_acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', svm_acc, epoch)

    return Acc_Metric(svm_acc)


def test_net():
    pass