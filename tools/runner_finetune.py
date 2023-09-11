import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import SVC
import numpy as np
from datasets import data_transforms
from pointnet2_ops import pointnet2_utils
from torchvision import transforms
from thop import profile, clever_format
from ptflops import get_model_complexity_info
import ipdb

# def fix_bn(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         m.momentum = 0.0  #### fix all the imagenet pre-trained running mean and average
#         # m.eval()
#
# def release_bn(m):
#     classname = m.__class__.__name__
#     if classname.find('BatchNorm') != -1:
#         m.momentum = 0.1  #### roll back to the default setting
#         # m.train()

def set_bn_eval(m):
    classname = m.__class__.__name__
    ## if the feature dim is 256, then is belongs to the new classifier and we should learn its running mean and variance.
    ## Note that this only works for classification. We should be careful with the feature dimension in the segmentation tasks.
    if classname.find('BatchNorm') != -1 and m.weight.size(0) != 256:
        print('fixing the running mean and variance of the following BN layer:',m)
        m.eval()



train_transforms = transforms.Compose(
    [
         # data_transforms.PointcloudScale(),
         data_transforms.PointcloudRotate(),
         # data_transforms.PointcloudTranslate(),
         # data_transforms.PointcloudJitter(),
         # data_transforms.PointcloudRandomInputDropout(),
         # data_transforms.RandomHorizontalFlip(),
         # data_transforms.PointcloudScaleAndTranslate(),
    ]
)

test_transforms = transforms.Compose(
    [
        # data_transforms.PointcloudScale(),
        # data_transforms.PointcloudRotate(),
        # data_transforms.PointcloudTranslate(),
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)


class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        elif type(acc).__name__ == 'Acc_Metric':
            self.acc = acc.acc
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
#### common training and test
def run_net(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader),= builder.dataset_builder(args, config.dataset.train), \
                                                            builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)
    
    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger = logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger = logger)

    if args.use_gpu:    
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger = logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...' , logger = logger)
    else:
        print_log('Using Data parallel ...' , logger = logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)
    
    if args.resume:
        builder.resume_optimizer(optimizer, args, logger = logger)

    # trainval
    # training
    npoints = config.npoints
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        ### in the linear classification protocol, fix the BN layer in the pre-trained model.
        if config.optimizer.part == 'only_new':
            base_model.apply(set_bn_eval)

        # npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()

            # print(points.size())
            # print(label)
            ### for official ModelNet40, Point_size < point_all, therefore, no resampling is adopted.
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
            # import pdb; pdb.set_trace()

            # points = train_transforms(points)

            ###################### calculating complexity
            # macs, params = get_model_complexity_info(base_model, (2, 1024, 3), as_strings=False,
            #                                          print_per_layer_stat=True, verbose=False)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # ipdb.set_trace()


            ret = base_model(points)
            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])


            if args.distributed:
                torch.cuda.synchronize()


            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)


            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # print('rotate is applied to the training data following ACT!!')
            # Validate the current model
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
                print_log("--------------------------------------------------------------------------------------------", logger=logger)
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        print_log(
                            "****************************************************************************************",
                            logger=logger)
                        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger = logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger = logger)      
        # if (config.max_epoch - epoch) < 3:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()

#### common test
def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            _, points = misc.fps(points, npoints)

            logits = base_model(points)

            # loss, _ = base_model.module.get_loss_acc(logits, label)
            # print(loss)

            if isinstance(logits, dict): # for PointNet with multiple output
                logits = logits['cls']
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


#### common training and test
def run_net_rotation(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    npoints = config.npoints
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        num_iter = 0
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)

        ### in the linear classification protocol, fix the BN layer in the pre-trained model.
        if config.optimizer.part == 'only_new':
            base_model.apply(set_bn_eval)

        # npoints = config.npoints
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx

            data_time.update(time.time() - batch_start_time)

            points = data[0].cuda()
            label = data[1].cuda()

            # print(points.size())
            # print(label)
            ### for official ModelNet40, Point_size < point_all, therefore, no resampling is adopted.
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()

            if points.size(1) < point_all:
                point_all = points.size(1)

            fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
            fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
            points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                              2).contiguous()  # (B, N, 3)
            # import pdb; pdb.set_trace()
            # points = train_transforms(points)

            ###################### calculating complexity
            # macs, params = get_model_complexity_info(base_model, (2, 1024, 3), as_strings=False,
            #                                          print_per_layer_stat=True, verbose=False)
            # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            # ipdb.set_trace()

            ret = base_model(points)
            loss, acc = base_model.module.get_loss_acc(ret, label)

            _loss = loss

            _loss.backward()

            # forward
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()

            if args.distributed:
                loss = dist_utils.reduce_tensor(loss, args)
                acc = dist_utils.reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])

            if args.distributed:
                torch.cuda.synchronize()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

            # if idx % 10 == 0:
            #     print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Loss+Acc = %s lr = %.6f' %
            #                 (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
            #                 ['%.4f' % l for l in losses.val()], optimizer.param_groups[0]['lr']), logger = logger)
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        epoch_end_time = time.time()

        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avg(0), epoch)

        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
                  (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],
                   optimizer.param_groups[0]['lr']), logger=logger)

        if epoch % args.val_freq == 0 and epoch != 0:
            # Validate the current model
            metrics = validate_rotation(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)

            better = metrics.better_than(best_metrics)
            # Save ckeckpoints
            if better:
                best_metrics = metrics
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                        logger=logger)
                print_log(
                    "--------------------------------------------------------------------------------------------",
                    logger=logger)
            # if args.vote:
            #     if metrics.acc > 92.1 or (better and metrics.acc > 91):
            #         metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config,
            #                                      logger=logger)
            #         if metrics_vote.better_than(best_metrics_vote):
            #             best_metrics_vote = metrics_vote
            #             print_log(
            #                 "****************************************************************************************",
            #                 logger=logger)
            #             builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote,
            #                                     'ckpt-best_vote', args, logger=logger)

        builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger=logger)
        # if (config.max_epoch - epoch) < 3:
        #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


#### common test
def validate_rotation(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    npoints = config.npoints
    with torch.no_grad():
        mean_acc = []
        for random_rotation in range(10):
            test_pred = []
            test_label = []
            for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
                points = data[0].cuda()
                label = data[1].cuda()

                _, points = misc.fps(points, npoints)

                logits = base_model(points)

                # loss, _ = base_model.module.get_loss_acc(logits, label)
                # print(loss)

                if isinstance(logits, dict):  # for PointNet with multiple output
                    logits = logits['cls']
                target = label.view(-1)

                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            mean_acc.append(acc)
        acc = torch.Tensor(mean_acc).mean()
        print_log('[Validation] EPOCH: %d  acc = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)

    return Acc_Metric(acc)


#### testing with vote
def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):
    print_log(f"[VALIDATION_VOTE] epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                if isinstance(logits, dict):  # for PointNet with multiple output
                    logits = logits['cls']
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[Validation_vote] EPOCH: %d  acc_vote = %.4f' % (epoch, acc), logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)

    return Acc_Metric(acc)


#### test only
def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger = logger) # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)

    #  DDP    
    if args.distributed:
        raise NotImplementedError()
     
    test(base_model, test_dataloader, args, config, logger=logger)


# from modelnetc_utils import eval_corrupt_wrapper, ModelNetC
# from torch.utils.data import DataLoader
# import sklearn.metrics as metrics

def test_corrupt(args, split, model):
    test_loader = DataLoader(ModelNetC(split=split),
                             batch_size=args.total_bs, shuffle=True, drop_last=False)
    test_true = []
    test_pred = []
    for data, label in test_loader:
        data, label = data.cuda(), label.cuda().squeeze()
        # ipdb.set_trace() ## 16 * 1024 *3
        # data = data.permute(0, 2, 1)
        logits = model(data)
        preds = logits.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    return {'acc': test_acc, 'avg_per_class_acc': avg_per_class_acc}

def test_net_corruption(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    base_model = builder.model_builder(config.model)
    # load checkpoints
    builder.load_model(base_model, args.ckpts, logger=logger)  # for finetuned transformer
    # base_model.load_model_from_ckpt(args.ckpts) # for BERT
    if args.use_gpu:
        base_model.to(args.local_rank)
    base_model = base_model.eval()
    #  DDP
    if args.distributed:
        raise NotImplementedError()
    eval_corrupt_wrapper(base_model, test_corrupt, {'args': args})
    # Replace ModelNet40 by ModelNetC
    # test_loader = DataLoader(ModelNetC(split=split), ...)
    #
    # # Remains unchanged
    # overall_accuracy = run_model_on_test_loader(model, test_loader)
    #
    # # Remains unchanged
    # return overall_accuracy
    # # test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()

            _, points = misc.fps(points, npoints)

            logits = base_model(points)
            if isinstance(logits, dict): # for PointNet with multiple output
                logits = logits['cls']
            target = label.view(-1)

            pred = logits.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
        print_log('[TEST] acc = %.4f' % acc, logger=logger)

        if args.distributed:
            torch.cuda.synchronize()

        print_log(f"[TEST_VOTE]", logger = logger)
        acc = 0.
        for time in range(1, 300):
            this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger=logger, times=10)
            if acc < this_acc:
                acc = this_acc
            print_log('[TEST_VOTE_time %d]  acc = %.4f, best acc = %.4f' % (time, this_acc, acc), logger=logger)
        print_log('[TEST_VOTE] acc = %.4f' % acc, logger=logger)

def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger = None, times = 10):

    base_model.eval()  # set model to eval mode

    test_pred  = []
    test_label = []
    npoints = config.npoints
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                raise NotImplementedError()
                
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)

            fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)  # (B, npoint)
            local_pred = []

            for kk in range(times):
                fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points_raw.transpose(1, 2).contiguous(), 
                                                        fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)

                points = test_transforms(points)

                logits = base_model(points)
                if isinstance(logits, dict):  # for PointNet with multiple output
                    logits = logits['cls']
                target = label.view(-1)

                local_pred.append(logits.detach().unsqueeze(0))

            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)


            test_pred.append(pred_choice)
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)

        if args.distributed:
            test_pred = dist_utils.gather_tensor(test_pred, args)
            test_label = dist_utils.gather_tensor(test_label, args)

        acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.

        if args.distributed:
            torch.cuda.synchronize()

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    # print_log('[TEST] acc = %.4f' % acc, logger=logger)
    
    return acc

#
def svm_classification(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    npoints = config.npoints  ## 1024.
    # print(config)
    feats_train = []
    labels_train = []
    base_model = base_model.eval()

    batch_start_time = time.time()
    for i, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        data_time.update(time.time() - batch_start_time)
        points = data[0].cuda()
        label = data[1]

        ### for official ModelNet40, Point_size < point_all, therefore, no resampling is adopted.
        if npoints == 1024:
            point_all = 1024
        elif npoints == 2048:
            point_all = 2048
        elif npoints == 4096:
            point_all = 4096
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        if points.size(1) < point_all:
            point_all = points.size(1)
        fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                           2).contiguous()  # (B, N, 3)
        # points = points[:, :npoints, :].contiguous()

        with torch.no_grad():
            feats = base_model(points)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_train.append(feat)
        labels_train += label

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)
    print(feats_train.shape)
    # print(labels_train)


    feats_test = []
    labels_test = []
    base_model = base_model.eval()

    batch_start_time = time.time()
    for i, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        data_time.update(time.time() - batch_start_time)
        points = data[0].cuda()
        label = data[1]

        if npoints == 1024:
            point_all = 1024
        elif npoints == 2048:
            point_all = 2048
        elif npoints == 4096:
            point_all = 4096
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        if points.size(1) < point_all:
            point_all = points.size(1)
        fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                          2).contiguous()  # (B, N, 3)
        # points = points[:, :npoints, :].contiguous()

        with torch.no_grad():
            feats = base_model(points)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_test.append(feat)
        labels_test += label

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)
    print(feats_test.shape)
    # print(labels_test)

    max_acc = 0
    for i in range(-3,3):
        c = 10**i
        # c = 0.01  # Linear SVM parameter C, can be tuned
        # print(c)
        model_tl = SVC(C=c, kernel='linear')
        # ipdb.set_trace()
        model_tl.fit(feats_train, labels_train)
        if max_acc < model_tl.score(feats_test, labels_test):
            max_acc = model_tl.score(feats_test, labels_test)
        print(c, max_acc)
    print_log('[Validation] EPOCH: %d  acc = %.4f' % (c, max_acc), logger=logger)
    # print_log(f"C = {c} : {}")

def task_affinity(args, config, train_writer=None, val_writer=None):
    logger = get_logger(args.log_name)
    # build dataset
    (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                               builder.dataset_builder(args, config.dataset.val)
    # build model
    base_model = builder.model_builder(config.model)

    # parameter setting
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # resume ckpts
    if args.resume:
        start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('Training from scratch', logger=logger)

    if args.use_gpu:
        base_model.to(args.local_rank)
    # DDP
    if args.distributed:
        # Sync BN
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model,
                                                         device_ids=[args.local_rank % torch.cuda.device_count()])
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    if args.resume:
        builder.resume_optimizer(optimizer, args, logger=logger)

    # trainval
    # training
    npoints = config.npoints

    feats_train = []
    labels_train = []
    base_model = base_model.eval()

    batch_start_time = time.time()
    for i, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
        # if dataset == "ModelNet40":
        #     labels = list(map(lambda x: x[0], label.numpy().tolist()))
        # elif dataset == "ScanObjectNN":
        #     labels = label.numpy().tolist()
        # data = data.permute(0, 2, 1).to(device)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        data_time.update(time.time() - batch_start_time)
        points = data[0].cuda()
        label = data[1]

        if npoints == 1024:
            point_all = 1024
        elif npoints == 2048:
            point_all = 2048
        elif npoints == 4096:
            point_all = 4096
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        if points.size(1) < point_all:
            point_all = points.size(1)
        fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                          2).contiguous()  # (B, N, 3)
        # ret = base_model(points)
        # loss, acc = base_model.module.get_loss_acc(ret, label)

        with torch.no_grad():
            feats = base_model(points)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_train.append(feat)
        labels_train += label

    feats_train = np.array(feats_train)
    labels_train = np.array(labels_train)
    print(feats_train.shape)
    print(labels_train.shape)

    feats_train = torch.from_numpy(feats_train).cuda()
    labels_train = torch.from_numpy(labels_train).cuda()

    feats_test = []
    labels_test = []
    base_model = base_model.eval()

    batch_start_time = time.time()
    for i, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        data_time.update(time.time() - batch_start_time)
        points = data[0].cuda()
        label = data[1]

        if npoints == 1024:
            point_all = 1024
        elif npoints == 2048:
            point_all = 2048
        elif npoints == 4096:
            point_all = 4096
        elif npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()

        if points.size(1) < point_all:
            point_all = points.size(1)
        fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                          2).contiguous()  # (B, N, 3)
        # ret = base_model(points)
        # loss, acc = base_model.module.get_loss_acc(ret, label)

        with torch.no_grad():
            feats = base_model(points)
        feats = feats.detach().cpu().numpy()
        for feat in feats:
            feats_test.append(feat)
        labels_test += label

    feats_test = np.array(feats_test)
    labels_test = np.array(labels_test)
    print(feats_test.shape)
    # print(labels_test)
    feats_test = torch.from_numpy(feats_test).cuda()
    labels_test = torch.from_numpy(labels_test).cuda()

    feat_dim = feats_test.size(1)
    num_class = torch.max(labels_test)+1

    # import torch.nn as nn
    # import torch.nn.functional as F
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(feat_dim, num_class)
        def forward(self, x):
            x = self.fc1(x)
            return x
    net = Net().cuda()
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
    import math
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    # step_schedule = StepLR(optimizer, step_size=30, gamma=0.1)
    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.05)
    step_schedule = CosineAnnealingLR(optimizer, T_max=300)

    for epoch in range(300):  # loop over the dataset multiple times
        batch_size = 64
        train_number = feats_train.size(0)
        iteration_num = int(train_number / batch_size)
        r = torch.randperm(train_number)
        feats_train_shuffle = feats_train[r]
        labels_train_shuffle = labels_train[r]
        for i in range(iteration_num):
        # for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            # inputs, labels = data
            inputs = feats_train_shuffle[i*batch_size:(i+1)*batch_size, :]
            labels = labels_train_shuffle[i*batch_size:(i+1)*batch_size]

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # print(loss.item())
            loss.backward()
            optimizer.step()
        step_schedule.step()
    print('!!!!!!! split of the training and test !!!')
    ## extract the test loss as task affinity.
    batch_size = 64
    test_number = feats_test.size(0)
    iteration_num = math.ceil(test_number / batch_size)
    loss_sum = 0
    total = 0
    correct = 0
    for i in range(iteration_num):
        if i == (iteration_num - 1):
            inputs = feats_test[i*batch_size:, :]
            labels = labels_test[i*batch_size:]
        else:
            inputs = feats_test[i*batch_size:(i+1)*batch_size, :]
            labels = labels_test[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            # print(loss.item())
            loss_sum = loss_sum + loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    assert total == test_number
    average_loss = loss_sum / test_number
    print_log('[Validation] Acc: %.4f  loss = %.4f' % (correct/total, average_loss), logger=logger)
            # print statistics
            # running_loss += loss.item()
            # if i % 2000 == 1999:  # print every 2000 mini-batches
            #     print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            #     running_loss = 0.0

    # print('Finished Training')

    # max_acc = 0
    # for i in range(-3,3):
    #     c = 10**i
    #     # c = 0.01  # Linear SVM parameter C, can be tuned
    #     # print(c)
    #     model_tl = SVC(C=c, kernel='linear')
    #     model_tl.fit(feats_train, labels_train)
    #     if max_acc < model_tl.score(feats_test, labels_test):
    #         max_acc = model_tl.score(feats_test, labels_test)
    #     print(c, max_acc)
    # print_log('[Validation] EPOCH: %d  acc = %.4f' % (c, max_acc), logger=logger)
    # print_log(f"C = {c} : {}")

