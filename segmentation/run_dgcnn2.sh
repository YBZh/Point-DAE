#!/bin/bash

#YAML=pretrain_PointCAE_affine_r3_dropout_local_4xlongerPoint_CAE_DGCNN_PartSeg
#CUDA_VISIBLE_DEVICES=2 python main.py --optimizer_part all --log_dir scratch_0 \
# --root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
#--learning_rate 0.0002 --epoch 300 --model dgcnn_partseg

YAML=pretrain_PointCAE_scale_nonorm_4xlongerPoint_CAE_PointNetNoT_PartSeg
CUDA_VISIBLE_DEVICES=2 python main.py --optimizer_part all --log_dir scratch_0 \
--root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ \
--learning_rate 0.0002 --epoch 300 --model pointnetnot_partseg