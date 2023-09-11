#!/bin/bash

#YAML=pretrain_PointCAE_transformer_dropout_patch_affine_r3PointCAE_transformer_v2
#CUDA_VISIBLE_DEVICES=0 python main.py --optimizer_part all --log_dir ${YAML}_all --ckpts ../experiments/${YAML}/cfgs/log/ckpt-last.pth --root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --learning_rate 0.0002 --epoch 300
#
#YAML=pretrain_PointCAE_transformer_dropout_patch_rotatePointCAE_transformer_v2
#CUDA_VISIBLE_DEVICES=0 python main.py --optimizer_part all --log_dir ${YAML}_all --ckpts ../experiments/${YAML}/cfgs/log/ckpt-last.pth --root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --learning_rate 0.0002 --epoch 300

#YAML=pretrain_PointCAE_transformer_dropout_patch_scale_nonormPointCAE_transformer_v2
#CUDA_VISIBLE_DEVICES=1 python main.py --optimizer_part all --log_dir ${YAML}_all --ckpts ../experiments/${YAML}/cfgs/log/ckpt-last.pth --root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --learning_rate 0.0002 --epoch 300

#FolderNmae=new_exp_transformer
#ModelName_pretrain=PointCAE_transformer_fc_global_folding_local
#YAML=pretrain_PointCAE_transformer_dropout_patch_affine_r3_maskpatch_p0005_longer4x_svd
##YAML=pretrain_PointCAE_transformer_dropout_patch_affine_r3_maskpatch_p001_double
#CUDA_VISIBLE_DEVICES=7 python main.py --optimizer_part all --log_dir ${YAML}${ModelName_pretrain} \
#--ckpts ../${FolderNmae}/${YAML}${ModelName_pretrain}/cfgs/log1/ckpt-last.pth \
#--root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --learning_rate 0.0002 --epoch 300

FolderNmae=new_exp_transformer
ModelName_pretrain=PointCAE_transformer_fc_global_folding_local
YAML=pretrain_PointCAE_transformer_dropout_patch_affine_r3_maskpatch_p0005
#YAML=pretrain_PointCAE_transformer_dropout_patch_affine_r3_maskpatch_p001_double
CUDA_VISIBLE_DEVICES=7 python main.py --optimizer_part all --log_dir ${YAML}${ModelName_pretrain}log2 \
--ckpts ../${FolderNmae}/${YAML}${ModelName_pretrain}/cfgs/log2/ckpt-last.pth \
--root ../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/ --learning_rate 0.0002 --epoch 300