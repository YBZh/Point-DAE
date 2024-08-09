#!/bin/bash

## pre-training and linear-probing evaluation with non-Transformer backbone.

LanguageArray=(
# "pretrain_PointCAE_clean" ## No_corruptions
# ###add noise
# "pretrain_PointCAE_add_global"
# "pretrain_PointCAE_add_local"
# "pretrain_PointCAE_jitter"
# # drop points
# "pretrain_PointCAE_dropout_global"
# "pretrain_PointCAE_dropout_local"
# "pretrain_PointCAE_dropout_patch"
# "pretrain_PointCAE_nonuniform_density"  ## Scan
# ##
# "pretrain_PointCAE_rotate_z"
# "pretrain_PointCAE_rotate"
# "pretrain_PointCAE_reflection"
# "pretrain_PointCAE_scale_nonorm"
# "pretrain_PointCAE_shear"
# "pretrain_PointCAE_translate"
# "pretrain_PointCAE_affine_r3"  ## Affine
# # affine transformation combination
# "pretrain_PointCAE_affine_r3_dropout_local"
"pretrain_PointCAE_affine_r3_dropout_local_4xlonger"
# "pretrain_PointCAE_affine_r3_dropout_local_10xlonger"  ### longer training for better results.
# "pretrain_PointCAE_affine_r3_dropout_patch"
)

# !!!!!!!!!!! IF you want to do testing only, please update the "./experiments/${YAML}${ModelName_method}/cfgs/tslog/ckpt-last.pth" to the ckpt-last.pth downloaded from:
# https://drive.google.com/drive/folders/1Nl3g-_HqHJ3J-Cjdx99ZmM29JXnJJZqZ?usp=sharing  , which provide the checkpoint for 'pretrain_PointCAE_affine_r3_dropout_local_4xlonger' setup.


for random in $(seq 1 1); do
  for YAML in ${LanguageArray[*]}; do
    ModelName_method=Point_CAE_DGCNN_FCOnly
    total_bs=256
    ## pretraining
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --config cfgs/${YAML}.yaml --exp_name log --model_name ${ModelName_method} --total_bs ${total_bs} --num_workers 8
    ModelName=DGCNN_feat
    total_bs=16
    ## training a SVM classifier on pre-extracted features.
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_hardest_svm_classification_clean.yaml \
    --finetune_model --svm_classification --exp_name ${YAML} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objbg_svm_classification_clean.yaml \
    --finetune_model --svm_classification --exp_name ${YAML} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
    # CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_scan_objonly_svm_classification_clean.yaml \
    # --finetune_model --svm_classification --exp_name ${YAML} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
#    CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_svm_classification.yaml \
#    --finetune_model --svm_classification --exp_name ${YAML} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
  done
done