#!/bin/bash

LanguageArray=(
"pretrain_PointCAE_clean" ## No_corruptions
###add noise
"pretrain_PointCAE_add_global"
"pretrain_PointCAE_add_local"
"pretrain_PointCAE_jitter"
# drop points
"pretrain_PointCAE_dropout_global"
"pretrain_PointCAE_dropout_local"
"pretrain_PointCAE_dropout_patch"
"pretrain_PointCAE_nonuniform_density"  ## Scan
##
"pretrain_PointCAE_rotate_z"
"pretrain_PointCAE_rotate"
"pretrain_PointCAE_reflection"
"pretrain_PointCAE_scale_nonorm"
"pretrain_PointCAE_shear"
"pretrain_PointCAE_translate"
"pretrain_PointCAE_affine_r3"  ## Affine
# affine transformation combination
"pretrain_PointCAE_affine_r3_dropout_local"
"pretrain_PointCAE_affine_r3_dropout_patch"
)


for random in $(seq 1 3); do
  for YAML in ${LanguageArray[*]}; do
    ModelName_method=Point_CAE_DGCNN
    total_bs=256
    ### pretraining
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --config cfgs/${YAML}.yaml --exp_name log --model_name ${ModelName_method} --total_bs ${total_bs} --num_workers 32
    ModelName=DGCNN_feat
    total_bs=16
    ### training a SVM classifier on pre-extracted features.
    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_scan_hardest_svm_classification_clean.yaml \
    --finetune_model --svm_classification --exp_name ${YAML} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_scan_objbg_svm_classification_clean.yaml \
    --finetune_model --svm_classification --exp_name ${YAML} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_scan_objonly_svm_classification_clean.yaml \
    --finetune_model --svm_classification --exp_name ${YAML} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
#    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet_svm_classification.yaml \
#    --finetune_model --svm_classification --exp_name ${YAML} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
  done
done


###################################### test code for corruption robustness.
###### To test corruptions, we use DGCNN model only; only scale and translate for training; No NORM. smoothloss, no vote, no point resampling.
########################################### 这个是可以使用的。

#YAML=test
#ModelName=DGCNN
#total_bs=32
#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_transferring_features_1k_smooth_officialmodelnet.yaml --seed 0 \
#--finetune_model --exp_name ${YAML}  --model_name ${ModelName} --total_bs ${total_bs} \
#--ckpts ./experiments/pretrain_PointCAE_affine_r3_dropout_local_4xlongerPoint_CAE_DGCNN/cfgs/log/ckpt-last.pth
### 93.1

#YAML=test_corruption
#ModelName_method=Point_CAE_DGCNN
#total_bs=256
#ModelName=DGCNN
#total_bs=16
#CUDA_VISIBLE_DEVICES=0 python main.py --config cfgs/finetune_modelnet_transferring_features_1k_smooth_officialmodelnet.yaml \
#--test_corruption --exp_name ${YAML}  --model_name ${ModelName} --total_bs ${total_bs} \
#--ckpts ./experiments/finetune_modelnet_transferring_features_1k_smooth_officialmodelnetDGCNN/cfgs/test/ckpt-best.pth
#2022-12-31 07:25:26,110 - finetune_modelnet_transferring_features_1k_smooth_officialmodelnet - INFO - ckpts @ 54 epoch( performance = {'acc': tensor(93.1118)})
 #{'acc': 0.9307131280388979, 'avg_per_class_acc': 0.8978720930232559, 'corruption': 'clean'}
 #{'OA': 0.931, 'corruption': 'clean', 'level': 'Overall'}
 #{'acc': 0.9141004862236629, 'avg_per_class_acc': 0.8817209302325582, 'corruption': 'scale', 'level': 0}
 #{'acc': 0.916936790923825, 'avg_per_class_acc': 0.8794709302325583, 'corruption': 'scale', 'level': 1}
 #{'acc': 0.9088330632090762, 'avg_per_class_acc': 0.870936046511628, 'corruption': 'scale', 'level': 2}
 #{'acc': 0.9051863857374393, 'avg_per_class_acc': 0.8568953488372092, 'corruption': 'scale', 'level': 3}
 #{'acc': 0.9043760129659644, 'avg_per_class_acc': 0.8609360465116278, 'corruption': 'scale', 'level': 4}
 #{'CE': 0.957, 'OA': 0.91, 'RCE': 1.05, 'corruption': 'scale', 'level': 'Overall'}
 #{'acc': 0.9258508914100486, 'avg_per_class_acc': 0.8787093023255814, 'corruption': 'jitter', 'level': 0}
 #{'acc': 0.893030794165316, 'avg_per_class_acc': 0.8218139534883722, 'corruption': 'jitter', 'level': 1}
 #{'acc': 0.8160453808752026, 'avg_per_class_acc': 0.695046511627907, 'corruption': 'jitter', 'level': 2}
 #{'acc': 0.5935980551053485, 'avg_per_class_acc': 0.46534883720930226, 'corruption': 'jitter', 'level': 3}
 #{'acc': 0.37560777957860614, 'avg_per_class_acc': 0.28627325581395346, 'corruption': 'jitter', 'level': 4}
 #{'CE': 0.883, 'OA': 0.721, 'RCE': 0.868, 'corruption': 'jitter', 'level': 'Overall'}
 #{'acc': 0.9278768233387358, 'avg_per_class_acc': 0.8989651162790697, 'corruption': 'rotate', 'level': 0}
 #{'acc': 0.9193679092382496, 'avg_per_class_acc': 0.8893895348837209, 'corruption': 'rotate', 'level': 1}
 #{'acc': 0.8841166936790924, 'avg_per_class_acc': 0.8479883720930234, 'corruption': 'rotate', 'level': 2}
 #{'acc': 0.7787682333873582, 'avg_per_class_acc': 0.7400813953488372, 'corruption': 'rotate', 'level': 3}
 #{'acc': 0.6693679092382496, 'avg_per_class_acc': 0.6355872093023256, 'corruption': 'rotate', 'level': 4}
 #{'CE': 0.763, 'OA': 0.836, 'RCE': 0.674, 'corruption': 'rotate', 'level': 'Overall'}
 #{'acc': 0.9145056726094003, 'avg_per_class_acc': 0.8771918604651162, 'corruption': 'dropout_global', 'level': 0}
 #asdf{'acc': 0.8877633711507293, 'avg_per_class_acc': 0.8411046511627907, 'corruption': 'dropout_global', 'level': 1}
 #{'acc': 0.8338735818476499, 'avg_per_class_acc': 0.7731976744186048, 'corruption': 'dropout_global', 'level': 2}
 #{'acc': 0.6815235008103727, 'avg_per_class_acc': 0.5947848837209302, 'corruption': 'dropout_global', 'level': 3}
 #{'acc': 0.3521069692058347, 'avg_per_class_acc': 0.3166569767441861, 'corruption': 'dropout_global', 'level': 4}
 #{'CE': 1.073, 'OA': 0.734, 'RCE': 1.132, 'corruption': 'dropout_global', 'level': 'Overall'}
 #{'acc': 0.9124797406807131, 'avg_per_class_acc': 0.8715581395348838, 'corruption': 'dropout_local', 'level': 0}
 #{'acc': 0.8889789303079416, 'avg_per_class_acc': 0.8506627906976743, 'corruption': 'dropout_local', 'level': 1}
 #{'acc': 0.8622366288492707, 'avg_per_class_acc': 0.810796511627907, 'corruption': 'dropout_local', 'level': 2}
 #{'acc': 0.7893030794165316, 'avg_per_class_acc': 0.7169186046511629, 'corruption': 'dropout_local', 'level': 3}
 #{'acc': 0.6770664505672609, 'avg_per_class_acc': 0.6100813953488371, 'corruption': 'dropout_local', 'level': 4}
 #{'CE': 0.841, 'OA': 0.826, 'RCE': 0.789, 'corruption': 'dropout_local', 'level': 'Overall'}
 #{'acc': 0.8375202593192869, 'avg_per_class_acc': 0.8042616279069768, 'corruption': 'add_global', 'level': 0}
 #{'acc': 0.8091572123176661, 'avg_per_class_acc': 0.7690755813953489, 'corruption': 'add_global', 'level': 1}
 #{'acc': 0.796191247974068, 'avg_per_class_acc': 0.739279069767442, 'corruption': 'add_global', 'level': 2}
 #{'acc': 0.7860615883306321, 'avg_per_class_acc': 0.7251220930232558, 'corruption': 'add_global', 'level': 3}
 #{'acc': 0.7856564019448946, 'avg_per_class_acc': 0.715203488372093, 'corruption': 'add_global', 'level': 4}
 #{'CE': 0.668, 'OA': 0.803, 'RCE': 0.579, 'corruption': 'add_global', 'level': 'Overall'}
 #{'acc': 0.8630470016207455, 'avg_per_class_acc': 0.8040348837209302, 'corruption': 'add_local', 'level': 0}
 #{'acc': 0.8140194489465153, 'avg_per_class_acc': 0.7235406976744186, 'corruption': 'add_local', 'level': 1}
 #{'acc': 0.779578606158833, 'avg_per_class_acc': 0.6956337209302326, 'corruption': 'add_local', 'level': 2}
 #{'acc': 0.7317666126418152, 'avg_per_class_acc': 0.6317674418604652, 'corruption': 'add_local', 'level': 3}
 #{'acc': 0.6904376012965965, 'avg_per_class_acc': 0.5913197674418604, 'corruption': 'add_local', 'level': 4}
 #{'CE': 0.815, 'OA': 0.776, 'RCE': 0.771, 'corruption': 'add_local', 'level': 'Overall'}
 #{'RmCE': 0.838, 'mCE': 0.857, 'mOA': 0.801}


######################################## rotation testing.
#LanguageArray=(
## affine transformation combination
#"pretrain_PointCAE_affine_r3_dropout_local_4xlonger"
#)
#
#
#for random in $(seq 1 2); do
#  for YAML in ${LanguageArray[*]}; do
#    ModelName_method=Point_CAE_DGCNN
##    total_bs=256
##    CUDA_VISIBLE_DEVICES=0,1 python main.py --config cfgs/${YAML}.yaml --exp_name log --model_name ${ModelName_method} --total_bs ${total_bs} --num_workers 8
##    total_bs=256
##    ## pretraining
##    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --config cfgs/${YAML}.yaml --exp_name log --model_name ${ModelName_method} --total_bs ${total_bs} --num_workers 8
#    ModelName=DGCNN
#    total_bs=32
##    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet_rotation_z2so3_officialmodelnet.yaml \
##    --finetune_model --so3_rotation --exp_name scratch  --model_name ${ModelName} --total_bs ${total_bs}
##    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet_rotation_so32so3_officialmodelnet.yaml \
##    --finetune_model --so3_rotation --exp_name scratch  --model_name ${ModelName} --total_bs ${total_bs}
##    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet_rotation_z2z_officialmodelnet.yaml \
##    --finetune_model --so3_rotation --exp_name scratch  --model_name ${ModelName} --total_bs ${total_bs}
###    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet_rotation_z2so3_officialmodelnet.yaml \
###    --finetune_model --so3_rotation --exp_name ${YAML}${ModelName_method} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
###    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet_rotation_so32so3_officialmodelnet.yaml \
###    --finetune_model --so3_rotation --exp_name ${YAML}${ModelName_method} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
##    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet_rotation_z2z_officialmodelnet.yaml \
##    --finetune_model --so3_rotation --exp_name ${YAML}${ModelName_method} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
######
##    ### training a SVM classifier on pre-extracted features.
##    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_scan_hardest_svm_classification_clean_1k.yaml \
##    --finetune_model --svm_classification --exp_name ${YAML}${ModelName_method} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
##    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_scan_objbg_svm_classification_clean_1k.yaml \
##    --finetune_model --svm_classification --exp_name ${YAML}${ModelName_method} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
##    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_scan_objonly_svm_classification_clean_1k.yaml \
##    --finetune_model --svm_classification --exp_name ${YAML}${ModelName_method} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
##    CUDA_VISIBLE_DEVICES=1 python main.py --config cfgs/finetune_modelnet_svm_classification.yaml \
##    --finetune_model --svm_classification --exp_name ${YAML}${ModelName_method} --ckpts ./experiments/${YAML}${ModelName_method}/cfgs/log/ckpt-last.pth --model_name ${ModelName} --total_bs ${total_bs}
#  done
#done


