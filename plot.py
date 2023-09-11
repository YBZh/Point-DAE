import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import random
from skimage.exposure import match_histograms
import matplotlib.gridspec as gridspec
import torch
from scipy.interpolate import make_interp_spline
from scipy.stats.stats import pearsonr
import ipdb

project_root = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/PointCloud/Point-DAE-all/Point-MA2E_iccv2023/images/'

# project_root = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/PointCloud/MaskSurf_all/iccv2023AuthorKit/images/'


# ## rotate, \alpha
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [0, 0.2, 0.4, 0.6, 0.8, 1]
alpha = np.array(alpha)
acc = [77.1, 80.9, 81.8, 82.3, 82.8, 82.8]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)

# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\alpha$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'rotate_alpha.pdf'
f.savefig(image_name)

# ## translate, tau
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [0, 0.1, 0.3, 0.5, 0.8]
alpha = np.array(alpha)
acc = [77.1, 79.7, 80.8, 81.2, 81.2]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\tau_t$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'tranlate_tau.pdf'
f.savefig(image_name)

# ## shear, eta
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [0, 0.1, 0.3, 0.5, 0.8, 1.0, 2.0]
alpha = np.array(alpha)
acc = [77.1, 78.6, 79.3, 80.5, 81.2, 81.4, 81.3]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\eta$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'shear_eta.pdf'
f.savefig(image_name)


# ## scale, eta
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [1.0, 1.5, 2.0, 4.0, 10.0]
alpha = np.array(alpha)
acc = [77.1, 80.6, 82.5, 81.6, 80.8]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\eta$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'scale_eta.pdf'
f.savefig(image_name)



# ## drop-local, dgcnn, alpha
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
alpha = np.array(alpha)
acc = [77.1, 78.7, 81.3, 81.1, 81.1, 80.0]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\alpha$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'drop_local_alpha.pdf'
f.savefig(image_name)


# ## drop-local, dgcnn, kappa
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [1, 2, 3, 5, 8]
alpha = np.array(alpha)
acc = [80.7, 81.1, 81.3, 81.3, 81.0]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\kappa$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'drop_local_kappa.pdf'
f.savefig(image_name)

# ## add-global, dgcnn, alpha
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
alpha = np.array(alpha)
acc = [77.1, 77.8, 77.5, 77.1, 76.8, 74.5]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\alpha$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'add_global_alpha.pdf'
f.savefig(image_name)

# ## add-local, dgcnn, alpha
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
alpha = np.array(alpha)
acc = [77.1, 77.5, 77.7, 77.1, 76.5, 75.5]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\alpha$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'add_local_alpha.pdf'
f.savefig(image_name)

# ## add-local, dgcnn, kappa
figsize = 18,10
f, ax = plt.subplots(figsize=figsize)
alpha = [1, 2, 3, 5, 8]
alpha = np.array(alpha)
acc = [77.2, 77.4, 77.7, 77.7, 77.6]
ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
# ax.legend(loc='best', fontsize=60)
plt.tick_params(labelsize=40)
ax.set_xlabel(r'Values of $\kappa$', fontsize=35)
ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
image_name = project_root + 'add_local_kappa.pdf'
f.savefig(image_name)

# #####################
# corruptions = ['No-Corruption',  'Masking Only', 'Affine Only', 'Masking + Affine']
#
# name='dgcnn_objbg_dataset_variants_mask_affine'
# random = [74.4, 76.1, 80.4,  81.0]  ## Add local 76.2 ???  Second clolum 74.9??
# svd    = [75.0, 79.0, 81.4,  82.4]  ## affine <--> affine + Drop Patch.
# man    = [77.4, 79.5, 84.4,  85.5]  ## affine <--> affine + Drop Patch.
# min_value = min(random)
# max_value = max(man)
# ######## NO supervised learning adopted here. Totally 17.
# x = np.arange(4)
# total_width, n = 0.7, 3
# width = total_width / n
# x = x - (total_width - width) / 3
# figsize = 18,5
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(random), fc='g', width=width, label ="Random-Pose")
# ax.bar(x + width, np.array(svd), fc='b', width=width, label ="SVD-Pose")
# ax.bar(x + 2 * width, np.array(man), fc='r', width=width, label ="Manual-Pose")
# ax.legend(fontsize=20)
# plt.tick_params(labelsize=20)
# ax.set_ylim([min_value*0.99, max_value*1.005])
# # ax.set_xlim([-0.5, 16.8])
# plt.xticks(x+2.2*width, corruptions, fontsize=20, rotation=0, ha='right', rotation_mode='anchor')
# ax.set_ylabel('Acc.(%)', fontsize=20)
# image_name = project_root + name + '.pdf'
# f.savefig(image_name)

# # ## downsampling, alpha
# figsize = 18,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = [0, 0.1, 0.3, 0.5, 0.7, 0.9]
# alpha = np.array(alpha)
#
# acc = [77.1, 79.3, 78.0, 77.1, 76.8, 71.9]
#
# ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
#
# # ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'Values of $\alpha$', fontsize=35)
# ax.set_ylabel('Acc.(%)', fontsize=40)
# # ax.set_ylim([70.0, 74.0])
# # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
# image_name = project_root + 'downsampling_alpha.pdf'
# f.savefig(image_name)


# # ## jitter, sigma
# figsize = 18,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = [0, 0.01, 0.03, 0.05, 0.07, 0.1]
# alpha = np.array(alpha)
#
# acc = [77.1, 78.1, 78.8, 80.7, 79.6, 77.3]
#
# ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
#
# # ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'Values of $\sigma$', fontsize=35)
# ax.set_ylabel('Acc.(%)', fontsize=40)
# # ax.set_ylim([70.0, 74.0])
# # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
# image_name = project_root + 'jitter_sigma.pdf'
# f.savefig(image_name)


# # ## hyper-parameter alpha, balancing global and local reconstructions
# figsize = 18,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
# alpha = np.array(alpha)
#
# acc = [81.3, 83.6, 84.62, 84.6, 83.9]
#
# ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
#
# # ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'Values of $\lambda$', fontsize=35)
# ax.set_ylabel('Acc.(%)', fontsize=40)
# # ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5], ['5e-4', '1e-3', '5e-3', '1e-2', '5e-2'])
# image_name = project_root + 'lambda.pdf'
# f.savefig(image_name)


# ################## Point-MA2E, affine corruption analyses ( to update.)
# corruptions = ['Affine', 'Rotate', 'Reflect', 'Scale', 'Shear', 'Translate']
#
# name='global_corruption_variants_dgcnn'
# DGCNN = [84.4, 82.8, 82.8, 82.5, 81.4, 81.2]
# Transformer    = [79.8, 78.5, 78.2, 77.6, 74.0, 72.9]
# min_value = min(DGCNN)
# max_value = max(DGCNN)
# ######## NO supervised learning adopted here. Totally 17.
# x = np.arange(6)
# total_width, n = 0.45, 1
# width = total_width / n
# x = x - (total_width - width) / 1
# figsize = 8,7
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(DGCNN), fc='y', width=width)
# # ax.bar(x + width, np.array(svd), fc='b', width=width, label ="SVD-Pose")
# # ax.bar(x + 2 * width, np.array(man), fc='r', width=width, label ="Manual-Pose")
# # ax.legend(fontsize=20)
# plt.tick_params(labelsize=25)
# ax.set_ylim([min_value*0.99, max_value*1.005])
# # ax.set_xlim([-0.5, 16.8])
# plt.xticks(x, corruptions, fontsize=22, rotation=15, ha='right', rotation_mode='anchor')
# ax.set_ylabel('Acc.(%)', fontsize=25)
# image_name = project_root + name + '.pdf'
# f.savefig(image_name)
# #
# name='global_corruption_variants_transformer'
# DGCNN = [84.4, 82.8, 82.8, 82.5, 81.4, 81.2]
# Transformer    = [79.8, 78.5, 78.2, 77.6, 74.0, 72.9]
# min_value = min(Transformer)
# max_value = max(Transformer)
# ######## NO supervised learning adopted here. Totally 17.
# x = np.arange(6)
# total_width, n = 0.45, 1
# width = total_width / n
# x = x - (total_width - width) / 1
# figsize = 8,7
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(Transformer), fc='y', width=width)
# # ax.bar(x + width, np.array(svd), fc='b', width=width, label ="SVD-Pose")
# # ax.bar(x + 2 * width, np.array(man), fc='r', width=width, label ="Manual-Pose")
# # ax.legend(fontsize=20)
# plt.tick_params(labelsize=25)
# ax.set_ylim([min_value*0.99, max_value*1.005])
# # ax.set_xlim([-0.5, 16.8])
# plt.xticks(x, corruptions, fontsize=22, rotation=15, ha='right', rotation_mode='anchor')
# ax.set_ylabel('Acc.(%)', fontsize=25)
# image_name = project_root + name + '.pdf'
# f.savefig(image_name)



# # ## acc vs pre-training epoches, ? backbone ? downstream tasks.
# figsize = 26,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = [150, 300, 600, 1200, 3000]
# alpha = np.array(alpha)
#
# dgcnn = [83.5, 85.5, 86.9, 87.9, 88.0]
# transformer = [83.9, 84.6, 84.6, 84.6, 84.6]
#
# ax.plot(alpha, dgcnn, 'ro-', linewidth=8, ms=30, label="DGCNN")
# ax.plot(alpha, transformer, 'b^-', linewidth=8, ms=30, label="Transformer")
#
# ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'Number of Pre-training Epochs', fontsize=35)
# ax.set_ylabel('Acc.(%)', fontsize=40)
# image_name = project_root + 'acc_pretraining_epochs.pdf'
# f.savefig(image_name)

# # different reconstruction targets
# corrupted =   [74.5, 77.1, 73.0]
# uncorrupted = [80.8, 84.4, 85.5]
#
# x = np.arange(3)
# total_width, n = 0.8, 2
# width = total_width / n
# x = x - (total_width - width) / 2
# figsize = 15, 5
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(corrupted), fc='b', width=width, label ="Corrupted Samples")
# ax.bar(x + width, np.array(uncorrupted), fc='r', width=width, label ="Uncorrupted Samples")
# plt.axhline(y=77.1, color='k', linestyle='--', lw=4, label='PointDAE with No Corruption')
# ax.legend(fontsize=20)
# plt.tick_params(labelsize=20)
# ax.set_ylim([72, 86])
# ax.set_xlim([-0.5, 5])
# plt.xticks([0, 1.0, 2.0],['Drop-Local', 'Affine', 'Affine+Drop-Local'], fontsize=20)
# ax.set_ylabel('Acc. (%)', fontsize=20)
# image_name = project_root + 'reconstruction_target.pdf'
# f.savefig(image_name)






## loss Vs epochs Vs corruptions
# Droplocal = [7.2, 6.4]
# Affine = [11.2, 8.8]
#
# x = np.arange(2)
# total_width, n = 0.8, 2
# width = total_width / n
# x = x - (total_width - width) / 2
# figsize = 14,12
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(Droplocal), fc='b', width=width, label ="Drop Local")
# ax.bar(x + width, np.array(Affine), fc='r', width=width, label ="Affine")
# ax.legend(fontsize=40)
# plt.tick_params(labelsize=40)
# ax.set_ylim([5, 12])
# # ax.set_xlim([-0.5, 1.5])
# plt.xticks([0, 1.0],['300 epochs', '1200 epochs'], fontsize=40)
# ax.set_ylabel('Reconstruction CD Losses (x 1e-3)', fontsize=40)
# image_name = project_root + 'reconstruction_CD_loss_epochs_corruptions.pdf'
# f.savefig(image_name)
#
# ## acc Vs epochs Vs corruptions
# Droplocal = [68.9, 71.6]
# Affine = [74.5, 76.6]
# x = np.arange(2)
# total_width, n = 0.8, 2
# width = total_width / n
# x = x - (total_width - width) / 2
# figsize = 14,12
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(Droplocal), fc='b', width=width, label ="Drop Local")
# ax.bar(x + width, np.array(Affine), fc='r', width=width, label ="Affine")
# ax.legend(fontsize=40)
# plt.tick_params(labelsize=40)
# ax.set_ylim([65, 78])
# # ax.set_xlim([-0.5, 1.5])
# plt.xticks([0, 1.0],['300 epochs', '1200 epochs'], fontsize=40)
# ax.set_ylabel('Accuracy (%)', fontsize=40)
# image_name = project_root + 'accuracy_epochs_corruptions.pdf'
# f.savefig(image_name)




#
# ################## plot loss curve
# local_global = './new_exp_transformer/pretrain_PointCAE_transformer_dropout_patch_affine_r3_maskpatch_p0005PointCAE_transformer_with_fc_center_p/\
# cfgs/log1/20221218_023151.log'
# local_only = './new_exp_transformer/pretrain_PointCAE_transformer_dropout_patch_affine_r3_maskpatchPointCAE_transformer/\
# cfgs/log1/20230101_140833.log'
# global_only = './new_exp_transformer/pretrain_PointCAE_transformer_dropout_patch_affine_r3_maskpatchPointCAE_transformer_fc_center\
# /cfgs/log1/20230103_124941.log'
#
# log_file = open(local_global)
# log_lines = log_file.readlines()
# local_global_log = []
# for line in log_lines:
#     if (line.find('(s) Losses = [') != -1):
#         local_global_log.append(float(line.split('[\'')[1].split('\']')[0]))
# print(len(local_global_log))
#
# log_file = open(local_global)
# log_lines = log_file.readlines()
# local_global_log_global = []
# for line in log_lines:
#     if (line.find('(s) Losses = [') != -1):
#         local_global_log_global.append(float(line.split('[\'')[2].split('\']')[0]))
# print(len(local_global_log_global))
#
# log_file = open(local_only)
# log_lines = log_file.readlines()
# local_only_log = []
# for line in log_lines:
#     if (line.find('(s) Losses = [') != -1):
#         local_only_log.append(float(line.split('[\'')[1].split('\']')[0]))
# print(len(local_only_log))
#
# log_file = open(global_only)
# log_lines = log_file.readlines()
# global_only_log = []
# for line in log_lines:
#     if (line.find('(s) Losses = [') != -1):
#         global_only_log.append(float(line.split('[\'')[1].split('\']')[0]))
# print(len(global_only_log))
# # droplocalx4_log = gaussian_filter1d(droplocalx4_log, sigma=2)
#
#
# from matplotlib.pyplot import MultipleLocator, FormatStrFormatter
# figsize = 16, 14
# f, ax = plt.subplots(figsize=figsize)
# iteration = np.array(list(range(0, 301)))
# begin_iter = 100
# gap_iter = 10
# ax.plot(iteration[begin_iter:][::gap_iter], local_only_log[begin_iter:][::gap_iter], '--r', linewidth=10, label="$\mathcal{L}_{local}$ Only", alpha=1.0)
# ax.plot(iteration[begin_iter:][::gap_iter], local_global_log[begin_iter:][::gap_iter], 'r', linewidth=10, label="$\mathcal{L}_{global}+\mathcal{L}_{local}$", alpha=1.0)
#
# ymajorFormatter = FormatStrFormatter('%1.1f') # y轴刻度格式为两位小数
# ymajorLocator  = MultipleLocator(0.1)    # y轴刻度间隔 5
# ax.yaxis.set_major_locator(ymajorLocator)
# ax.yaxis.set_major_formatter(ymajorFormatter)
# # ax.plot(rotation, intra, '--c', linewidth=8, label="Intra-Domain $\mathcal{T}$")
# # plt.tick_params(labelsize=10)
# # plt.ylim(20, 40)
# ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel('Pre-training Epoches', fontsize=50)
# ax.set_ylabel('Value of $\mathcal{L}_{local}$ (x1e-3)', fontsize=50)
# image_name = project_root +  'pretraining_local_loss_curve.pdf'
# f.savefig(image_name)
#
#
# from matplotlib.pyplot import MultipleLocator, FormatStrFormatter
# figsize = 16, 14
# f, ax = plt.subplots(figsize=figsize)
# iteration = np.array(list(range(0, 301)))
#
# begin_iter = 100
# gap_iter = 10
# ax.plot(iteration[begin_iter:][::gap_iter], global_only_log[begin_iter:][::gap_iter], '--r', linewidth=10, label="$\mathcal{L}_{global}$ Only", alpha=1.0)
# ax.plot(iteration[begin_iter:][::gap_iter], local_global_log_global[begin_iter:][::gap_iter], 'r', linewidth=10, label="$\mathcal{L}_{global}+\mathcal{L}_{local}$", alpha=1.0)
#
# ymajorFormatter = FormatStrFormatter('%1.f') # y轴刻度格式为两位小数
# ymajorLocator  = MultipleLocator(5.0)    # y轴刻度间隔 5
# ax.yaxis.set_major_locator(ymajorLocator)
# ax.yaxis.set_major_formatter(ymajorFormatter)
# # ax.plot(rotation, intra, '--c', linewidth=8, label="Intra-Domain $\mathcal{T}$")
# # plt.tick_params(labelsize=10)
# # plt.ylim(20, 40)
# ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel('Pre-training Epoches', fontsize=50)
# ax.set_ylabel('Value of $\mathcal{L}_{global}$ (x1e-3)', fontsize=50)
#
# image_name = project_root +  'pretraining_global_loss_curve.pdf'
# f.savefig(image_name)






# ## masking strategies
# figsize = 26,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# alpha = np.array(alpha)
#
# block = [84.70, 84.77, 84.80, 84.48, 84.22, 83.98]
# random = [85.0, 85.12, 85.35, 85.29, 85.29, 85.00]
#
#
# ax.plot(alpha, block, 'b^-', linewidth=8, ms=30, label="Block masking")
# ax.plot(alpha, random, 'ro-', linewidth=8, ms=30, label="Random masking")
#
# ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'Values of $m$', fontsize=35)
# ax.set_ylabel('Acc.(%)', fontsize=40)
# # ax.set_ylim([80,85])
# # plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ['Input images', '1st conv', '1st block', '2nd block', '3rd block', '4th block'])
# image_name = project_root + 'masking_trategies.pdf'
# f.savefig(image_name)


# # ################################### estimated noraml vs. GT normal
# point_mae = [89.26, 88.19, 84.66]
# estimated = [90.18, 88.57, 84.74]
# gt = [90.76, 88.74, 85.35]
#
# #
# x = np.arange(3)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2
# figsize = 13,7
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(point_mae), fc='g', width=width, label ="PC Only (i.e.,Point-MAE)")
# ax.bar(x+ width, np.array(estimated), fc='b', width=width, label ="Estimated Surfels")
# ax.bar(x + 2* width, np.array(gt), fc='r', width=width, label ="Ground Truth Surfels")
# ax.legend(fontsize=28)
# plt.tick_params(labelsize=35)
# ax.set_ylim([84.5, 91.5])
# # ax.set_xlim([-0.5, 1.5])
# plt.xticks([0, 1.0, 2.0],['OBJ-BG', 'OBJ-ONLY', 'PB-T50-RS'], fontsize=35)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Acc.(%)', fontsize=35)
# image_name = project_root + 'estimated_normal.pdf'
# f.savefig(image_name)
#
#
# ## reconstructing all or reconstructing masked surface.
#
# all_surface = [89.27, 88.47, 85.27]
# gt = [90.76, 88.74, 85.35]
#
# x = np.arange(3)
# total_width, n = 0.6, 2
# width = total_width / n
# x = x - (total_width - width) / 2
# figsize = 13,6
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(all_surface), fc='b', width=width, label ="All surfels")
# ax.bar(x + width, np.array(gt), fc='r', width=width, label ="Masked surfels only")
# ax.legend(fontsize=30)
# plt.tick_params(labelsize=35)
# ax.set_ylim([84.5, 91.0])
# # ax.set_xlim([-0.5, 1.5])
# plt.xticks([0, 1.0, 2.0],['OBJ-BG', 'OBJ-ONLY', 'PB-T50-RS'], fontsize=35)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Acc.(%)', fontsize=35)
# image_name = project_root + 'masked_or_all_surfaces.pdf'
# f.savefig(image_name)

## oriented normal distance or unoriented normal distance

# oriented = [89.64, 87.61, 84.91]
# unoriented = [90.76, 88.74, 85.35]
#
# x = np.arange(3)
# total_width, n = 0.6, 2
# width = total_width / n
# x = x - (total_width - width) / 2
# figsize = 13,6
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(oriented), fc='b', width=width, label ="Oriented Normal Distance")
# ax.bar(x + width, np.array(unoriented), fc='r', width=width, label ="Unoriented Normal Distance")
# ax.legend(fontsize=28)
# plt.tick_params(labelsize=32)
# ax.set_ylim([84.5, 91.0])
# # ax.set_xlim([-0.5, 1.5])
# plt.xticks([0, 1.0, 2.0],['OBJ-BG', 'OBJ-ONLY', 'PB-T50-RS'], fontsize=35)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Acc.(%)', fontsize=35)
# image_name = project_root + 'oriented_or_unoriented_normal.pdf'
# f.savefig(image_name)


# # ## hyper-parameter alpha.
# figsize = 13,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = [0.1, 0.2, 0.3, 0.4]
# alpha = np.array(alpha)
#
# acc = [71.2, 72.6, 72.8, 71.0]
#
# ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
#
# # ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'Values of $\alpha$ (log scale)', fontsize=35)
# ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.1, 0.2, 0.3, 0.4], ['1e-4', '1e-3', '1e-2', '1e-1'])
# image_name = project_root + 'alpha.pdf'
# f.savefig(image_name)
#
#
# # ## hyper-parameter alpha.
# figsize = 13,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = [0.2, 0.3, 0.4, 0.5]
# alpha = np.array(alpha)
#
# acc = [71.9, 72.7, 72.8, 71.3]
#
# ax.plot(alpha, acc, 'ro-', linewidth=8, ms=30)
#
# # ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'Values of $\beta$ (log scale)', fontsize=35)
# ax.set_ylabel('Acc.(%)', fontsize=40)
# ax.set_ylim([70.0, 74.0])
# plt.xticks([0.2, 0.3, 0.4, 0.5], ['1e-3', '1e-2', '1e-1', '1.0'])
# image_name = project_root + 'beta.pdf'
# f.savefig(image_name)


# ######################################################## for ECCV2022
# # lambda1 = [0.01, 0.1 ,1, 5, 10, 50, 100]
# lambda1 = [1,2,3,4,5,6,7]
# acc = [55.0, 56.0, 57.1, 58.7, 55.5, 53.0, 50.3]
#
# figsize = 26,13
# f, ax = plt.subplots(figsize=figsize)
# # alpha = np.array(powern)
#
# ax.axhline(y=46.6, color='b', lw=8, dashes=[1,1], label="ResNet-18")
# ax.plot(lambda1, acc, 'ro-', linewidth=8, ms=20, label="ResNet-18 + AdvStyle")
#
# ax.legend(loc='best', fontsize=45)
# plt.tick_params(labelsize=45)
# plt.xticks([1,2,3,4,5,6,7], \
#            ['$0.01$', '$0.1$' ,'$1.0$','$5.0$','$10.0$','$50.0$','$100.0$'])
# ax.set_xlabel(r'Values of $\lambda$', fontsize=50)
# ax.set_ylabel(r'Accuracy (\%)', fontsize=55)
#
# image_name = 'res18_adv_lambda.pdf'
# f.savefig(image_name)

#
# # lambda1 = [0.01, 0.1 ,1, 5, 10, 50, 100]
#
# acc = [59.2, 59.3, 59.5, 63.7, 67.1, 60.7, 49.5]
#
# figsize = 26,13
# f, ax = plt.subplots(figsize=figsize)
# # alpha = np.array(powern)
#
# ax.axhline(y=50.5, color='b', lw=8, dashes=[1,1], label="ResNet-50")
# ax.plot(lambda1, acc, 'ro-', linewidth=6, ms=20, label="ResNet-50 + AdvStyle")
#
# ax.legend(loc='best', fontsize=45)
# plt.tick_params(labelsize=45)
# plt.xticks([1,2,3,4,5,6,7], \
#            ['$0.01$', '$0.1$' ,'$1.0$','$5.0$','$10.0$','$50.0$','$100.0$'])
#
# ax.set_xlabel(r'Values of $\lambda$', fontsize=50)
# ax.set_ylabel(r'Accuracy (\%)', fontsize=55)
# image_name = 'res50_adv_lambda.pdf'
# f.savefig(image_name)




################################### for CVPR2022
# powern = np.array([1,2,3,4,5])
# ratio = np.array([3,6,8,9,10])
# print(powern)
# print(ratio)
#
# figsize = 13,13
# f, ax = plt.subplots(figsize=figsize)
# alpha = np.array(powern)
#
# model=make_interp_spline(powern, ratio)
# xs=np.linspace(1,5,500)
# ys=model(xs)
#
# ax.plot(xs, ys, linewidth=6, ms=20)
# ax.plot(powern, ratio, 'o', linewidth=6, ms=20)
#
#
# # ax.axhline(y=1, color='b', lw=6, dashes=[1,1])
# # ax.legend(loc='best', fontsize=45)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'X', fontsize=39)
# ax.set_ylabel(r'O', fontsize=40)
# # ax.set_ylim([80,85])
# plt.xticks(powern)
# plt.yticks(ratio)
# image_name = '357.pdf'
# f.savefig(image_name)


#################################### adain vs efdm on features of different dimension.
import time
# repeat = 1000
# powern = [64, 256, 1024, 2048, 4096, 8000, 12000, 16000, 20000, 24000, 28000, 32768, 32770, 36000, 40000, 65536, 80000, 90000]
# ratio = []
# #powern = []
# #powern.append(512)
# for i in powern:
# # for i in range(5,17):
#     # length = 2 ** i
#     # powern.append(length)
#     length = i
#     print(length)
#     adain_tensor = torch.zeros(repeat)
#     efdm_tensor = torch.zeros(repeat)
#     for j in range(repeat):
#         a = torch.rand(length)
#         b = torch.rand(length)
#         timer = time.time()
#         x_normed = (a - a.mean()) / a.std() * b.std() + b.mean()
#         adain_time = time.time() - timer
#         adain_tensor[j] = adain_time
#
#         timer = time.time()
#         value_x, index_x = torch.sort(a)
#         value_b, index_b = torch.sort(b)
#         inverse_index = index_b.argsort(-1)
#         new_x = value_x.gather(-1, inverse_index)
#         new_x = a + new_x - a
#         efdm_time = time.time() - timer
#         efdm_tensor[j] = efdm_time
#     # ratio.append(efdm_tensor.mean().item())
#     # print(efdm_tensor.mean().item())
#     # print(adain_tensor.mean().item())
#     ratio.append((efdm_tensor.mean() / adain_tensor.mean()).item())
#     # ratio.append(adain_tensor.mean().item())
#     print(efdm_tensor.mean() / adain_tensor.mean())




# powern = [64, 256, 1024, 2048, 4096, 8000, 12000, 16000, 20000, 24000, 28000, 32000, 36000]
# ratio = [0.771128237247467, 1.292392611503601, 3.475908041000366, 6.7286505699157715, 12.113306045532227, 17.788484573364258, 25.612506866455078, 30.264127731323242, 34.562076568603516, 39.255672454833984, 42.52086715698242, 45.06665344238281, 45.64132583618164]
# print(powern)
# print(ratio)
#
# figsize = 26,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = np.array(powern)
#
# ax.plot(alpha, ratio, 'ro-', linewidth=6, ms=20, label="EFDM / AdaIN")
# ax.axhline(y=1, color='b', lw=6, dashes=[1,1])
# ax.legend(loc='best', fontsize=45)
# plt.tick_params(labelsize=40)
# ax.set_xlabel(r'Feature dimension', fontsize=39)
# ax.set_ylabel(r'Time cost ratio', fontsize=40)
# # ax.set_ylim([80,85])
# # plt.xticks([4,6,8,10, 12, 14, 16, 18], \
# #            ['$2^4$', '$2^6$' ,'$2^8$','$2^{10}$','$2^{12}$','$2^{14}$','$2^{16}$','$2^{18}$'])
#
# image_name = 'ratio_diff_dim.pdf'
# f.savefig(image_name)

# import torch
# x = torch.Tensor([0, 0, 0.1, 0.1, 0.2])
# a = torch.Tensor([0, 0.1, 0.3, 0.4, 0.5])
# # a = torch.Tensor([0.1, 0.1, 0.4, 0.4, 0.5])
#
#
# a = (x - x.mean())/x.std() * a.std() + a.mean()
# print(a)
# mean = a.mean()
# std = a.std()
# third = (((a-mean)/std).pow(3)).mean()
# fourth = (((a-mean)/std).pow(4)).mean()
# print(mean, std, third, fourth)


# ################################### speed comparison
# gaty = [25.61]
# kalis = [19.84]
# AdaIN = [0.0038]
# HM = [0.33]
# Sort_Matching = [0.0039]
# n=1; m=2
# gs = gridspec.GridSpec(2,1, height_ratios=[n,m], hspace=0.1)
# plt.figure()
# ax = plt.subplot(gs[0,0:])
# ax2 = plt.subplot(gs[1,0:], sharex=ax)
#
# figsize = 13,13
# # f, ax = plt.subplots(figsize=figsize)
# fig, (ax, ax2) = plt.subplots(2,1, figsize=figsize, sharex=True)
# fig.subplots_adjust(hspace=0.05)  # adjust space between axes
#
# ax.spines['top'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# ax.axes.get_xaxis().set_visible(False)
# ax.tick_params(labeltop=False)
# ax2.xaxis.tick_bottom()
#
# ax.set_ylim(15,30)
# ax2.set_ylim(0, 0.5)
#
# on = (n+m)/n; om=(n+m)/m
# d = .5  # proportion of vertical to horizontal extent of the slanted line
# kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
#               linestyle="none", color='k', mec='k', mew=1, clip_on=False)
# ax.plot([0, 1], [0, 0], transform=ax.transAxes, **kwargs)
# ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
#
# x = np.arange(1)
# total_width, n = 1.0, 5
# width = total_width / n
# x = x - (total_width - width) / 2
#
# ax.bar(x, np.array(gaty), fc='y', width=width, label ="Gaty")
# ax.bar(x + width, np.array(kalis), fc='m', width=width, label ="Kalischeket")
# ax.bar(x + width * 2, np.array(HM), fc='g', width=width, label ="HM")
# ax.bar(x + width * 3, np.array(Sort_Matching), fc='r', width=width, label ="Sort-Matching")
# ax.bar(x + width * 4, np.array(AdaIN), fc='b', width=width, label ="AdaIN")
# ax2.bar(x, np.array(gaty), fc='y', width=width, label ="Gaty")
# ax2.bar(x + width, np.array(kalis), fc='m', width=width, label ="Kalischeket")
# ax2.bar(x + width * 2, np.array(HM), fc='g', width=width, label ="HM")
# ax2.bar(x + width * 3, np.array(Sort_Matching), fc='r', width=width, label ="Sort-Matching")
# ax2.bar(x + width * 4, np.array(AdaIN), fc='b', width=width, label ="AdaIN")
# ax.legend(fontsize=30)
# plt.tick_params(labelsize=20)
# # plt.xticks([])
# # ax.set_ylim([80,90])
# plt.xticks([-0.4, -0.2, 0, 0.2, 0.4],['Gaty', 'Kalischeket', 'HM', 'Sort-Matching', 'AdaIN'], fontsize=22)
# # ax.set_xlabel('Methods', fontsize=20)
# # ax.set_ylabel('Time (s)', fontsize=20)
# # ax2.set_ylabel(fontsize=20)
#
# # plt.show()
#
#
# # x = np.arange(1)
# # total_width, n = 0.8, 5
# # width = total_width / n
# # x = x - (total_width - width) / 2
# # figsize = 13,13
# # f, ax = plt.subplots(figsize=figsize)
# # ax.bar(x, np.array(gaty), fc='y', width=width, label ="Gaty")
# # ax.bar(x + width, np.array(kalis), fc='m', width=width, label ="Kalischeket")
# # ax.bar(x + width * 2, np.array(HM), fc='g', width=width, label ="HM")
# # ax.bar(x + width * 3, np.array(Sort_Matching), fc='r', width=width, label ="Sort-Matching")
# # ax.bar(x + width * 3, np.array(AdaIN), fc='b', width=width, label ="AdaIN")
# # ax.legend(fontsize=40)
# # plt.tick_params(labelsize=35)
# # # ax.set_ylim([80,90])
# # # plt.xticks([0,1],['ResNet18', 'ResNet50'], fontsize=35)
# # # ax.set_xlabel('Methods', fontsize=20)
# # ax.set_ylabel('Time (s)', fontsize=35)
# image_name = 'speed.pdf'
# fig.savefig(image_name)



# ##################### equivalent values percent.
# figsize = 26,10
# f, ax = plt.subplots(figsize=figsize)
# alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
# alpha = np.array(alpha)
#
# with_relu = [100, 60, 53, 42, 38, 36]
# with_prelu = [100, 60, 34, 12, 5, 2]
#
#
# ax.plot(alpha, with_relu, 'ro-', linewidth=8, ms=30, label="ReLU")
# ax.plot(alpha, with_prelu, 'b^-', linewidth=8, ms=30, label="PReLU")
#
# ax.legend(loc='best', fontsize=60)
# plt.tick_params(labelsize=40)
# # ax.set_xlabel(r'Values of $\alpha$', fontsize=35)
# ax.set_ylabel('Percent of equivalent values', fontsize=40)
# # ax.set_ylim([80,85])
# plt.xticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], ['Input images', '1st conv', '1st block', '2nd block', '3rd block', '4th block'])
# image_name = 'percent_equivalent.pdf'
# f.savefig(image_name)



# ################################## sorting match vs. histogram matching
# EFDMix = [83.4, 86.6]
# sort_matching = [83.0, 86.5]
# mixstyle = [82.6, 85.8]
#
# histogram_matching = [81.7, 85.6]
# adain = [82.1, 85.6]
#
# EFDMix_error = [0.5, 0.6]
# sort_matching_error = [0.6, 0.5]
# mixsytle_error = [0.4, 0.6]
# error_params=dict(elinewidth=5,capsize=8)#设置误差标记参数
# # labels = ['Acc.(\%)']
# x = np.arange(2)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 3
# figsize = 13,6.5
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(EFDMix),yerr=EFDMix_error, error_kw=error_params, fc='r', width=width, label ="EFDMix")
# ax.bar(x + width, np.array(sort_matching),yerr=sort_matching_error, error_kw=error_params, fc='y', width=width, label ="EFDM")
# ax.bar(x + width * 2, np.array(mixstyle),yerr=mixsytle_error, error_kw=error_params, fc='b', width=width, label ="MixStyle")
#
# ax.legend(fontsize=30)
# plt.tick_params(labelsize=30)
# ax.set_ylim([81,87.5])
# ax.set_xlim([-0.4, 2.5])
# plt.xticks([0.1,1.1],['ResNet18', 'ResNet50'], fontsize=30)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Acc.(%)', fontsize=30)
# image_name = 'mix_vs_replace.pdf'
# f.savefig(image_name)


# ################## plot loss curve
# ms_file = './imcls/output_histogram_matching/pacs/Vanilla2/resnet18_ms_l123/random/sketch/seed1/log.txt'
# sort_file = './imcls/output_sorting/pacs/Vanilla2/resnet18_his_l123/random_quicksort/sketch/seed1/log.txt'
# hist_file = './imcls/output_histogram_matching/pacs/Vanilla2/resnet18_realhis_l123/random/sketch/seed1/log.txt'
#
# log_file = open(ms_file)
# log_lines = log_file.readlines()
# ms_loss = []
# for line in log_lines:
#     if (line.find('50][80/85]') != -1):
#         # print(line.split(' ')[8].split('(')[1].split(')')[0])
#         ms_loss.append(float(line.split(' ')[8].split('(')[1].split(')')[0]))
# ms_loss = ms_loss[4:]
#
# log_file = open(sort_file)
# log_lines = log_file.readlines()
# sort_loss = []
# for line in log_lines:
#     if (line.find('50][80/85]') != -1):
#         # print(line.split(' ')[8].split('(')[1].split(')')[0])
#         sort_loss.append(float(line.split(' ')[8].split('(')[1].split(')')[0]))
# sort_loss = sort_loss[4:]
#
# log_file = open(hist_file)
# log_lines = log_file.readlines()
# his_loss = []
# for line in log_lines:
#     if (line.find('50][80/85]') != -1):
#         # print(line.split(' ')[8].split('(')[1].split(')')[0])
#         his_loss.append(float(line.split(' ')[8].split('(')[1].split(')')[0]))
# his_loss = his_loss[4:]
#
# figsize = 12,6
# f, ax = plt.subplots(figsize=figsize)
# rotation = np.array(list(range(5, 51)))
#
# ax.plot(rotation, sort_loss, 'r', linewidth=8, label="EFDM", alpha=1.0)
# ax.plot(rotation, his_loss, 'b', linewidth=8, label="HM", alpha=1.0)
# ax.plot(rotation, ms_loss, 'g', linewidth=8, label="AdaIN", alpha=0.5)
#
# # ax.plot(rotation, intra, '--c', linewidth=8, label="Intra-Domain $\mathcal{T}$")
# # plt.tick_params(labelsize=10)
# # plt.ylim(20, 40)
# ax.legend(loc='best', fontsize=30)
# plt.tick_params(labelsize=20)
# ax.set_xlabel('Epoches', fontsize=20)
# ax.set_ylabel('Training Losses', fontsize=20)
#
# image_name = 'loss_curve.pdf'
# f.savefig(image_name)


# ##################### hyperparameter alpha
# figsize = 13,7
# f, ax = plt.subplots(figsize=figsize)
# alpha = [0.1, 0.2, 0.3, 0.5, 0.9]
# alpha = np.array(alpha)
#
# mixsorting = [83.4, 82.7, 83.0, 82.9, 82.5]
# mixstyle = [82.6, 82.0, 81.7, 81.7, 81.2]
#
# mixsorting_std = [0.5, 0.4, 0.5, 0.4, 0.4]
# mixstyle_std = [0.4, 0.5, 0.5, 0.4, 0.5]
#
# ax.plot(alpha, mixsorting, 'ro-', linewidth=8, ms=30, label="EFDMix")
# r1 = list(map(lambda x: x[0]-x[1], zip(mixsorting, mixsorting_std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(mixsorting, mixsorting_std)))
# ax.fill_between(alpha, r1, r2, color='r', alpha=0.2)
#
# ax.plot(alpha, mixstyle, 'b^-', linewidth=8, ms=30, label="MixStyle")
# r1 = list(map(lambda x: x[0]-x[1], zip(mixstyle, mixstyle_std)))
# r2 = list(map(lambda x: x[0]+x[1], zip(mixstyle, mixstyle_std)))
# ax.fill_between(alpha, r1, r2, color='b', alpha=0.2)
#
# ax.legend(loc='best', fontsize=35)
# plt.tick_params(labelsize=25)
# ax.set_xlabel(r'Values of $\alpha$', fontsize=25)
# ax.set_ylabel('Acc. (%)', fontsize=25)
# ax.set_ylim([80,86])
# #ax.set_xlim([0.05, 1.5])
# image_name = 'alpla.pdf'
# f.savefig(image_name)


# ################################### sorting match vs. histogram matching
# sorting_matching = [83.0, 86.6]
# histogram_matching = [81.7, 85.6]
# adain = [82.1, 85.8]
#
#
# quicksort_error = [0.7, 0.6]
# index_error = [0.6, 0.5]
# random_error = [0.4, 0.7]
# error_params=dict(elinewidth=5,capsize=8)#设置误差标记参数
# # labels = ['Acc.(\%)']
# x = np.arange(2)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 3
# figsize = 13,13
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(sorting_matching),yerr=quicksort_error, error_kw=error_params, fc='r', width=width, label ="Sort Matching")
# ax.bar(x + width, np.array(histogram_matching),yerr=index_error, error_kw=error_params, fc='b', width=width, label ="Histogram Matching")
# ax.bar(x + width * 2, np.array(adain),yerr=random_error, error_kw=error_params, fc='g', width=width, label ="AdaIN", alpha=0.5)
#
# ax.legend(fontsize=40)
# plt.tick_params(labelsize=35)
# ax.set_ylim([80,90.5])
# plt.xticks([0,1],['ResNet18', 'ResNet50'], fontsize=35)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Acc.(%)', fontsize=35)
# image_name = 'sorting_vs_histogram.pdf'
# f.savefig(image_name)

#
# ################################### various ordering strategy  of equavilient values
# quicksort = [83.1, 86.8]
# index = [83.1, 86.5]
# rando = [82.9, 86.4]
# neighbor = [83.0, 86.8]
# 
# quicksort_error = [0.7, 0.6]
# index_error = [0.6, 0.8]
# random_error = [0.7, 0.7]
# neighbor_error = [0.5, 0.7]
# error_params=dict(elinewidth=5,capsize=8)#设置误差标记参数
# # labels = ['Acc.(\%)']
# x = np.arange(2)
# total_width, n = 0.8, 4
# width = total_width / n
# x = x - (total_width - width) / 2
# figsize = 13,7
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(quicksort),yerr=quicksort_error, error_kw=error_params, fc='r', width=width, label ="Quicksort")
# ax.bar(x + width, np.array(index),yerr=index_error, error_kw=error_params, fc='y', width=width, label ="Preserving")
# ax.bar(x + width * 2, np.array(rando),yerr=random_error, error_kw=error_params, fc='g', width=width, label ="Random")
# ax.bar(x + width * 3, np.array(neighbor), yerr=neighbor_error, error_kw=error_params, fc='b', width=width, label ="Neighbor")
# ax.legend(fontsize=30)
# plt.tick_params(labelsize=35)
# ax.set_ylim([81.5, 87.8])
# ax.set_xlim([-0.5, 1.5])
# plt.xticks([0,1],['ResNet18', 'ResNet50'], fontsize=35)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Acc.(%)', fontsize=35)
# image_name = 'ordering_strategies.pdf'
# f.savefig(image_name)



########### an unknown test.
# import torch
#
# a = torch.rand(1,1,10)
# a[0][0][0] = 0
# a[0][0][1] = 0
# a[0][0][5] = 0
# b = torch.rand(1,1,10) * 1e-32
# a = a + b
# value_x, index_x = torch.sort(a)
# src_values, src_unique_indices, src_counts = torch.unique(a[0][0],
#                                                               return_inverse=True,
#                                                               return_counts=True)
#
# print(src_values)
# print(src_unique_indices)
# print(src_counts)
# print(src_values[src_counts>1])
# print(index_x)
# print(index_x[value_x == src_values[src_counts>1][0].item()] )
# duplicated_index = index_x[value_x == src_values[src_counts>1][0].item()]
# # print(duplicated_index[torch.randperm(duplicated_index.nelement())])
# index_x[value_x == src_values[src_counts > 1][0].item()] = duplicated_index[torch.randperm(duplicated_index.nelement())]
#
# print(a)
# print(value_x)
# print(index_x)

# from cycler import cycler
# matplotlib.rcParams['axes.prop_cycle'] = cycler(markevery=cases, color=colors)
# ################ plot 累计概率分布和反累计概率分布
# figsize = 13,11
# f, ax = plt.subplots(figsize=figsize)
# # value = np.array(list(range(0, 5)))
# value_x = np.array([0.00,0.10,0.20])
# x = [0.4, 0.8, 1.0]
# value_y = np.array([0.00, 0.10, 0.30, 0.40, 0.50])
# y = [0.2, 0.4, 0.6, 0.8, 1.0]
#
# value_o = np.array([0.10, 0.40, 0.50])
# o = [0.4, 0.8, 1.0]
#
# # value_x_gt = np.array([0,1,2,2.999, 3, 3.999, 4])
# # interval0_x = [1 if (i>=0 and i <3) else 0 for i in value_x_gt]
# # interval1_x = [1 if (i>=3 and i <4) else 0 for i in value_x_gt]
# # interval2_x = [1 if (i>=4) else 0 for i in value_x_gt]
# # x_gt = np.array([0.2]*7)  * interval0_x + np.array([0.8] * 7) * interval1_x + np.array([1.0]*7) * interval2_x
# #
# # value_y_gt = np.array([0, 0.999, 1, 1.999, 2, 3.999, 4])
# # interval0_y = [1 if (i>=0 and i <1) else 0 for i in value_y_gt]
# # interval1_y = [1 if (i>=1 and i <2) else 0 for i in value_y_gt]
# # interval2_y = [1 if (i>=2 and i <4) else 0 for i in value_y_gt]
# # interval3_y = [1 if (i>=4) else 0 for i in value_y_gt]
# # y_gt = np.array([0.2]*7) * interval0_y + np.array([0.4] * 7) * interval1_y + np.array([0.8]*7) * interval2_y + np.array([1.0]*7) * interval3_y
# #
#
# ax.plot(value_x, x, 'bo-', linewidth=8, ms=30, alpha=1.0,  drawstyle='steps-post', label=r"eCDF $\mathbf{x}$")
# # ax.plot(value_x_gt, x_gt, 'b--', linewidth=8, ms=30, alpha=0.5, label="CDF-GT $x$")
# ax.plot(value_y, y, 'r^-', linewidth=8, ms=30, alpha=1.0, drawstyle='steps-post', label=r"eCDF $\mathbf{y}$")
# ax.plot(value_o, o, 'gD-', linewidth=8, ms=30, alpha=0.5, drawstyle='steps-post', label=r"eCDF $\mathbf{o}$")
# # ax.plot(value_y_gt, y_gt, 'r--', linewidth=8, ms=30, alpha=0.5, label="CDF-GT $y$")
# ax.legend(loc='best', fontsize=35)
# plt.tick_params(labelsize=35)
# ax.set_xlabel('Values', fontsize=35)
# ax.set_ylabel('Cumulative Probability', fontsize=35)
# my_x_ticks = np.arange(0,7,1) * 0.1
# plt.xticks(my_x_ticks)
# ax.set_ylim([0.1, 1.05])
# image_name = 'cdf.pdf'
# f.savefig(image_name)
# # #
# #
# #
# figsize = 13,11
# f, ax = plt.subplots(figsize=figsize)
# value = np.array(list(range(1, 6))) / 5
# x = [0, 3, 3, 3, 4]
# y = [0, 1, 2, 2, 4]
# 
# value_x_gt = np.array([0.2, 0.2001, 0.4, 0.8, 0.8001, 1.0])
# interval0_x = [1 if (i<=0.2) else 0 for i in value_x_gt]
# interval1_x = [1 if (i>0.2 and i <=0.8) else 0 for i in value_x_gt]
# interval2_x = [1 if (i>0.8 and i <=1.0) else 0 for i in value_x_gt]
# x_gt = np.array([0]*6)  * interval0_x + np.array([3] * 6) * interval1_x + np.array([4]*6) * interval2_x
# 
# 
# value_y_gt = np.array([0.2, 0.2001, 0.4, 0.4001, 0.8, 0.8001, 1.0])
# interval0_y = [1 if (i<=0.2) else 0 for i in value_y_gt]
# interval1_y = [1 if (i>0.2 and i <=0.4) else 0 for i in value_y_gt]
# interval2_y = [1 if (i>0.4 and i <=0.8) else 0 for i in value_y_gt]
# interval3_y = [1 if (i>0.8) else 0 for i in value_y_gt]
# y_gt = np.array([0]*7)  * interval0_y + np.array([1] * 7) * interval1_y + np.array([2]*7) * interval2_y + np.array([4]*7) * interval3_y
# 
# ax.plot(value, x, 'bo-', linewidth=8, ms=30, alpha=0.5, label="ICDF $x$")
# ax.plot(value_x_gt, x_gt, 'b--', linewidth=8, ms=30,  alpha=0.5,label="ICDF-GT $x$")
# ax.plot(value, y, 'r^-', linewidth=8, ms=30,  alpha=0.5,label="ICDF $y$")
# ax.plot(value_y_gt, y_gt, 'r--', linewidth=8, ms=30,  alpha=0.5,label="ICDF-GT $y$")
# ax.legend(loc='best', fontsize=32.5)
# plt.tick_params(labelsize=35)
# ax.set_xlabel('Cumulative Probability $p$', fontsize=35)
# ax.set_ylabel('Values', fontsize=35)
# my_y_ticks = np.arange(0,5,1) #* 0.1
# plt.yticks(my_y_ticks)
# # my_x_ticks = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6])#np.arange(0,4,1) * 0.1
# plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], [ r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1$'])
# image_name = 'inverse_cdf.pdf'
# f.savefig(image_name)
# 
# 
# ######### cdf and icdf with output
# figsize = 13,11
# f, ax = plt.subplots(figsize=figsize)
# # value = np.array(list(range(0, 5)))
# value_x = np.array([0,2,4])
# x = [0.2, 0.8, 1.0]
# value_y = np.array([0,1,2,4])
# y = [0.2, 0.4, 0.8, 1.0]
# 
# # value_x_gt = np.array([0,1,2,2.999, 3, 3.999, 4])
# # interval0_x = [1 if (i>=0 and i <3) else 0 for i in value_x_gt]
# # interval1_x = [1 if (i>=3 and i <4) else 0 for i in value_x_gt]
# # interval2_x = [1 if (i>=4) else 0 for i in value_x_gt]
# # x_gt = np.array([0.2]*7)  * interval0_x + np.array([0.8] * 7) * interval1_x + np.array([1.0]*7) * interval2_x
# 
# value_y_gt = np.array([0, 0.999, 1, 1.999, 2, 3.999, 4])
# interval0_y = [1 if (i>=0 and i <1) else 0 for i in value_y_gt]
# interval1_y = [1 if (i>=1 and i <2) else 0 for i in value_y_gt]
# interval2_y = [1 if (i>=2 and i <4) else 0 for i in value_y_gt]
# interval3_y = [1 if (i>=4) else 0 for i in value_y_gt]
# y_gt = np.array([0.2]*7) * interval0_y + np.array([0.4] * 7) * interval1_y + np.array([0.8]*7) * interval2_y + np.array([1.0]*7) * interval3_y
# 
# 
# ax.plot(value_x, x, 'yo-', linewidth=8, ms=30, alpha=0.5,  label="CDF $o$")
# # ax.plot(value_x_gt, x_gt, 'b--', linewidth=8, ms=30, alpha=0.5, label="CDF-GT $x$")
# ax.plot(value_y, y, 'r^-', linewidth=8, ms=30, alpha=0.5, label="CDF $y$")
# ax.plot(value_y_gt, y_gt, 'r--', linewidth=8, ms=30, alpha=0.5, label="CDF-GT $y$")
# ax.legend(loc='best', fontsize=35)
# plt.tick_params(labelsize=35)
# ax.set_xlabel('Values', fontsize=35)
# ax.set_ylabel('Cumulative Probability $p$', fontsize=35)
# my_x_ticks = np.arange(0,5,1) #* 0.1
# plt.xticks(my_x_ticks)
# image_name = 'cdf_output.pdf'
# f.savefig(image_name)
# 
# 
# 
# #
# figsize = 13,11
# f, ax = plt.subplots(figsize=figsize)
# value = np.array(list(range(1, 6))) / 5
# x = [0, 1, 2, 2, 4]
# y = [0, 1, 2, 2, 4]
# 
# # value_x_gt = np.array([0.2, 0.2001, 0.4, 0.8, 0.8001, 1.0])
# # interval0_x = [1 if (i<=0.2) else 0 for i in value_x_gt]
# # interval1_x = [1 if (i>0.2 and i <=0.8) else 0 for i in value_x_gt]
# # interval2_x = [1 if (i>0.8 and i <=1.0) else 0 for i in value_x_gt]
# # x_gt = np.array([0]*6)  * interval0_x + np.array([3] * 6) * interval1_x + np.array([4]*6) * interval2_x
# 
# 
# value_y_gt = np.array([0.2, 0.2001, 0.4, 0.4001, 0.8, 0.8001, 1.0])
# interval0_y = [1 if (i<=0.2) else 0 for i in value_y_gt]
# interval1_y = [1 if (i>0.2 and i <=0.4) else 0 for i in value_y_gt]
# interval2_y = [1 if (i>0.4 and i <=0.8) else 0 for i in value_y_gt]
# interval3_y = [1 if (i>0.8) else 0 for i in value_y_gt]
# y_gt = np.array([0]*7)  * interval0_y + np.array([1] * 7) * interval1_y + np.array([2]*7) * interval2_y + np.array([4]*7) * interval3_y
# 
# ax.plot(value, x, 'go-', linewidth=8, ms=30, alpha=0.5, label="ICDF $o$")
# # ax.plot(value_x_gt, x_gt, 'b--', linewidth=8, ms=30,  alpha=0.5,label="ICDF-GT $x$")
# ax.plot(value, y, 'r^-', linewidth=8, ms=30,  alpha=0.5,label="ICDF $y$")
# ax.plot(value_y_gt, y_gt, 'r--', linewidth=8, ms=30,  alpha=0.5,label="ICDF-GT $y$")
# ax.legend(loc='best', fontsize=32.5)
# plt.tick_params(labelsize=35)
# ax.set_xlabel('Cumulative Probability $p$', fontsize=35)
# ax.set_ylabel('Values', fontsize=35)
# my_y_ticks = np.arange(0,5,1) #* 0.1
# plt.yticks(my_y_ticks)
# # my_x_ticks = np.array([0, 1/6, 2/6, 3/6, 4/6, 5/6])#np.arange(0,4,1) * 0.1
# plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], [ r'$0.2$', r'$0.4$', r'$0.6$', r'$0.8$', r'$1$'])
# image_name = 'inverse_cdf_output.pdf'
# f.savefig(image_name)


############################################################## test torch.unique
# import torch
# import time
# a = torch.rand(6).normal_(0,1)
# b = torch.rand(6).normal_(10,1)
#
# a[0] = a[3]
# a_array = np.array(a)
# b_array = np.array(b)
# src_values, src_unique_indices, src_counts = np.unique(a_array.ravel(),
#                                                            return_inverse=True,
#                                                            return_counts=True)
# print(a_array)
# print(b_array)
# print(src_values)
# print(src_unique_indices)
# print(src_counts)
# tmpl_values, tmpl_counts = np.unique(b_array.ravel(), return_counts=True)
# src_quantiles = np.cumsum(src_counts) / a_array.size
# tmpl_quantiles = np.cumsum(tmpl_counts) / b_array.size
#
# print(src_quantiles)
# print(tmpl_quantiles)
#
# interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
# # print('interp_a_values')
# print(interp_a_values)
# print(interp_a_values[src_unique_indices].reshape(a_array.shape))
#
# print(tmpl_values[np.searchsorted(tmpl_quantiles, src_quantiles, side='left')][src_unique_indices].reshape(a_array.shape))




# src_values, src_unique_indices, src_counts = torch.unique(a.ravel(),
#                                                            return_inverse=True,
#                                                            return_counts=True)
# # print(a_array)
# # print(src_values)
# print(src_unique_indices)
# print(src_counts)
# tmpl_values, tmpl_counts = torch.unique(b.ravel(), return_counts=True)
# src_quantiles = torch.cumsum(src_counts, 0) / a_array.size
# tmpl_quantiles = torch.cumsum(tmpl_counts, 0) / b_array.size
# ## Pytorch implementation
# def search_sorted(bin_locations, inputs, eps=-1e-6):
#     """
#     Searches for which bin an input belongs to (in a way that is parallelizable and amenable to autodiff)
#     """
#     bin_locations[..., -1] += eps
#     return torch.sum(
#         inputs[..., None] >= bin_locations,
#         dim=-1
#     ) - 1
# print(tmpl_values[search_sorted(tmpl_quantiles, src_quantiles)][src_unique_indices])





# print(a)
# print(b)
# start = time.time()
# # value_a, index_a = torch.sort(a)
# # value_b, _ =
# # inverse_index = index_a.argsort(-1)
# transformeed = torch.sort(b)[0].gather(-1,  torch.sort(a)[1].argsort(-1))
# print(time.time() - start)
# print(transformeed)
#
#
# a = np.array(a)
# b = np.array(b)
# start = time.time()
# transformeed = match_histograms(a, b, channel_axis=0)
# print(time.time() - start)
# print(transformeed)





# ################################### various order only
# resnet = [82.5]
# mean = [84.4]
# var = [84.0]
# style = [85.8]
# hist = [85.6]
# sort = [86.6]
#
# res_error = [0.6]
# mean_error = [0.5]
# var_error = [0.5]
# in_error = [0.5]
# hist_error = [0.6]
# sort_error = [0.4]
# error_params=dict(elinewidth=5,capsize=8)#设置误差标记参数
# # labels = ['Acc.(\%)']
# x = np.arange(1)
# total_width, n = 0.8, 6
# width = total_width / n
# x = x - (total_width - width) / 2
# # error_params=dict(elinewidth=2,capsize=5)#设置误差标记参数
#
# figsize = 13,7
# f, ax = plt.subplots(figsize=figsize)
# ax.bar(x, np.array(resnet), fc='c', yerr=res_error, error_kw=error_params, width=width, label ="ResNet50")
# ax.bar(x + width, np.array(mean), fc='y', yerr=mean_error, error_kw=error_params, width=width, label ="+ AdaMean")
# ax.bar(x + width * 2, np.array(var), fc='g', yerr=var_error, error_kw=error_params, width=width, label ="+ AdaStd")
# ax.bar(x + width * 3, np.array(style), fc='g', yerr=in_error, error_kw=error_params, width=width, label ="+ AdaIN", alpha=0.5)
# ax.bar(x + width * 4, np.array(hist), fc='b', yerr=hist_error, error_kw=error_params, width=width, label ="+ HM")
# ax.bar(x + width * 5, np.array(sort), fc='r', yerr=sort_error, error_kw=error_params, width=width, label ="+ EFDM")
# ax.legend(fontsize=25)
# ax.set_ylim([81.5,87.3])
# ax.set_xlim([-0.43,0.85])
# plt.xticks([])
# plt.tick_params(labelsize=20)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Acc.(%)', fontsize=20)
# image_name = 'various_order.pdf'
# f.savefig(image_name)
#
#











# ################################### discrimination
# resnet = [1.997]
# dann = [1.823]
# auxselftrain = [1.988]
#
# labels = ['$\mathcal{A}_{dis}$']
# x = np.arange(1)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2
#
# f, ax = plt.subplots(1, 1)
# ax.bar(x, np.array(auxselftrain), fc='r', width=width, label ="AuxSelfTrain")
# ax.bar(x + width, np.array(resnet), fc='g', width=width, label ="ResNet34")
# ax.bar(x + width * 2, np.array(dann), fc='b', width=width, label ="DANN")
# ax.legend(fontsize=20)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('$\mathcal{A}_{dis}$', fontsize=20)
# image_name = 'A_distance.pdf'
# f.savefig(image_name)

# #################### acc_score_and_A_distance
# figsize = 17,11
# f, ax = plt.subplots(figsize=figsize)
# rotation = np.array(list(range(0, 12))) * 5 + 2.5
#
# acc = [0.98785, 0.9855, 0.9810, 0.9697, 0.9516, 0.8975, 0.8473, 0.7612, 0.6919, 0.5838, 0.4967, 0.4274]
# score = [0.9908, 0.9906, 0.9868, 0.9792, 0.9685, 0.9473, 0.9218, 0.8936, 0.8727, 0.8267, 0.8129, 0.8011]
# a_dis = [0.0496, 0.3296, 0.6848, 1.05, 1.226, 1.414, 1.5744, 1.6689, 1.7456, 1.7776, 1.7904, 1.8640]
#
# ax.plot(rotation, acc, 'r', linewidth=8, label="Accuracy")
# ax.plot(rotation, score, 'g', linewidth=8, label="Max. probability")
# plt.tick_params(labelsize=25)
# ax2 = ax.twinx()
# ax2.plot(rotation, a_dis, 'b', linewidth=8, label="$\mathcal{A}_{dis}$")
#
#
# lines, labels = ax.get_legend_handles_labels()
# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines + lines2, labels + labels2, loc=8, fontsize=40)
#
# plt.tick_params(labelsize=25)
# ax.set_xlabel('Rotation Degrees', fontsize=25)
# ax.set_ylabel('Values of Acc. and Prob.', fontsize=25)
# ax2.set_ylabel('$\mathcal{A}_{dis}$', fontsize=25)
# image_name = 'acc_score_dis.pdf'
# f.savefig(image_name)

# ######## two discrimination in one graph
# auxselftrain = [1.7]
# resnet = [0.84]
# dann = [0.7]
# auxselftrain_2 = [0.1]
# resnet_2 = [0.13]
# dann_2 = [0.14]
# labels = ['$\max J(\mathbf{W})$', 'Average error rates']
# x = np.arange(1)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2
#
# f, ax = plt.subplots(1, 1)
# ax.bar(x - width * 1.5, np.array(auxselftrain), fc='r', width=width, label ="AuxSelfTrain")
# ax.bar(x - width * 0.5, np.array(resnet), fc='g', width=width, label ="ResNet34")
# ax.bar(x + width * 0.5, np.array(dann), fc='b', width=width, label ="DANN")
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=20)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Values of $\max J(\mathbf{W})$', fontsize=20)
# ax2 = ax.twinx()
# #labels = []
# x = np.arange(1)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2
# ax2.bar(x + width * 2.7, np.array(auxselftrain_2), fc='r', width=width, label ="AuxSelfTrain")
# ax2.bar(x + width * 3.7, np.array(resnet_2), fc='g', width=width, label ="ResNet50")
# ax2.bar(x + width * 4.7, np.array(dann_2), fc='b', width=width, label ="DANN")
# ax2.set_ylabel('Values of aver. error rates', fontsize=20)
# ax2.legend(loc=9,fontsize=20)
# image_name = 'discrimination.pdf'
# f.savefig(image_name)

# ################################### discrimination
# resnet = [2.71, 4.15]
# dann = [2.06, 5.77]
# auxselftrain = [4.57, 6.43]
#
# labels = ['Target', 'Source']
# x = np.arange(2)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2
#
# f, ax = plt.subplots(1, 1)
# ax.bar(x - width * 1.5, np.array(auxselftrain), fc='r', width=width, label ="AuxSelfTrain")
# ax.bar(x - width * 0.5, np.array(resnet), fc='g', width=width, label ="ResNet34")
# ax.bar(x + width * 0.5, np.array(dann), fc='b', width=width, label ="DANN")
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=20)
# ax.legend(loc='best',fontsize=20)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('$\max J(\mathbf{W})$', fontsize=20)
# image_name = 'j_w_discrimination.pdf'
# f.savefig(image_name)
#
# ################################### discrimination
# resnet = [19.68, 10.57]
# dann = [23.2, 3.04]
# auxselftrain = [11.28, 1.52]
#
# labels = ['Target', 'Source']
# x = np.arange(2)
# total_width, n = 0.8, 3
# width = total_width / n
# x = x - (total_width - width) / 2
#
# f, ax = plt.subplots(1, 1)
# ax.bar(x - width * 1.5, np.array(auxselftrain), fc='r', width=width, label ="AuxSelfTrain")
# ax.bar(x - width * 0.5 , np.array(resnet), fc='g', width=width, label ="ResNet34")
# ax.bar(x + width * 0.5, np.array(dann), fc='b', width=width, label ="DANN")
# ax.set_xticks(x)
# ax.set_xticklabels(labels, fontsize=20)
# ax.legend(loc='best',fontsize=20)
# # ax.set_xlabel('Methods', fontsize=20)
# ax.set_ylabel('Error Rate', fontsize=20)
# image_name = 'error_discrimination.pdf'
# f.savefig(image_name)


# ################## plot of the wasser distance
# figsize = 11,9
# f, ax = plt.subplots(figsize=figsize)
# rotation = np.array(list(range(1, 25)))
#
# st =     [36.08, 33.61, 33.84, 33.20, 33.17, 34.69, 34.48, 34.62, 36.23, 35.00, 36.09, 35.84, 35.28, 36.10, 36.57, 36.58, 36.94, 37.64, 36.17, 36.64, 36.23, 37.17, 36.94, 36.49]
# consec = [27.13, 26.10, 25.24, 25.33, 25.45, 25.57, 25.19, 25.00, 25.74, 25.65, 26.35, 26.13, 25.97, 26.71, 27.29, 28.19, 27.80, 28.22, 28.13, 27.84, 28.34, 28.73, 28.07, 28.11]
# random = [28.31, 28.70, 28.83, 29.00, 29.41, 29.78, 29.04, 29.12, 29.44, 29.29, 29.12, 29.53, 29.36, 29.50, 29.65, 29.46, 29.94, 29.41, 29.34, 29.75, 29.44, 29.21, 29.19, 29.56]
# # intra =  [27.55, 25.69, 25.25, 26.09, 26.03, 26.48, 25.87, 25.61, 26.30, 26.47, 27.09, 26.43, 27.41, 27.62, 28.59, 27.91, 28.07, 26.79, 27.86, 27.54, 27.28, 28.67, 28.11, 28.00]
#
#
# print(len(rotation), len(st))
#
# ax.plot(rotation, st, 'g', linewidth=8, label="Between $\mathcal{S}$ and $\mathcal{T}$")
# ax.plot(rotation, random, 'b', linewidth=8, label="AuxSelfTrain (R)")
# ax.plot(rotation, consec, 'r', linewidth=8, label="AuxSelfTrain")
#
# # ax.plot(rotation, intra, '--c', linewidth=8, label="Intra-Domain $\mathcal{T}$")
# plt.tick_params(labelsize=10)
# plt.ylim(20, 40)
# ax.legend(loc='best', fontsize=22)
#
# plt.tick_params(labelsize=20)
# ax.set_xlabel('Index $m$ of Auxiliary Models', fontsize=25)
# ax.set_ylabel('W$_{\infty}$-based discrepancy', fontsize=25)
#
# image_name = 'wasser_mnist_distance_consective.pdf'
# f.savefig(image_name)



# x_list = [1,    3,    5,    7,    10,    15,   20,    30,  40,   50,   60]
# y_list = [50.2, 55.2, 57.3, 58.6, 59.1, 59.3, 60.2, 60.7, 60.8, 60.9, 61.1]
#
#
# f, ax = plt.subplots(1, 1)
# ax.plot(np.array(x_list), np.array(y_list), 'r', linewidth=8)
# # ax.legend(fontsize=30)
# ax.set_xlabel('Values of M', fontsize=30)
# ax.set_ylabel('Accuracy', fontsize=30)
# image_name = 'variousM.pdf'
# f.savefig(image_name)


# ##################### domain divergence between consecutive domains, OfficeHome
# figsize = 11,9
# f, ax = plt.subplots(figsize=figsize)
# rotation = np.array(list(range(1, 100)))
#
# acc = []
# score = []
# for i in range(99):
#     zero_one = random.random()
#     p_or_n = random.random()
#     if p_or_n > 0.5:
#         zero_one = zero_one * 0.03
#     else:
#         zero_one = -zero_one * 0.03
#     acc.append(1.7 + zero_one)
#
# for i in range(99):
#     zero_one = random.random()
#     p_or_n = random.random()
#     if p_or_n > 0.5:
#         zero_one = zero_one * 0.02
#     else:
#         zero_one = -zero_one * 0.02
#     score.append(0.3 + zero_one)
#
# print(len(rotation), len(acc), len(score))
#
# ax.plot(rotation, acc, 'r', linewidth=8, label="Between $\mathcal{S}$ and $\mathcal{T}$")
# ax.plot(rotation, score, 'g', linewidth=8, label="Between Consecutive Domains")
# plt.tick_params(labelsize=25)
#
# ax.legend(loc='best', fontsize=30)
#
# plt.tick_params(labelsize=25)
# ax.set_xlabel('Auxiliary Models', fontsize=25)
# ax.set_ylabel('$\mathcal{A}_{dis}$', fontsize=25)
#
# image_name = 'officehome_distance_consective.pdf'
# f.savefig(image_name)


# ##################### domain divergence between consecutive domains
# figsize = 11,9
# f, ax = plt.subplots(figsize=figsize)
# rotation = np.array(list(range(1, 24)))
#
# acc = [1.333, 1.371, 1.312, 1.334, 1.328, 1.328, 1.315, 1.310, 1.300, 1.3216, 1.352, 1.3328, 1.312, 1.310, 1.344, 1.362, 1.310, 1.314, 1.291, 1.347, 1.320, 1.266, 1.257]
# score = [0.08, 0.069, 0.0624, 0.0544, 0.0672, 0.072, 0.088, 0.078, 0.0816, 0.0256, 0.0544, 0.0384, 0.0864, 0.0992, 0.0336, 0.0600, 0.0768, 0.1090, 0.0544, 0.0672, 0.1312, 0.072, 0.088,]
# score_R = [0.09, 0.098, 0.053, 0.0774, 0.0832, 0.065, 0.078, 0.088, 0.0716, 0.0456, 0.0644, 0.0784, 0.0964, 0.0792, 0.0536, 0.0734, 0.0818, 0.0990, 0.0644, 0.0872, 0.1012, 0.092, 0.108,]
# print(len(rotation), len(acc), len(score))
#
# ax.plot(rotation, acc, 'g', linewidth=8, label="Between $\mathcal{S}$ and $\mathcal{T}$")
# ax.plot(rotation, score_R, 'b', linewidth=8, label="AuxSelfTrain (R)")
# ax.plot(rotation, score, 'r', linewidth=8, label="AuxSelfTrain")
# plt.tick_params(labelsize=25)
#
# ax.legend(loc='best', fontsize=30)
#
# plt.tick_params(labelsize=25)
# ax.set_xlabel('Index $m$ of Auxiliary Models', fontsize=25)
# ax.set_ylabel('$\mathcal{A}_{dis}$', fontsize=25)
#
# image_name = 'mnist_distance_consective.pdf'
# f.savefig(image_name)


#################################### sample of selection, bar illustration
# sx_list = [0, 5, 10, 15, 20, 25]
# sy_list = [0, 10, 20, 30, 40, 50]
# x_list = [30, 35, 40, 45, 50, 55]
# y_list = [60, 50, 40, 30, 20, 10]
#
#
# f, ax = plt.subplots(1, 1)
# ax.bar(np.array(x_list), np.array(y_list), fc='r', width=4, label ="Target")
# ax.bar(np.array(sx_list), np.array(sy_list), fc='b', width=4, label ="Source")
# ax.legend(fontsize=20)
# ax.set_xlabel('Rotation Degrees', fontsize=20)
# ax.set_ylabel('Number of Samples', fontsize=20)
# image_name = str(10) + 'score_rotation_' + str(10) + 'to' + str(30) + '.pdf'
# f.savefig(image_name)




# f, ax = plt.subplots(1, 1)
# x = np.array(list(range(5, 60)))
# y = np.array(list(range(5, 60)))
#
# print(x.shape)
# print(y.shape)
#
# ax.plot(x, y, 'r', linewidth=4)
# ax.set_xlabel('Evolving Process', fontsize=26)
# ax.set_ylabel('Rotation Degrees', fontsize=26)
# image_name = 'linear_shift_vs_time.png'
# f.savefig(image_name)
#
#
#
# f, ax = plt.subplots(1, 1)
# x = np.array(list(range(5, 60)))
# # y = np.array(list(range(0, 61)))
# y = (x) * 3 / 2
# for i in range(len(y)):
#     if y[i] > 60:
#         y[i] = 120 - y[i]
# print(y)
# # y = np.sqrt((70 * x - x * x) * 144 / 49)
#
# ax.plot(x, y, 'r', linewidth=4)
# ax.set_xlabel('Evolving Process', fontsize=26)
# ax.set_ylabel('Rotation Degrees', fontsize=26)
# image_name = 'curve_shift_vs_time.png'
# f.savefig(image_name)


# ##################### group meeting toy
# figsize = 11,9
# f, ax = plt.subplots(figsize=figsize)
# rotation = np.array(list(range(1, 7)))
#
# acc = [0.5, 0.8, 0.85, 0.83, 0.8, 0.78]
#
#
# print(len(rotation), len(acc))
#
# ax.plot(rotation, acc, 'r', linewidth=8, label="ACC")
#
# plt.tick_params(labelsize=25)
#
# ax.legend(loc='best', fontsize=30)
#
# plt.tick_params(labelsize=25)
# ax.set_xlabel('epochs', fontsize=25)
# ax.set_ylabel('ACC', fontsize=25)
#
# image_name = 'toy.pdf'
# f.savefig(image_name)