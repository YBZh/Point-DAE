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

project_root = '/Users/zhangyabin/Nutstore Files/我的坚果云/paper draft/PointCloud/PointDAE_CVPR2023/images/'

corruptions = ['No-Corruption', 'Drop-Global', 'Scan', 'Drop-Patch', 'Drop-Local', 'Add-Global', 'Add-Local', \
               'Jitter', 'Shear', 'Rotate-Z', 'Scale', 'Translate', 'Reflect', 'Rotate', 'Affine', \
               'Affine+Drop-Patch', 'Affine+Drop-Local']

name='dgcnn_objbg_dataset_variants'
random = [74.4, 74.2, 73.9, 75.1, 76.1, 72.8, 74.1, 73.6, 76.1, 72.0, 76.2, 78.4, 78.6, 76.2, 80.4, 80.4, 80.6]  ## Add local 76.2 ???  Second clolum 74.9??
svd    = [75.0, 74.3, 76.5, 75.3, 79.4, 74.2, 74.3, 73.8, 77.0, 76.7, 79.9, 78.7, 80.2, 77.8, 81.4, 82.3, 82.4]  ## affine <--> affine + Drop Patch.
man    = [77.4, 76.4, 78.9, 77.1, 79.5, 77.3, 77.2, 78.0, 79.4, 79.2, 80.4, 80.7, 82.8, 82.8, 84.4, 84.8, 85.5]  ## affine <--> affine + Drop Patch.
min_value = min(random)
max_value = max(man)
######## NO supervised learning adopted here. Totally 17.
x = np.arange(17)
total_width, n = 0.85, 3
width = total_width / n
x = x - (total_width - width) / 3
figsize = 30,7
f, ax = plt.subplots(figsize=figsize)
ax.bar(x, np.array(random), fc='g', width=width, label ="Random-Pose")
ax.bar(x + width, np.array(svd), fc='b', width=width, label ="SVD-Pose")
ax.bar(x + 2 * width, np.array(man), fc='r', width=width, label ="Manual-Pose")
ax.legend(fontsize=20)
plt.tick_params(labelsize=20)
ax.set_ylim([min_value*0.99, max_value*1.005])
ax.set_xlim([-0.5, 16.8])
plt.xticks(x+2*width, corruptions, fontsize=17, rotation=13, ha='right', rotation_mode='anchor')
ax.set_ylabel('Acc.(%)', fontsize=20)
image_name = project_root + name + '.pdf'
f.savefig(image_name)


########################################################################################################################################################
name='dgcnn_objhard_dataset_variants'
random = [64.2, 63.6, 64.6, 65.7, 65.6, 63.2, 64.9, 65.0, 65.7, 60.5, 65.5, 68.2, 64.0, 62.6, 68.7, 68.8, 69.3] ## Translate 68.2 ??
svd    = [65.6, 65.5, 66.0, 66.6, 68.9, 65.3, 65.9, 66.1, 67.0, 63.8, 68.9, 67.9, 67.0, 66.5, 71.7, 70.8, 73.9] ## + drop local leads to bad results.
man    = [66.4, 66.6, 66.7, 67.8, 68.9, 66.3, 68.1, 67.7, 70.0, 68.9, 70.7, 69.8, 70.8, 72.3, 74.7, 74.9, 76.8]
min_value = min(random)
max_value = max(man)
######## NO supervised learning adopted here. Totally 17.
x = np.arange(17)
total_width, n = 0.85, 3
width = total_width / n
x = x - (total_width - width) / 3
figsize = 30,7
f, ax = plt.subplots(figsize=figsize)
ax.bar(x, np.array(random), fc='g', width=width, label ="Random-Pose")
ax.bar(x + width, np.array(svd), fc='b', width=width, label ="SVD-Pose")
ax.bar(x + 2 * width, np.array(man), fc='r', width=width, label ="Manual-Pose")
ax.legend(fontsize=20)
plt.tick_params(labelsize=20)
ax.set_ylim([min_value*0.99, max_value*1.005])
ax.set_xlim([-0.5, 16.8])
plt.xticks(x+2*width, corruptions, fontsize=17, rotation=13, ha='right', rotation_mode='anchor')
ax.set_ylabel('Acc.(%)', fontsize=20)
image_name = project_root + name + '.pdf'
f.savefig(image_name)


########################################################################################################################################################
name='dgcnn_modelnet_dataset_variants'
random = [88.5, 88.7, 88.8, 89.1, 89.4, 88.3, 88.9, 88.8, 89.2, 88.1, 89.1, 89.9, 88.0, 87.6, 89.5, 89.6, 90.1]
svd    = [89.0, 88.7, 89.4, 89.4, 89.9, 88.8, 88.9, 89.0, 89.2, 89.0, 90.1, 90.1, 89.3, 89.2, 90.8, 90.1, 90.7]
man    = [89.0, 89.1, 89.4, 89.4, 90.2, 89.0, 89.6, 89.3, 89.9, 90.8, 90.7, 90.3, 90.2, 90.5, 91.0, 90.7, 91.2]
min_value = min(random)
max_value = max(man)
######## NO supervised learning adopted here. Totally 17.
x = np.arange(17)
total_width, n = 0.85, 3
width = total_width / n
x = x - (total_width - width) / 3
figsize = 30,7
f, ax = plt.subplots(figsize=figsize)
ax.bar(x, np.array(random), fc='g', width=width, label ="Random-Pose")
ax.bar(x + width, np.array(svd), fc='b', width=width, label ="SVD-Pose")
ax.bar(x + 2 * width, np.array(man), fc='r', width=width, label ="Manual-Pose")
ax.legend(fontsize=20)
plt.tick_params(labelsize=20)
ax.set_ylim([min_value*0.99, max_value*1.005])
ax.set_xlim([-0.5, 16.8])
plt.xticks(x+2*width, corruptions, fontsize=17, rotation=13, ha='right', rotation_mode='anchor')
ax.set_ylabel('Acc.(%)', fontsize=20)
image_name = project_root + name + '.pdf'
f.savefig(image_name)


########################################################################################################################################################
name='pointnetv2_objhard_dataset_variants'
random = [62.9, 63.0, 61.9, 66.3, 65.7, 64.6, 63.2, 63.2, 63.5, 60.2, 65.1, 66.4, 63.0, 58.9, 68.3, 69.8, 71.1]
svd    = [63.7, 64.2, 64.2, 67.5, 67.0, 64.4, 64.3, 63.2, 64.5, 64.3, 65.7, 67.0, 64.7, 65.7, 70.1, 72.4, 73.2]
man    = [65.1, 64.8, 64.8, 67.6, 68.1, 66.4, 64.3, 64.5, 66.6, 66.0, 66.8, 68.0, 70.4, 72.0, 74.0, 76.3, 76.8]
min_value = min(random)
max_value = max(man)
######## NO supervised learning adopted here. Totally 17.
x = np.arange(17)
total_width, n = 0.85, 3
width = total_width / n
x = x - (total_width - width) / 3
figsize = 30,7
f, ax = plt.subplots(figsize=figsize)
ax.bar(x, np.array(random), fc='g', width=width, label ="Random-Pose")
ax.bar(x + width, np.array(svd), fc='b', width=width, label ="SVD-Pose")
ax.bar(x + 2 * width, np.array(man), fc='r', width=width, label ="Manual-Pose")
ax.legend(fontsize=20)
plt.tick_params(labelsize=20)
ax.set_ylim([min_value*0.99, max_value*1.005])
ax.set_xlim([-0.5, 16.8])
plt.xticks(x+2*width, corruptions, fontsize=17, rotation=13, ha='right', rotation_mode='anchor')
ax.set_ylabel('Acc.(%)', fontsize=20)
image_name = project_root + name + '.pdf'
f.savefig(image_name)


########################################################################################################################################################
name='pointnetnot_objhard_dataset_variants'
random = [59.5, 59.9, 59.1, 59.8, 57.4, 55.9, 59.1, 60.6, 59.9, 59.1, 60.7, 58.9, 61.6, 59.2, 61.9, 62.4, 62.1]
svd    = [60.3, 60.9, 61.0, 60.3, 59.6, 58.3, 60.3, 60.7, 61.2, 60.8, 61.0, 60.2, 62.6, 61.7, 62.2, 62.9, 62.3]
man    = [62.0, 62.1, 61.9, 61.1, 59.6, 58.4, 62.0, 62.6, 62.3, 60.9, 64.1, 61.9, 63.7, 62.3, 63.6, 63.2, 62.5]
min_value = min(random)
max_value = max(man)
######## NO supervised learning adopted here. Totally 17.
x = np.arange(17)
total_width, n = 0.85, 3
width = total_width / n
x = x - (total_width - width) / 3
figsize = 30,7
f, ax = plt.subplots(figsize=figsize)
ax.bar(x, np.array(random), fc='g', width=width, label ="Random-Pose")
ax.bar(x + width, np.array(svd), fc='b', width=width, label ="SVD-Pose")
ax.bar(x + 2 * width, np.array(man), fc='r', width=width, label ="Manual-Pose")
ax.legend(fontsize=20)
plt.tick_params(labelsize=20)
ax.set_ylim([min_value*0.99, max_value*1.01])
ax.set_xlim([-0.5, 16.8])
plt.xticks(x+2*width, corruptions, fontsize=17, rotation=13, ha='right', rotation_mode='anchor')
ax.set_ylabel('Acc.(%)', fontsize=20)
image_name = project_root + name + '.pdf'
f.savefig(image_name)
