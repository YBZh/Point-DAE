import os

# path = '/home/yabin/syn_project/point_cloud/CorruptedAE/experiments/'
#
# directory_list = os.listdir(path)
#
# for filename in directory_list:
#     src = filename
#     dst = filename.replace('Point_CAE_PointNetv2', "_svdPoint_CAE_PointNetv2")
#
#     print(dst)
#
#     os.rename(os.path.join(path, src), os.path.join(path, dst))
#
# print("File renamed!")

path = '/home/yabin/syn_project/point_cloud/CorruptedAE/experiments/finetune_scan_objbg_svm_classification_cleanPointNetv2_feat/cfgs/'

directory_list = os.listdir(path)

for filename in directory_list:
    src = filename
    dst = filename + '_svd'

    print(dst)

    os.rename(os.path.join(path, src), os.path.join(path, dst))

print("File renamed!")