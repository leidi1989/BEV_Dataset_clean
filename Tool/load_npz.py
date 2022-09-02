'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2022-07-06 10:58:38
'''
import cv2
import numpy as np

npz_path = r'/mnt/data_1/Dataset/dataset_temp/bev_6v_hongqi1_wudatohjh_0507_20220609_n_classes_20220705/CROSSVIEW/dynamic_gt/test@015005.npz'
# image_path = r'/mnt/data_2/Dataset/Autopilot_bev_dataset/others/hy_bev_wd@jsj_2hz_hq1_6v_020304050607_80_80_80_80_multi_class_20220517/0428_BEV_6V_hongqi1_wuda_jsjTols3_0506_0_0504_020465.jpg'

tenosr = np.load(npz_path)
print(tenosr['data'])

cv2.imshow('one', tenosr['data'])
cv2.waitKey(0)
