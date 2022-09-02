'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2022-08-24 14:01:20
'''
from PIL import Image


image_path = r'/home/leidi/Pictures/wuda_baishazhou@003570_id.png'
# image_path = r'/mnt/data_2/Dataset/Autopilot_bev_dataset/others/hy_bev_wd@jsj_2hz_hq1_6v_020304050607_80_80_80_80_multi_class_20220517/0428_BEV_6V_hongqi1_wuda_jsjTols3_0506_0_0504_020465.jpg'

image = Image.open(image_path)
# image = cv2.resize(image, (1280, 720))
image.show()
