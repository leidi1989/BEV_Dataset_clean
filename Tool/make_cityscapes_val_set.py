'''
Description: 
Version: 
Author: Leidi
Date: 2021-09-18 10:46:15
LastEditors: Leidi
LastEditTime: 2021-10-17 16:56:25
'''
import os
import json
import argparse
from cv2 import putText
from tqdm import tqdm
from utils.utils import check_output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='make_cityscapes_val_set.py')
    parser.add_argument('--root', default=r'/home/leidi/Desktop/test/whu_cam_front_center_20211015',
                        type=str, help='image root')
    opt = parser.parse_args()
    
    val_image_floder = opt.root
    val_annotation_floder = check_output_path(os.path.join(val_image_floder, 'annotations'))

    print('Start create fake json:')
    for root, dirs, files in tqdm(os.walk(val_image_floder)):
        for n in files:
            fake_js = {}
            json_name = n.replace('.jpg', '.json')
            json_output_path = os.path.join(val_annotation_floder, json_name)
            json.dump(fake_js, open(json_output_path, 'w'))
