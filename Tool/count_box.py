'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-09 10:38:13
LastEditors: Leidi
LastEditTime: 2021-12-22 17:12:30
'''
import os
import json
from tqdm import tqdm
from pathlib import Path


file = r'/home/leidi/Downloads/数据堂第一交付最终数据'
image_count = 0
box_count = 0

for root, dirs, files in tqdm(os.walk(file)):
    for n in files:
        file = Path() / root / n
        if file.suffix == '.json':
            image_count += 1
            with open(file) as f:
                data = json.load(f)
                box_count += len(data['dataList'])
print('Total images: {}'.format(image_count))
print('Total boxs: {}'.format(box_count))
