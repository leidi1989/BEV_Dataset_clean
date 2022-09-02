'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-09 15:55:45
LastEditors: Leidi
LastEditTime: 2021-10-17 16:55:54
'''
from pathlib import Path
import shutil
import os

folder = r'/home/leidi/Desktop/test/whu_cam_front_center_20211015'

for root, dirs, files in os.walk(folder):
    for n in files:
        file = Path() / root / n
        if file.suffix == '.jpg' or file.suffix == '.png':
            rename = os.path.join(root, file.stem.replace('.', '') + file.suffix)
            file.replace(rename)
