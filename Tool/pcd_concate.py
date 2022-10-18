'''
Description: 
Version: 
Author: Leidi
Date: 2022-10-18 11:13:38
LastEditors: Leidi
LastEditTime: 2022-10-18 16:31:20
'''
import os
import open3d as o3d
import numpy as np
from tqdm import tqdm

pcd_folder = '/home/user/leidi/data/PCD/hefei_lukou/6'
pcd_concate_output='/home/user/leidi/data/PCD/hefei_lukou/concate/6.pcd'

def read_pcd(file_path):
	pcd = o3d.io.read_point_cloud(file_path)
	print(np.asarray(pcd.points))
	colors = np.asarray(pcd.colors) * 255
	points = np.asarray(pcd.points)
	print(points.shape, colors.shape)
 
	return np.concatenate([points, colors], axis=-1)

total_pcd = None
for pcd_file in tqdm(os.listdir(pcd_folder)):
    pcd_file_path = os.path.join(pcd_folder, pcd_file)
    pcd = o3d.io.read_point_cloud(pcd_file_path)
    if total_pcd is None:
        total_pcd = pcd
    else:
        total_pcd+=pcd
        
o3d.io.write_point_cloud(pcd_concate_output, total_pcd)