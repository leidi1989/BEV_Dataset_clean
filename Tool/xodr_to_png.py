'''
Description: 
Version: 
Author: Leidi
Date: 2022-10-26 16:14:14
LastEditors: Leidi
LastEditTime: 2022-10-26 20:03:57
'''
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

source_annotations_path = os.path.join('/mnt/data_1/Dataset/dataset_temp',
                                       'hefei_lukou_73-83(1.4).xodr')
tree = ET.parse(source_annotations_path)
root = tree.getroot()

header = root.find('header')

road_planView_geo = 0
road_object_dict = {}
object_dict = {}

for road_id, road in enumerate(root.findall('road')):
    road_planView = road.find('planView')
    road_planView_geo = road_planView.find('geometry')
    for objects in road.findall('objects'):
        for object in objects.findall('object'):
            object_outline = object.find('outline')
            for cornerLocal in object_outline.findall('cornerLocal'):

                print(0)


def EulerAngles2RotationMatrix(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.基于右手系
    param theta: 1-by-3 list [rx, ry, rz] angle in degree
        theta[0]: roll          绕定轴X转动    
        theta[1]: pitch      绕定轴Y转动
        theta[2]: yaw        绕定轴Z转动  
        
    return:
        YPR角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]

    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])

    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])

    return R_x, R_y, R_z
