'''
Description: 
Version: 
Author: Leidi
Date: 2022-10-26 16:14:14
LastEditors: Leidi
LastEditTime: 2022-10-27 19:22:38
'''
import os
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

source_annotations_path = os.path.join('/mnt/data_1/Dataset/dataset_temp',
                                       'hefei_lukou_0-44.xodr')
tree = ET.parse(source_annotations_path)
root = tree.getroot()

header = root.find('header')

road_object_inertial_point_dict = {}

road_inertial_object = {}
for road_id, road in enumerate(root.findall('road')):
    road_planView = road.find('planView')
    road_planView_geo = road_planView.find('geometry')
    rline_to_inertial_R = R.from_euler(
        'zyx', list(map(float,
                        [road_planView_geo.attrib['hdg'], 0, 0]))).as_matrix()
    rline_to_inertial_T = np.array(
        list(
            map(float, [
                road_planView_geo.attrib['x'], road_planView_geo.attrib['y'], 0
            ])))
    object_inertial_point_dict = {}
    for objects in road.findall('objects'):
        for object in objects.findall('object'):
            local_to_rline_R = R.from_euler(
                'zyx',
                list(
                    map(float, [
                        object.attrib['hdg'], object.attrib['pitch'],
                        object.attrib['roll']
                    ]))).as_matrix()
            local_to_rline_T = np.array(
                list(
                    map(float, [
                        object.attrib['s'], object.attrib['t'],
                        object.attrib['zOffset']
                    ])))
            object_outline = object.find('outline')
            cornerLocal_inertial_point_list = []
            for cornerLocal in object_outline.findall('cornerLocal'):
                cornerLocal_point = np.array(
                    list(
                        map(float, [
                            cornerLocal.attrib['u'], cornerLocal.attrib['v'],
                            cornerLocal.attrib['z']
                        ])))
                inertial_point = np.dot(
                    rline_to_inertial_R,
                    np.dot(local_to_rline_R, cornerLocal_point) +
                    local_to_rline_T) + rline_to_inertial_T
                cornerLocal_inertial_point_list.append(inertial_point)

            object_inertial_point_dict.update(
                {object.attrib['id']: cornerLocal_inertial_point_list})

    road_object_inertial_point_dict.update(
        {road.attrib['name']: object_inertial_point_dict})
print(0)
