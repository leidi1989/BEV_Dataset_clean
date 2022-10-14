'''
Description: 
Version: 
Author: Leidi
Date: 2022-08-03 14:19:20
LastEditors: Leidi
LastEditTime: 2022-10-14 11:32:06
'''
import argparse
import json
import math
import os
import sys
import time
import cv2

sys.path.append(os.path.abspath(os.curdir))
sys.path.append(os.path.abspath(os.curdir) + os.sep + 'Clean_up')

from Clean_up.base.image_base import OBJECT
import Clean_up.dataset
import numpy as np
import yaml
from Clean_up.utils.utils import *

from get_image import TEMP_LOAD


class CameraParas():

    def set_datas(self, paras):
        R = np.array(paras['R'], dtype=np.float64)
        self.R = R.T
        T = np.array([paras['T']], dtype=np.float64)
        self.T = T.T
        distortion_center = np.array([paras['distortion_center']],
                                     dtype=np.float64)
        self.distortion_center = distortion_center.T
        self.stretch_matrix = np.array(paras['stretch_matrix'],
                                       dtype=np.float64)
        self.a0 = paras['mapping_coefficients'][0]
        self.a2 = paras['mapping_coefficients'][1]
        self.a3 = paras['mapping_coefficients'][2]
        self.a4 = paras['mapping_coefficients'][3]

        return

    def to_image_pos(self, vehicle_x: float, vehicle_y: float,
                     vehicle_z: float) -> tuple:
        Pw = np.array([[vehicle_x, vehicle_y, vehicle_z]])
        Pc = np.dot(self.R, Pw.T) + self.T
        Xc = Pc[0, 0]
        Yc = Pc[1, 0]
        Zc = Pc[2, 0]
        ZZ = Xc * Xc + Yc * Yc
        C0 = self.a4 * ZZ * ZZ
        C1 = self.a3 * ZZ * math.sqrt(ZZ)
        C2 = self.a2 * ZZ
        C3 = -Zc
        C4 = self.a0
        p = np.poly1d([C4, C3, C2, C1, C0])
        roots = p.roots
        s = 0
        for root in roots:
            if root.real > 0 and math.fabs(root.imag) < 0.0000000001:
                s = root.real
                break
        if s == 0:
            return
        u = Xc / s
        v = Yc / s
        Puv = np.array([[u, v]])
        Puv_ = np.dot(self.stretch_matrix, Puv.T) + self.distortion_center

        return (Puv_[0, 0], Puv_[1, 0])


def read_json():
    bev2image_json_path = os.path.join(os.path.abspath(os.curdir), 'Tool',
                                       'python_version', 'bev2image_hq2.json')
    with open(bev2image_json_path, 'r', encoding='utf8') as fp:
        json_data = json.load(fp)
        camera_paras = json_data['camera_paras']
        left_front_paras = CameraParas()
        left_front_paras.set_datas(camera_paras['left_front'])
        front_right_paras = CameraParas()
        front_right_paras.set_datas(camera_paras['front_right'])
        right_front_paras = CameraParas()
        right_front_paras.set_datas(camera_paras['right_front'])
        left_back_paras = CameraParas()
        left_back_paras.set_datas(camera_paras['left_back'])
        back_paras = CameraParas()
        back_paras.set_datas(camera_paras['back'])
        right_back_paras = CameraParas()
        right_back_paras.set_datas(camera_paras['right_back'])

        return left_front_paras, front_right_paras, right_front_paras, left_back_paras, back_paras, right_back_paras


def get_vehicle_xyz(dataset_instance, object: OBJECT) -> list:

    box_rotated_rect_points_vehicle = []
    self_center = dataset_instance.self_position_parse(
        dataset_instance.camera_image_wh[1],
        dataset_instance.label_image_wh[0], dataset_instance.label_image_wh[1],
        dataset_instance.label_range[0], dataset_instance.label_range[1],
        dataset_instance.label_range[2], dataset_instance.label_range[3])
    # 距离像素换算关系 m/pixel
    dp_rate = dataset_instance.distance_pixel_rate(
        dataset_instance.label_range[2] + dataset_instance.label_range[3],
        dataset_instance.label_image_wh[0])

    for point in object.box_rotated_rect_points:
        vehicle_x = (self_center[1] - point[1]) * dp_rate
        vehicle_y = (self_center[0] - point[0]) * dp_rate
        vehicle_z = 0
        box_rotated_rect_points_vehicle.append(
            [vehicle_x, vehicle_y, vehicle_z])

    return box_rotated_rect_points_vehicle


if __name__ == '__main__':

    time_start = time.time()
    parser = argparse.ArgumentParser(prog='clean.py')
    parser.add_argument('--config',
                        '--c',
                        dest='config',
                        default=r'Clean_up/config/default.yaml',
                        type=str,
                        help='dataset config file path')
    parser.add_argument(
        '--workers',
        '--w',
        dest='workers',
        default=16,
        type=int,
        help='maximum number of dataloader workers(multiprocessing.cpu_count())'
    )

    opt = parser.parse_args()
    # load dataset config file
    dataset_config = yaml.load(open(opt.config, 'r', encoding="utf-8"),
                               Loader=yaml.FullLoader)
    dataset_config.update({'workers': opt.workers})

    Input_dataset = Clean_up.dataset.__dict__[
        dataset_config['Source_dataset_style']](dataset_config)

    camera_name = [
        'left_front', 'front_right', 'right_front', 'left_back', 'back',
        'right_back'
    ]
    # 读取内外参，创建透视图包围框列表字典
    camera_paras = {name: None for name in camera_name}
    for key, vaule in zip(camera_paras.keys(), read_json()):
        camera_paras[key] = vaule
    # 读取标注文件
    for json_name in os.listdir(Input_dataset.temp_annotations_folder):
        image_bbox_points_list_dict = {name: [] for name in camera_name}
        json_path = os.path.join(Input_dataset.temp_annotations_folder,
                                 json_name)
        image = TEMP_LOAD(Input_dataset, json_path)
        # 获取包围框图像坐标
        for object in image.object_list:
            object_rotated_rect_points_vehicle = get_vehicle_xyz(
                Input_dataset, object)
            for key, value in camera_paras.items():
                image_bbox_points = []
                for vehicle_point in object_rotated_rect_points_vehicle:
                    image_point = value.to_image_pos(vehicle_point[0],
                                                     vehicle_point[1],
                                                     vehicle_point[2])
                    image_point = list(map(int, image_point))
                    image_bbox_points.append(image_point)
                in_image = 1
                for image_point in image_bbox_points:
                    if 0 > image_point[0] or \
                        image_point[0] > 1280 or \
                            0 > image_point[1] or \
                                image_point[1] > 720:
                        in_image = 0
                if in_image:
                    image_bbox_points_list_dict[key].append(image_bbox_points)

        # 获取数据集中各个摄像头图像
        each_camera_source_image_dict = {name: None for name in camera_name}
        img = cv2.imread(image.image_path)
        camera_img = img[0:Input_dataset.camera_image_wh[1],
                         0:Input_dataset.camera_image_wh[0]]
        camera_img = cv2.resize(camera_img, (1280 * 3, 720 * 2))
        each_camera_source_image_dict['left_front'] = camera_img[0:720, 0:1280]
        each_camera_source_image_dict['front_right'] = camera_img[0:720,
                                                                  1280:1280 *
                                                                  2]
        each_camera_source_image_dict['right_front'] = camera_img[0:720, 1280 *
                                                                  2:1280 * 3]

        each_camera_source_image_dict['left_back'] = camera_img[720:720 * 2,
                                                                0:1280]
        each_camera_source_image_dict['back'] = camera_img[720:720 * 2,
                                                           1280:1280 * 2]
        each_camera_source_image_dict['right_back'] = camera_img[720:720 * 2,
                                                                 1280 *
                                                                 2:1280 * 3]

        each_camera_mask_image_dict = {name: None for name in camera_name}
        for key, _ in each_camera_mask_image_dict.items():
            mask = np.zeros((720, 1280, 3), np.uint8)
            for points_list in image_bbox_points_list_dict[key]:
                points_list = np.array(points_list)
                cv2.fillConvexPoly(mask, points_list, (0, 255, 0))
            each_camera_mask_image_dict[key] = mask
        for key, _ in each_camera_source_image_dict.items():
            image_addWeighted = np.zeros((720, 1280, 3), np.uint8)
            cv2.addWeighted(each_camera_source_image_dict[key], 1,
                            each_camera_mask_image_dict[key], 0.5, 0,
                            image_addWeighted)
            each_camera_source_image_dict[key] = image_addWeighted
        total_mask_image = np.zeros((720 * 2, 1280 * 3, 3), np.uint8)
        total_mask_image[0:720,
                         0:1280] = each_camera_source_image_dict['left_front']
        total_mask_image[0:720, 1280:1280 *
                         2] = each_camera_source_image_dict['front_right']
        total_mask_image[0:720, 1280 * 2:1280 *
                         3] = each_camera_source_image_dict['right_front']

        total_mask_image[720:720 * 2,
                         0:1280] = each_camera_source_image_dict['left_back']
        total_mask_image[720:720 * 2,
                         1280:1280 * 2] = each_camera_source_image_dict['back']
        total_mask_image[720:720 * 2, 1280 * 2:1280 *
                         3] = each_camera_source_image_dict['right_back']
        total_mask_image = cv2.resize(total_mask_image, (1280, 720))
        cv2.imshow('total_mask_image', total_mask_image)
        cv2.waitKey(0)
