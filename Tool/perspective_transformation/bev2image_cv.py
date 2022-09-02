'''
Description: 
Version: 
Author: Leidi
Date: 2022-08-03 14:19:20
LastEditors: Leidi
LastEditTime: 2022-08-17 19:50:06
'''
import argparse
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
        '--camera_paras',
        '--cp',
        dest='camera_paras',
        default=r'Tool/data/hq1_calibration_result_20220817/extrinsics',
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
    for key in camera_paras.keys():
        camera_paras_folder_path = opt.camera_paras
        for camera_paras_file_name in os.listdir(camera_paras_folder_path):
            camera_paras_file = yaml.load(open(os.path.join(
                camera_paras_folder_path, camera_paras_file_name),
                                               'r',
                                               encoding="utf-8"),
                                          Loader=yaml.FullLoader)
            R_T_matrix = np.array(
                camera_paras_file['CameraExtrinsicMat']['data'])
            R_T_matrix.resize((4, 4))
            R_matrix = R_T_matrix[0:3, 0:3]
            R_matrix = np.linalg.inv(R_matrix)
            R_vect, _ = cv2.Rodrigues(R_matrix)
            T_matrix = R_T_matrix[0:3, 3]
            T_matrix[2] = 1.87
            Camera_mat = np.array(camera_paras_file['CameraMat']['data'])
            Camera_mat.resize((3, 3))
            DistCoeffs = np.array(camera_paras_file['DistCoeff']['data'],
                                  np.float32)
            camera_paras[key] = {
                'R_T_matrix': R_T_matrix,
                'R_matrix': R_matrix,
                'R_vect': R_vect,
                'T_matrix': T_matrix,
                'T_vect': T_matrix,
                'Camera_mat': Camera_mat,
                'DistCoeffs': DistCoeffs
            }
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
                    vehicle_point = np.array(vehicle_point)
                    image_point, a = cv2.projectPoints(
                        vehicle_point, camera_paras[key]['R_vect'],
                        camera_paras[key]['T_vect'],
                        camera_paras[key]['Camera_mat'],
                        camera_paras[key]['DistCoeffs'])
                    image_point = list(map(int, image_point[0][0]))
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
