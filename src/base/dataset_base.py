'''
Description:
Version:
Author: Leidi
Date: 2022-01-07 11:00:30
LastEditors: Leidi
LastEditTime: 2022-09-18 17:12:32
'''
import ftplib
import json
import math
import multiprocessing
import os
import random
import shutil

import cv2
import dataset
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils import image_form_transform
from utils.utils import *

from .image_base import IMAGE, OBJECT, LANELINE

matplotlib.rc("font", family='AR PL UMing CN')
plt.switch_backend('agg')
# matplotlib.use('TkAgg')

SOURCE_DATASET_STYLE = [
    'CVAT_IMAGE_BEV_GROUP', 'CVAT_IMAGE_BEV_1', 'CVAT_IMAGE_BEV_2',
    'CVAT_IMAGE_BEV_NAS', 'COCO2017', 'YUNCE_SEGMENT_COCO',
    'YUNCE_SEGMENT_COCO_ONE_IMAGE', 'HUAWEIYUN_SEGMENT', 'HY_VAL', 'BDD100K',
    'YOLO', 'TT100K', 'CCTSDB', 'LISA'
]

TARGET_DATASET_STYLE = [
    'PYVA', 'CVT', 'YOLO', 'COCO2017', 'CITYSCAPES', 'CITYSCAPES_VAL',
    'CVAT_IMAGE_1_1'
]

TARGET_DATASET_FILE_FORM = {
    'CVAT_IMAGE_1_1': {
        'image': 'jpg',
        'annotation': 'xml'
    },
    'CITYSCAPES_VAL': {
        'image': 'png',
        'annotation': 'json'
    },
    'COCO2017': {
        'image': 'jpg',
        'annotation': 'json'
    },
    'CITYSCAPES': {
        'image': 'png',
        'annotation': 'json'
    },
    'YOLO': {
        'image': 'jpg',
        'annotation': 'txt'
    },
    'NUSCENES': {
        'image': 'jpg',
        'annotation': 'json'
    },
    'PYVA': {
        'image': 'jpg',
        'annotation': 'json'
    },
    'CVT': {
        'image': 'jpg',
        'annotation': 'json'
    }
}


class Dataset_Base:
    """[数据集基础类]
    """

    def __init__(self, dataset_config: dict) -> None:
        """[数据集基础类]

        Args:
            dataset_config (dict): [数据集配置信息字典]
        """
        print('Start dataset instance initialize:')
        # Source_dataset
        self.save_in_nas = dataset_config['Save_in_nas']
        if self.save_in_nas:
            # ftp
            self.ftp_ip = dataset_config['Nas_ip']
            self.username = dataset_config['Nas_username']
            self.password = dataset_config['Nas_password']
            self.ftp = ftplib.FTP()
            self.ftp.set_pasv(False)
            self.ftp.connect(self.ftp_ip, 21)
            self.ftp.login(self.username, self.password)
            print(self.ftp.welcome)
            self.dataset_input_folder = None
            self.nas_dataset_input_folder = dataset_config[
                'Dataset_input_folder']
        else:
            # local
            self.dataset_input_folder = check_input_path(
                dataset_config['Dataset_input_folder'])
        self.source_dataset_style = dataset_config['Source_dataset_style']
        self.source_dataset_image_form_list = None
        self.source_dataset_images_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'],
                         'source_dataset_images'))
        self.source_dataset_annotation_form = None
        self.source_dataset_annotations_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'],
                         'source_dataset_annotations'))
        self.source_dataset_image_count = None
        self.source_dataset_annotation_count = None
        self.task_dict = {
            'Detection': None,
            'Semantic_segmentation': None,
            'Instance_segmentation': None,
            'Keypoints': None
        }
        self.annotation_car = dataset_config['Annotation_car']
        self.draw_car_mask = dataset_config['Draw_car_mask']

        # MAP
        self.get_local_map = dataset_config['Get_local_map']
        self.delete_no_map = dataset_config['Delete_no_map']
        self.lat_lon_origin_city = dataset_config['Lat_lon_origin_city']

        # File_prefix
        self.file_prefix_delimiter = '@' if dataset_config['File_prefix_delimiter'] == '' or \
            dataset_config['File_prefix_delimiter'] is None else \
            dataset_config['File_prefix_delimiter']
        self.file_prefix = check_prefix(
            dataset_config['File_prefix'],
            dataset_config['File_prefix_delimiter'])

        # temp dataset
        self.temp_image_form = TARGET_DATASET_FILE_FORM[
            dataset_config['Target_dataset_style']]['image']
        self.temp_annotation_form = 'json'
        self.temp_images_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'],
                         'source_dataset_images'))
        self.temp_annotations_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'],
                         'temp_annotations'))
        self.temp_informations_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'],
                         'temp_infomations'))
        self.temp_sample_statistics_folder = check_output_path(
            os.path.join(self.temp_informations_folder, 'sample_statistics'))
        self.temp_divide_file_list = [
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'],
                             'temp_infomations'), 'total.txt'),
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'],
                             'temp_infomations'), 'train.txt'),
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'],
                             'temp_infomations'), 'test.txt'),
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'],
                             'temp_infomations'), 'val.txt'),
            os.path.join(
                os.path.join(dataset_config['Dataset_output_folder'],
                             'temp_infomations'), 'redund.txt')
        ]
        self.temp_set_name_list = [
            'total_distibution.txt', 'train_distibution.txt',
            'val_distibution.txt', 'test_distibution.txt',
            'redund_distibution.txt'
        ]
        self.temp_annotation_name_list = self.get_temp_annotations_name_list()
        self.temp_annotations_path_list = self.get_temp_annotations_path_list()
        self.temp_image_name_list = self.get_temp_images_name_list()

        # target dataset
        self.dataset_output_folder = check_output_path(
            dataset_config['Dataset_output_folder'])
        self.target_dataset_style = dataset_config['Target_dataset_style']
        self.target_dataset_image_form = TARGET_DATASET_FILE_FORM[
            dataset_config['Target_dataset_style']]['image']
        self.target_dataset_annotation_form = TARGET_DATASET_FILE_FORM[
            dataset_config['Target_dataset_style']]['annotation']
        self.target_dataset_annotations_folder = check_output_path(
            os.path.join(dataset_config['Dataset_output_folder'],
                         'target_dataset_annotations'))
        self.target_dataset_output_folder = ''

        # temp dataset information
        self.total_file_name_path = total_file(self.temp_informations_folder)
        self.target_dataset_view = dataset_config['Target_dataset_view']
        self.target_dataset_divide_proportion = tuple(
            float(x) for x in (
                dataset_config['Target_dataset_divide_proportion'].split(',')))
        self.target_dataset_output_two_class = dataset_config['Two_class']
        self.label_object_rotation_angle = dataset_config[
            'Label_object_rotation_angle']
        self.temp_divide_file_annotation_path_dict = {}
        self.camera_image_wh = [
            int(x) for x in (dataset_config['Camera_image_wh'].split(','))
        ]
        self.label_image_wh = [
            int(x) for x in (dataset_config['Label_image_wh'].split(','))
        ]
        self.label_range = [
            int(x) for x in (dataset_config['Label_range'].split(','))
        ]
        self.adjust_label_range = [
            int(x) for x in (dataset_config['Adjust_label_range'].split(','))
        ]
        self.semantic_segmentation_label_image_wh = [
            int(x) for x in
            (dataset_config['Semantic_segmentation_label_image_wh'].split(','))
        ]
        self.keep_no_object = dataset_config['Keep_no_object']
        self.label_image_self_car_uv = [
            int(self.label_image_wh[0] *
                (self.label_range[2] /
                 (self.label_range[2] + self.label_range[3]))),
            int(self.label_image_wh[1] *
                (self.label_range[0] /
                 (self.label_range[0] + self.label_range[1])))
        ]

        # 声明set类别统计pandas字典
        self.temp_divide_object_count_dataframe_dict = {}
        self.temp_divide_each_class_pixel_count_dataframe_dict = {}
        self.temp_divide_each_pixel_proportion_dataframe_dict = {}

        self.temp_merge_class_list = {
            'Merge_source_dataset_class_list': [],
            'Merge_target_dataset_class_list': []
        }

        # target check
        self.target_dataset_annotations_check_count = dataset_config[
            'Target_dataset_check_annotations_count']
        self.target_dataset_annotation_check_output_folder = check_output_path(
            os.path.join(self.temp_informations_folder, 'check_annotation'))
        self.target_dataset_check_file_name_list = None
        self.target_dataset_check_images_list = None
        self.target_dataset_annotation_check_mask = dataset_config[
            'Target_dataset_check_annotations_output_as_mask']

        # others
        self.workers = dataset_config['workers']
        self.debug = dataset_config['debug']
        self.need_convert = dataset_config['Need_convert']
        for task, task_info in dataset_config['Task_and_class_config'].items():
            source_dataset_class = get_class_list(
                task_info['Source_dataset_class_file_path'])
            modify_class_dict = get_modify_class_dict(
                task_info['Modify_class_file_path'])
            target_dataset_class = get_new_class_names_list(
                source_dataset_class, modify_class_dict)
            object_pixel_limit_dict = get_class_pixel_limit(
                task_info['Target_each_class_object_pixel_limit_file_path'])
            self.task_dict[task] = {
                'Source_dataset_class': source_dataset_class,
                'Modify_class_dict': modify_class_dict,
                'Target_dataset_class': target_dataset_class,
                'Target_object_pixel_limit_dict': object_pixel_limit_dict,
            }
            self.temp_merge_class_list[
                'Merge_source_dataset_class_list'].extend(source_dataset_class)
            self.temp_merge_class_list[
                'Merge_target_dataset_class_list'].extend(target_dataset_class)

        self.total_task_source_class_list = self.get_total_task_source_class_list(
        )

        # 统计设置
        self.only_statistic = dataset_config['Only_statistic']
        # 按类别统计标注目标
        self.temp_sample_objec_class_statistics_folder = check_output_path(
            os.path.join(self.temp_sample_statistics_folder,
                         'object_class_statistics'))
        # 按距离统计标注目标
        self.statistic_label_object_distance = dataset_config[
            'Statistic_label_object_distance']
        if self.statistic_label_object_distance:
            self.temp_sample_objec_distance_statistics_folder = check_output_path(
                os.path.join(self.temp_sample_statistics_folder,
                             'object_distance_statistics'))
        self.statistic_label_distance = dataset_config[
            'Statistic_label_distance']
        self.statistic_label_distance_statistic_segmentation = dataset_config[
            'Statistic_label_distance_statistic_segmentation']
        # 按角度统计标注目标
        self.statistic_label_object_angle = dataset_config[
            'Statistic_label_object_angle']
        if self.statistic_label_object_angle:
            self.temp_sample_objec_angle_statistics_folder = check_output_path(
                os.path.join(self.temp_sample_statistics_folder,
                             'object_angle_statistics'))
        self.statistic_label_angle = dataset_config['Statistic_label_angle']
        self.statistic_label_angle_statistic_segmentation = dataset_config[
            'Statistic_label_angle_statistic_segmentation']

        # 相机内外参
        self.camera_calibration_extrinsics_file_path_dict = dataset_config[
            'Camera_calibration_file_path']['extrinsics']
        self.camera_calibration_intrinsics_file_path_dict = dataset_config[
            'Camera_calibration_file_path']['intrinsics']
        self.camera_calibration_extrinsics = {}
        self.camera_calibration_intrinsics = {}

        print('Dataset instance initialize end.')
        return True

    def get_total_task_source_class_list(self) -> list:
        """获取全部任务数据集列表

        Returns:
            list: _description_
        """

        total_task_source_class_list = []
        for task_class_dict in self.task_dict.values():
            if task_class_dict is not None:
                total_task_source_class_list.extend(
                    task_class_dict['Source_dataset_class'])
        total_task_source_class_list = list(set(total_task_source_class_list))

        return total_task_source_class_list

    def source_dataset_copy_image_and_annotation(self) -> None:
        """拷贝图片和标注文件
        """

        print('\nStart source dataset copy image and annotation:')
        if not self.only_statistic:
            pbar, update = multiprocessing_object_tqdm(
                self.source_dataset_image_count, 'Copy images')
            for root, _, files in os.walk(self.dataset_input_folder):
                pool = multiprocessing.Pool(self.workers)
                for n in files:
                    if os.path.splitext(n)[-1].replace('.', '') in \
                            self.source_dataset_image_form_list:
                        pool.apply_async(self.source_dataset_copy_image,
                                         args=(
                                             root,
                                             n,
                                         ),
                                         callback=update,
                                         error_callback=err_call_back)
                pool.close()
                pool.join()
            pbar.close()

        if self.source_dataset_style == 'HY_VAL':
            annotation_count = self.source_dataset_image_count
        else:
            annotation_count = self.source_dataset_annotation_count
        pbar, update = multiprocessing_object_tqdm(annotation_count,
                                                   'Copy annotations')
        for root, _, files in os.walk(self.dataset_input_folder):
            pool = multiprocessing.Pool(self.workers)
            for n in files:
                if self.source_dataset_style == 'HY_VAL':
                    pool.apply_async(self.source_dataset_copy_annotation,
                                     args=(
                                         root,
                                         n,
                                     ),
                                     callback=update,
                                     error_callback=err_call_back)
                else:
                    if n.endswith(self.source_dataset_annotation_form):
                        pool.apply_async(self.source_dataset_copy_annotation,
                                         args=(
                                             root,
                                             n,
                                         ),
                                         callback=update,
                                         error_callback=err_call_back)
            pool.close()
            pool.join()
        pbar.close()

        print('Copy images and annotations end.')

        return

    def source_dataset_copy_image(self, root: str, n: str) -> None:
        """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

        Args:
            root (str): [文件所在目录]
            n (str): [文件名]
        """

        image_path = os.path.join(root, n)
        temp_image_path = os.path.join(self.source_dataset_images_folder,
                                       self.file_prefix + n)
        image_suffix = os.path.splitext(n)[-1].replace('.', '')
        if image_suffix != self.target_dataset_image_form:
            image_transform_type = image_suffix + \
                '_' + self.target_dataset_image_form
            image_form_transform.__dict__[image_transform_type](
                image_path, temp_image_path)
            return
        else:
            shutil.copy(image_path, temp_image_path)
            return

    def source_dataset_copy_annotation(self, root: str, n: str) -> None:
        """[复制源数据集标签文件至目标数据集中的source_annotations中]

        Args:
            root (str): [文件所在目录]
            n (str): [文件名]
        """

        annotation_path = os.path.join(root, n)
        temp_annotation_path = os.path.join(
            self.source_dataset_annotations_folder, n)
        shutil.copy(annotation_path, temp_annotation_path)

        return

    def transform_to_temp_dataset(self) -> None:
        """[转换标注文件为暂存标注]
        """

        print('\nStart transform to temp dataset:')
        success_count = 0
        fail_count = 0
        no_object = 0
        temp_file_name_list = []

        pbar, update = multiprocessing_object_tqdm(
            self.source_dataset_annotation_count, 'Total annotations')
        process_temp_file_name_list = multiprocessing.Manager().list()
        process_output = multiprocessing.Manager().dict({
            'success_count':
            0,
            'fail_count':
            0,
            'no_object':
            0,
            'temp_file_name_list':
            process_temp_file_name_list
        })
        pool = multiprocessing.Pool(self.workers)
        for source_annotation_name in os.listdir(
                self.source_dataset_annotations_folder):
            pool.apply_async(func=self.load_image_annotation,
                             args=(
                                 source_annotation_name,
                                 process_output,
                             ),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        # 更新输出统计
        success_count += process_output['success_count']
        fail_count += process_output['fail_count']
        no_object += process_output['no_object']
        temp_file_name_list += process_output['temp_file_name_list']

        # 输出读取统计结果
        total_annotations = len(
            os.listdir(self.source_dataset_annotations_folder))
        print('\nSource dataset convert to temp dataset file count: ')
        print('Total annotations:         \t {} '.format(total_annotations))
        print('Convert fail:              \t {} '.format(fail_count))
        print('No object delete images: \t {} '.format(no_object))
        print(
            'Convert success:           \t {} '.format(total_annotations -
                                                       fail_count - no_object))
        self.temp_annotation_name_list = temp_file_name_list
        print('Source dataset annotation transform to temp dataset end.')

        return

    def get_source_dataset_image_count(self) -> int:
        """[获取源数据集图片数量]

        Returns:
            int: [源数据集图片数量]
        """

        image_count = 0
        for root, _, files in os.walk(self.dataset_input_folder):
            for n in files:
                if os.path.splitext(n)[-1].replace('.', '') in \
                        self.source_dataset_image_form_list:
                    image_count += 1

        return image_count

    def get_source_dataset_annotation_count(self) -> int:
        """[获取源数据集标注文件数量]

        Returns:
            int: [源数据集标注文件数量]
        """

        annotation_count = 0
        for root, _, files in os.walk(self.dataset_input_folder):
            for n in files:
                if n.endswith(self.source_dataset_annotation_form):
                    annotation_count += 1

        return annotation_count

    def output_classname_file(self) -> None:
        """[输出类别文件]
        """

        print('Output task class name file.')
        for task, task_class_dict in self.task_dict.items():
            if task_class_dict is None:
                continue
            with open(
                    os.path.join(self.temp_informations_folder,
                                 task + '_classes.names'), 'w') as f:
                if len(task_class_dict['Target_dataset_class']):
                    f.write('\n'.join(
                        str(n)
                        for n in task_class_dict['Target_dataset_class']))
                f.close()

        return

    def delete_redundant_image_annotation(self) -> None:
        """[删除无标注图片, 无标注temp annotation]
        """

        print('\nStar delete redundant image:')
        self.temp_image_name_list = self.get_temp_images_name_list()
        self.temp_annotation_name_list = self.get_temp_annotations_name_list()
        delete_image_count = 0
        for n in tqdm(os.listdir(self.temp_images_folder),
                      desc='Chech and delete redundant images'):
            image_name = os.path.splitext(n)[0]
            if image_name not in self.temp_annotation_name_list:
                delete_image_path = os.path.join(self.temp_images_folder, n)
                print('Delete redundant image: \t{}'.format(n))
                os.remove(delete_image_path)
                delete_image_count += 1
        self.temp_image_name_list = self.get_temp_images_name_list()

        delete_annotation_count = 0
        self.temp_annotation_name_list = self.get_temp_annotations_name_list()
        print('\nStar delete redundant annotation:')
        for n in tqdm(os.listdir(self.temp_annotations_folder),
                      desc='Chech and delete redundant annotations'):
            annotation_name = os.path.splitext(n)[0]
            if annotation_name not in self.temp_image_name_list:
                delete_image_path = os.path.join(self.temp_annotations_folder,
                                                 n)
                print('Delete redundant image: \t{}'.format(n))
                os.remove(delete_image_path)
                delete_annotation_count += 1
        self.temp_annotation_name_list = self.get_temp_annotations_name_list()

        print('Total delete redundant images count: {}'.format(
            delete_image_count))
        print('Total delete redundant annotation count: {}'.format(
            delete_annotation_count))
        # 更新文件名及文件路径成员
        print('Update temp annotation name list.')
        self.temp_annotation_name_list = self.get_temp_annotations_name_list()
        print('Update temp path name list.')
        self.temp_annotations_path_list = self.get_temp_annotations_path_list()
        print('Update total file name path.')
        self.total_file_name_path = self.total_file()

        return

    def get_temp_annotations_name_list(self) -> list:
        """[获取暂存数据集文件名称列表]

        Returns:
            list: [暂存数据集文件名称列表]
        """

        temp_file_name_list = []  # 暂存数据集文件名称列表
        print('Get temp file name list:')
        for n in tqdm(os.listdir(self.temp_annotations_folder)):
            temp_file_name_list.append(
                os.path.splitext(n.split(os.sep)[-1])[0])

        return temp_file_name_list

    def get_temp_images_name_list(self) -> list:
        """[获取暂存数据集图片名称列表]

        Returns:
            list: [暂存数据集文件名称列表]
        """

        temp_file_name_list = []  # 暂存数据集文件名称列表
        print('Get temp file name list:')
        for n in tqdm(os.listdir(self.temp_images_folder)):
            temp_file_name_list.append(
                os.path.splitext(n.split(os.sep)[-1])[0])

        return temp_file_name_list

    def get_temp_annotations_path_list(self) -> list:
        """[获取暂存数据集全量标签路径列表]

        Args:
            temp_annotations_folder (str): [暂存数据集标签文件夹路径]

        Returns:
            list: [暂存数据集全量标签路径列表]
        """

        temp_annotation_path_list = []  # 暂存数据集全量标签路径列表
        print('Get temp annotation path.')
        for n in os.listdir(self.temp_annotations_folder):
            temp_annotation_path_list.append(
                os.path.join(self.temp_annotations_folder, n))

        return temp_annotation_path_list

    def total_file(self) -> list:
        """[获取暂存数据集全量图片文件名列表]

        Args:
            temp_informations_folder (str): [暂存数据集信息文件夹]

        Returns:
            list: [暂存数据集全量图片文件名列表]
        """

        total_list = []  # 暂存数据集全量图片文件名列表
        print('Get total file name list.')
        try:
            with open(os.path.join(self.temp_informations_folder, 'total.txt'),
                      'r') as f:
                for n in f.read().splitlines():
                    total_list.append(os.path.splitext(n.split(os.sep)[-1])[0])
                f.close()

            total_file_name_path = os.path.join(self.temp_informations_folder,
                                                'total_file_name.txt')
            print('Output total_file_name.txt.')
            with open(total_file_name_path, 'w') as f:
                if len(total_list):
                    for n in total_list:
                        f.write('%s\n' % n)
                    f.close()
                else:
                    f.close()
        except:
            print('total.txt had not create, return None.')

            return None

        return total_file_name_path

    def divide_dataset(self) -> None:
        """按不同场景划分数据集, 并根据不同场景按比例抽取train、val、test、redundancy比例为
        train_ratio, val_ratio, test_ratio, redund_ratio
        """
        print('\nStart divide dataset:')
        Main_path = check_output_path(self.temp_informations_folder, 'Main')
        # 统计数据集不同场景图片数量
        scene_count_dict = {}  # 场景图片计数字典
        train_dict = {}  # 训练集图片字典
        test_dict = {}  # 测试集图片字典
        val_dict = {}  # 验证集图片字典
        redund_dict = {}  # 冗余图片字典
        set_dict_list = [train_dict, val_dict, test_dict,
                         redund_dict]  # 数据集字典列表
        total_list = []  # 全图片列表
        # 获取全图片列表
        for one_image_name in self.temp_annotation_name_list:
            one = str(one_image_name).replace('\n', '')
            total_list.append(one)
        # 依据数据集场景划分数据集
        for image_name in total_list:  # 遍历全部的图片名称
            image_name_list = image_name.split(
                self.file_prefix_delimiter)  # 对图片名称按前缀分段，区分场景
            image_name_str = ''
            # 读取切分图片名称的值，去掉编号及后缀
            for a in image_name_list[:-1]:
                # name_str为图片包含场景的名称
                image_name_str += a
            if image_name_str in scene_count_dict.keys():  # 判断是否已经存入场景计数字典
                # 若已经存在，则计数加1
                scene_count_dict[image_name_str][0] += 1
                scene_count_dict[image_name_str][1].append(
                    image_name)  # 同时将图片名称存入对应场景分类键下
            else:
                scene_count_dict.setdefault((image_name_str),
                                            []).append(1)  # 若为新场景，则添加场景
                scene_count_dict[image_name_str].append(
                    [image_name])  # 同时将图片名称存入对应场景分类键下
        # 计算不同场景按数据集划分比例选取样本数量
        # 遍历场景图片计数字典，获取键（不同场景）和键值（图片数、图片名称）
        if self.target_dataset_style == 'CITYSCAPES_VAL':
            self.target_dataset_divide_proportion = (0, 1, 0, 0)
        if self.target_dataset_style == 'CVAT_IMAGE_1_1':
            self.target_dataset_divide_proportion = (1, 0, 0, 0)
        for key, val in scene_count_dict.items():
            # 打包配对不同set对应不同的比例
            for diff_set_dict, diff_ratio in zip(
                    set_dict_list, self.target_dataset_divide_proportion):
                if diff_ratio == 0:  # 判断对应数据集下是否存在数据，若不存在则继续下一数据集数据挑选
                    continue
                diff_set_dict[key] = math.floor(
                    diff_ratio * val[0])  # 计算不同场景下不同的set应该收录的图片数
                # 依据获取的不同场景的图片数，顺序获取该数量的图片名字列表
                for a in range(diff_set_dict[key]):
                    diff_set_dict.setdefault('image_name_list', []).append(
                        scene_count_dict[key][1].pop())
        # 对分配的数据集图片名称，进行输出，分别输出为训练、测试、验证集的xml格式的txt文件
        set_name_list = ['train', 'val', 'test', 'redund']
        num_count = 0  # 图片计数
        trainval_list = []  # 训练集、验证集列表
        for set_name, set_one_path in zip(set_name_list, set_dict_list):
            print('Output images path {}.txt.'.format(set_name))
            with open(
                    os.path.join(self.temp_informations_folder,
                                 '%s.txt' % set_name), 'w') as f:
                # 判断读取列表是否不存在，入若不存在则遍历下一数据集图片
                if len(set_one_path):
                    if self.target_dataset_style != 'cityscapes_val':
                        random.shuffle(set_one_path['image_name_list'])
                    for n in set_one_path['image_name_list']:
                        image_path = os.path.join(
                            self.temp_images_folder,
                            n + '.' + self.target_dataset_image_form)
                        f.write('%s\n' % image_path)
                        num_count += 1
                    f.close()
                else:
                    print('No file divide to {}.'.format(set_name))
                    f.close()
                    continue
            print('Output file name {}.txt.'.format(set_name))
            with open(os.path.join(Main_path, '%s.txt' % set_name), 'w') as f:
                # 判断读取列表是否不存在，入若不存在则遍历下一数据集图片
                if len(set_one_path):
                    if self.target_dataset_style != 'cityscapes_val':
                        random.shuffle(set_one_path['image_name_list'])
                    for n in set_one_path['image_name_list']:
                        file_name = n.split(os.sep)[-1]
                        f.write('%s\n' % file_name)
                        if set_name == 'train' or set_name == 'val':
                            trainval_list.append(file_name)
                    f.close()
                else:
                    f.close()
                    continue
        print('Output file name trainval.txt.')
        with open(os.path.join(Main_path, 'trainval.txt'), 'w') as f:
            if len(trainval_list):
                f.write('\n'.join(str(n) for n in trainval_list))
                f.close()
            else:
                f.close()
        print('Output total.txt.')
        with open(os.path.join(self.temp_informations_folder, 'total.txt'),
                  'w') as f:
            if len(trainval_list):
                for n in total_list:
                    image_path = os.path.join(
                        self.temp_images_folder,
                        n + '.' + self.target_dataset_image_form)
                    f.write('%s\n' % image_path)
                f.close()
            else:
                f.close()
        print('Output total_file_name.txt.')
        with open(
                os.path.join(self.temp_informations_folder,
                             'total_file_name.txt'), 'w') as f:
            if len(total_list):
                for n in total_list:
                    f.write('%s\n' % n)
                f.close()
            else:
                f.close()
        print('Total images: %d' % num_count)
        print('Divide files has been create in:\n%s' %
              self.temp_informations_folder)
        print('Divide dataset end.')

        return

    def sample_statistics(self) -> None:
        """[数据集样本统计]
        """

        if self.target_dataset_style == 'CITYSCAPES_VAL':
            return

        # 分割后各数据集annotation文件路径
        for n in self.temp_divide_file_list:
            divide_file_name = os.path.splitext(n.split(os.sep)[-1])[0]
            with open(n, 'r') as f:
                annotation_path_list = []
                for m in f.read().splitlines():
                    file_name = os.path.splitext(m.split(os.sep)[-1])[0]
                    annotation_path = os.path.join(
                        self.temp_annotations_folder,
                        file_name + '.' + self.temp_annotation_form)
                    annotation_path_list.append(annotation_path)
            self.temp_divide_file_annotation_path_dict.update(
                {divide_file_name: annotation_path_list})

        print('\nStar statistic sample each dataset:')
        for task, task_class_dict in self.task_dict.items():
            if task == 'Detection' and task_class_dict is not None:
                self.detection_sample_statistics(task, task_class_dict)
            elif task == 'Semantic_segmentation' and task_class_dict is not None:
                self.segmentation_sample_statistics(task, task_class_dict)
            elif task == 'Instance_segmentation' and task_class_dict is not None:
                self.detection_sample_statistics(task, task_class_dict)
                self.segmentation_sample_statistics(task, task_class_dict)
            elif task == 'Keypoint' and task_class_dict is not None:
                self.keypoint_sample_statistics(task, task_class_dict)

        return

    def detection_sample_statistics(self, task: str,
                                    task_class_dict: dict) -> None:
        """[数据集样本统计]

        Args:
            task (str): [任务类型]
            task_class_dict (dict): [任务类别字典]
        """

        print('Start statistic detection sample:')
        # 声明类别数量统计的dataframe
        data = {}
        for divide_file_name in self.temp_divide_file_annotation_path_dict.keys(
        ):
            data.update({
                divide_file_name:
                [0 for _ in task_class_dict['Target_dataset_class']]
            })
        each_class_object_count_dataframe = pd.DataFrame(
            data, index=[x for x in task_class_dict['Target_dataset_class']])
        each_class_object_proportion_dataframe = pd.DataFrame(
            data, index=[x for x in task_class_dict['Target_dataset_class']])

        # 声明类别距离统计的dataframe
        data_object_distance_count = {}
        range_list = [
            f'{i}~{i+self.statistic_label_distance_statistic_segmentation}'
            for i in range(
                0, self.statistic_label_distance,
                self.statistic_label_distance_statistic_segmentation)
        ]
        for n in range_list:
            data_object_distance_count.update({n: 0})
        # 声明全部数据的目标距离统计字典，该字典包含4个数据集数据
        total_data_object_distance_count = {}
        total_data_object_distance_count_proportion = {}
        for divide_file_name in self.temp_divide_file_annotation_path_dict.keys(
        ):
            each_class_object_distance_count_dataframe = pd.DataFrame(
                data_object_distance_count,
                index=[x for x in task_class_dict['Target_dataset_class']])
            total_data_object_distance_count.update(
                {divide_file_name: each_class_object_distance_count_dataframe})
        for divide_file_name in self.temp_divide_file_annotation_path_dict.keys(
        ):
            each_class_object_distance_count_dataframe = pd.DataFrame(
                data_object_distance_count,
                index=[x for x in task_class_dict['Target_dataset_class']])
            total_data_object_distance_count_proportion.update(
                {divide_file_name: each_class_object_distance_count_dataframe})

        # 声明角度距离统计的dataframe
        data_object_angle_count = {}
        angle_list = [
            f'{i}~{i+self.statistic_label_angle_statistic_segmentation}'
            for i in range(0, self.statistic_label_angle,
                           self.statistic_label_angle_statistic_segmentation)
        ]
        for n in angle_list:
            data_object_angle_count.update({n: 0})
        # 声明全部数据的目标角度统计字典，该字典包含4个数据集数据
        total_data_object_angle_count = {}
        total_data_object_angle_count_proportion = {}
        for divide_file_name in self.temp_divide_file_annotation_path_dict.keys(
        ):
            each_class_object_angle_count_dataframe = pd.DataFrame(
                data_object_angle_count,
                index=[x for x in task_class_dict['Target_dataset_class']])
            total_data_object_angle_count.update(
                {divide_file_name: each_class_object_angle_count_dataframe})
        for divide_file_name in self.temp_divide_file_annotation_path_dict.keys(
        ):
            each_class_object_angle_count_dataframe = pd.DataFrame(
                data_object_angle_count,
                index=[x for x in task_class_dict['Target_dataset_class']])
            total_data_object_angle_count_proportion.update(
                {divide_file_name: each_class_object_angle_count_dataframe})

        for divide_file_name, divide_annotation_list in tqdm(
                self.temp_divide_file_annotation_path_dict.items(),
                total=len(self.temp_divide_file_annotation_path_dict),
                desc='Statistic detection sample'):
            if divide_file_name == 'total':
                continue

            # 统计全部labels各类别数量
            total_image_count_object_dict_list = []
            pbar, update = multiprocessing_list_tqdm(
                divide_annotation_list,
                desc='Count {} set class object'.format(divide_file_name),
                leave=False)
            pool = multiprocessing.Pool(self.workers)
            for temp_annotation_path in divide_annotation_list:
                image_object_dict_list = pool.apply_async(
                    func=self.get_temp_annotations_object_count,
                    args=(temp_annotation_path, ),
                    callback=update,
                    error_callback=err_call_back)
                total_image_count_object_dict_list.append(
                    image_object_dict_list.get())
            pool.close()
            pool.join()
            pbar.close()
            # 获取多进程结果
            for n in tqdm(total_image_count_object_dict_list,
                          desc='Collection multiprocessing result(class)',
                          leave=False):
                for m in n:
                    for key, value in m.items():
                        each_class_object_count_dataframe[divide_file_name][
                            key] += value

            if self.statistic_label_object_distance:
                total_image_object_distance_count_dict_list = []
                pbar, update = multiprocessing_list_tqdm(
                    divide_annotation_list,
                    desc='Count {} set object distance'.format(
                        divide_file_name),
                    leave=False)
                pool = multiprocessing.Pool(self.workers)
                for temp_annotation_path in divide_annotation_list:
                    image_object_distance_count_dict_list = pool.apply_async(
                        func=self.get_temp_annotations_object_distance_count,
                        args=(temp_annotation_path, ),
                        callback=update,
                        error_callback=err_call_back)
                    total_image_object_distance_count_dict_list.append(
                        image_object_distance_count_dict_list.get())
                pool.close()
                pool.join()
                pbar.close()
                # 获取距离统计多进程结果
                for n in tqdm(
                        total_image_object_distance_count_dict_list,
                        desc='Collection multiprocessing result(distance)',
                        leave=False):
                    for object_class, object_distance_count_list in n.items():
                        for index, object_distance_count in enumerate(
                                object_distance_count_list):
                            if object_distance_count:
                                total_data_object_distance_count[
                                    divide_file_name].loc[
                                        [object_class],
                                        [range_list[index]
                                         ]] += object_distance_count

            if self.statistic_label_object_angle:
                total_image_object_angle_count_dict_list = []
                pbar, update = multiprocessing_list_tqdm(
                    divide_annotation_list,
                    desc='Count {} set object angle'.format(divide_file_name),
                    leave=False)
                pool = multiprocessing.Pool(self.workers)
                for temp_annotation_path in divide_annotation_list:
                    image_object_angle_count_dict_list = pool.apply_async(
                        func=self.get_temp_annotations_object_angle_count,
                        args=(temp_annotation_path, ),
                        callback=update,
                        error_callback=err_call_back)
                    total_image_object_angle_count_dict_list.append(
                        image_object_angle_count_dict_list.get())
                pool.close()
                pool.join()
                pbar.close()
                # 获取距离统计多进程结果
                for n in tqdm(total_image_object_angle_count_dict_list,
                              desc='Collection multiprocessing result(angle)',
                              leave=False):
                    for object_class, object_angle_count_list in n.items():
                        for index, object_angle_count in enumerate(
                                object_angle_count_list):
                            if object_angle_count:
                                total_data_object_angle_count[
                                    divide_file_name].loc[
                                        [object_class],
                                        [angle_list[index]
                                         ]] += object_angle_count

        # get object count total
        each_class_object_count_dataframe[
            'total'] = each_class_object_count_dataframe.sum(axis=1)
        # get object distance count total
        if self.statistic_label_object_distance:
            for object_distance_count in total_data_object_distance_count.values(
            ):
                total_data_object_distance_count[
                    'total'] += object_distance_count
        # get object angle count total
        if self.statistic_label_object_angle:
            for object_angle_count in total_data_object_angle_count.values():
                total_data_object_angle_count['total'] += object_angle_count

        # 类别占比统计
        for n in each_class_object_proportion_dataframe.keys():
            total_count = each_class_object_count_dataframe[n].sum()
            each_class_object_proportion_dataframe[
                n] = each_class_object_count_dataframe[n].apply(
                    lambda x: x / total_count)
        each_class_object_proportion_dataframe = each_class_object_proportion_dataframe.fillna(
            int(0))
        each_class_object_count_dataframe = each_class_object_count_dataframe.sort_index(
            ascending=False)
        each_class_object_proportion_dataframe = each_class_object_proportion_dataframe.sort_index(
            ascending=False)
        self.temp_divide_object_count_dataframe_dict.update(
            {task: each_class_object_count_dataframe})
        self.temp_divide_each_pixel_proportion_dataframe_dict.update(
            {task: each_class_object_proportion_dataframe})
        # 记录类别分布
        each_class_object_count_dataframe.to_csv(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Detection_object_count.csv')))
        each_class_object_proportion_dataframe.to_csv(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Detection_object_proportion.csv')))
        # 绘图
        each_class_object_count_dataframe.plot(kind='bar')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Detection_object_count.png')))
        each_class_object_proportion_dataframe.plot()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Detection_object_proportion.png')))

        # 类别距离计数占比统计
        if self.statistic_label_object_distance:
            for divide_dataset_name, object_distance_count in total_data_object_distance_count.items(
            ):
                for each_class_distance_count in object_distance_count.itertuples(
                ):
                    each_class_distance_count_array = np.array(
                        each_class_distance_count[1:])
                    total_each_class_distance_count = each_class_distance_count_array.sum(
                    )
                    if total_each_class_distance_count:
                        each_class_distance_count_array_proportion = each_class_distance_count_array / total_each_class_distance_count
                    else:
                        each_class_distance_count_array_proportion = np.zeros_like(
                            each_class_distance_count_array)
                    total_data_object_distance_count_proportion[
                        divide_dataset_name].loc[each_class_distance_count[
                            0]] = each_class_distance_count_array_proportion
            # 记录目标距离计数分布
            for divide_file_name, object_distance_count_dataframe in total_data_object_distance_count.items(
            ):
                object_distance_count_dataframe.to_csv(
                    os.path.join(
                        self.temp_sample_objec_distance_statistics_folder,
                        'Detection_object_distance_count_{}.csv'.format(
                            divide_file_name)))
                # 绘图
                # object_distance_count_dataframe.plot(kind='bar')
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                # plt.savefig((os.path.join(
                #     self.temp_sample_objec_distance_statistics_folder,
                #     'Detection_object_distance_count_{}.png'.format(
                #         divide_file_name))))
            for divide_file_name, object_distance_count_proportion_dataframe in total_data_object_distance_count_proportion.items(
            ):
                object_distance_count_proportion_dataframe.to_csv(
                    os.path.join(
                        self.temp_sample_objec_distance_statistics_folder,
                        'Detection_object_distance_count_proportion_{}.csv'.
                        format(divide_file_name)))
                # 绘图
                # object_distance_count_proportion_dataframe.plot()
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                # plt.savefig((os.path.join(
                #     self.temp_sample_objec_distance_statistics_folder,
                #     'Detection_object_distance_count_proportion_{}.png'.format(
                #         divide_file_name))))

        # 类别角度计数占比统计
        if self.statistic_label_object_angle:
            for divide_dataset_name, object_angle_count in total_data_object_angle_count.items(
            ):
                for each_class_angle_count in object_angle_count.itertuples():
                    each_class_angle_count_array = np.array(
                        each_class_angle_count[1:])
                    total_each_class_angle_count = each_class_angle_count_array.sum(
                    )
                    if total_each_class_angle_count:
                        each_class_angle_count_array_proportion = each_class_angle_count_array / total_each_class_angle_count
                    else:
                        each_class_angle_count_array_proportion = np.zeros_like(
                            each_class_angle_count_array)
                    total_data_object_angle_count_proportion[
                        divide_dataset_name].loc[each_class_angle_count[
                            0]] = each_class_angle_count_array_proportion
            # 记录目标距离计数分布
            for divide_file_name, object_angle_count_dataframe in total_data_object_angle_count.items(
            ):
                object_angle_count_dataframe.to_csv(
                    os.path.join(
                        self.temp_sample_objec_angle_statistics_folder,
                        'Detection_object_angle_count_{}.csv'.format(
                            divide_file_name)))
                # 绘图
                # object_angle_count_dataframe.plot(kind='bar')
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                # plt.savefig((os.path.join(
                #     self.temp_sample_objec_angle_statistics_folder,
                #     'Detection_object_angle_count_{}.png'.format(
                #         divide_file_name))))
            for divide_file_name, object_angle_count_proportion_dataframe in total_data_object_angle_count_proportion.items(
            ):
                object_angle_count_proportion_dataframe.to_csv(
                    os.path.join(
                        self.temp_sample_objec_angle_statistics_folder,
                        'Detection_object_angle_count_proportion_{}.csv'.
                        format(divide_file_name)))
                # 绘图
                # object_angle_count_proportion_dataframe.plot()
                # plt.xticks(rotation=45)
                # plt.tight_layout()
                # plt.savefig((os.path.join(
                #     self.temp_sample_objec_angle_statistics_folder,
                #     'Detection_object_angle_count_proportion_{}.png'.format(
                #         divide_file_name))))

        return

    def get_temp_annotations_object_count(self,
                                          temp_annotation_path: str) -> list:
        """[获取暂存标签中目标与自车距离计数字典列表]

        Args:
            temp_annotation_path (str): [暂存标签路径]

        Returns:
            list: [类别统计字典列表]
        """

        total_annotation_object_count_dict_list = []
        image = self.TEMP_LOAD(self, temp_annotation_path)
        for object in image.object_list:
            if object.box_exist_flag:
                total_annotation_object_count_dict_list.append(
                    {object.box_clss: 1})

        return total_annotation_object_count_dict_list

    def get_temp_annotations_object_distance_count(
            self, temp_annotation_path: str) -> list:
        """[获取暂存标签类别按距离计数字典列表]

        Args:
            temp_annotation_path (str): [暂存标签路径]

        Returns:
            list: [类别统计字典列表]
        """

        object_distance_count_dict = {}  # 距离统计
        self_center = self.self_position_parse(
            self.camera_image_wh[1], self.label_image_wh[0],
            self.label_image_wh[1], self.label_range[0], self.label_range[1],
            self.label_range[2], self.label_range[3])
        # 距离像素换算关系 m/pixel
        dp_rate = self.distance_pixel_rate(
            self.label_range[2] + self.label_range[3], self.label_image_wh[0])
        image = self.TEMP_LOAD(self, temp_annotation_path)
        for object in image.object_list:
            if object.box_exist_flag:
                box_clss = object.box_clss
                # 计算box中心点像素坐标
                box_xcenter = (float(object.box_xtlytlxbrybr[0]) +
                               float(object.box_xtlytlxbrybr[2])) / 2
                box_ycenter = (float(object.box_xtlytlxbrybr[1]) +
                               float(object.box_xtlytlxbrybr[3])) / 2
                # print(box_xcenter, box_ycenter)
                # 计算box中心点与自车中心点的距离
                pixel_distance = calculate_distance(box_xcenter, box_ycenter,
                                                    self_center[0],
                                                    self_center[1])
                real_distance = pixel_distance * dp_rate
                if box_clss not in object_distance_count_dict.keys():
                    object_distance_count_dict[box_clss] = [0] * (
                        (self.statistic_label_distance + 1) //
                        self.statistic_label_distance_statistic_segmentation)
                if real_distance > self.statistic_label_distance:
                    continue
                object_distance_count_dict[box_clss][int(
                    real_distance //
                    self.statistic_label_distance_statistic_segmentation)] += 1

        return object_distance_count_dict

    def get_temp_annotations_object_angle_count(
            self, temp_annotation_path: str) -> list:
        """[获取暂存标签按角度类别计数字典列表]

        Args:
            temp_annotation_path (str): [暂存标签路径]

        Returns:
            list: [类别统计字典列表]
        """

        object_angle_count_dict = {}  # 距离统计
        image = self.TEMP_LOAD(self, temp_annotation_path)
        for object in image.object_list:
            if object.box_exist_flag:
                box_clss = object.box_clss
                real_angle = int(object.box_head_orientation)
                if box_clss not in object_angle_count_dict.keys():
                    object_angle_count_dict[box_clss] = [0] * (
                        (self.statistic_label_angle + 1) //
                        self.statistic_label_angle_statistic_segmentation)
                if abs(real_angle) >= 360:
                    real_angle = real_angle % 360
                if real_angle < 0:
                    real_angle += 360
                object_angle_count_dict[box_clss][int(
                    real_angle //
                    self.statistic_label_angle_statistic_segmentation)] += 1

        return object_angle_count_dict

    def segmentation_sample_statistics(self, task: str,
                                       task_class_dict: dict) -> None:
        """[语义分割样本统计]

        Args:
            task (str): [任务类型]
            task_class_dict (dict): [对应任务类别字典]
        """

        print('Start statistic semantic segmentation sample:')
        # 声明dataframe
        data = {}
        temp_task_class_list = task_class_dict['Target_dataset_class']
        if 'unlabeled' not in temp_task_class_list:
            temp_task_class_list.append('unlabeled')
        for divide_file_name in self.temp_divide_file_annotation_path_dict.keys(
        ):
            data.update({divide_file_name: [0 for _ in temp_task_class_list]})
        each_class_object_count_dataframe = pd.DataFrame(
            data, index=[x for x in temp_task_class_list])
        each_class_pixel_count_dataframe = pd.DataFrame(
            data, index=[x for x in temp_task_class_list])
        each_class_pixel_proportion_dataframe = pd.DataFrame(
            data, index=[x for x in temp_task_class_list])

        for divide_file_name, divide_annotation_list in tqdm(
                self.temp_divide_file_annotation_path_dict.items(),
                total=len(self.temp_divide_file_annotation_path_dict),
                desc='Statistic semantic segmentation sample'):
            if divide_file_name == 'total':
                continue

            # 统计全部labels各类别像素点数量
            total_image_count_pixel_dict_list = []
            pbar, update = multiprocessing_list_tqdm(
                divide_annotation_list,
                desc='Count {} set class pixal'.format(divide_file_name),
                leave=False)
            pool = multiprocessing.Pool(self.workers)
            for temp_annotation_path in divide_annotation_list:
                image_class_pixal_dict_list = pool.apply_async(
                    func=self.get_temp_segmentation_class_pixal,
                    args=(temp_annotation_path, ),
                    callback=update,
                    error_callback=err_call_back)
                total_image_count_pixel_dict_list.append(
                    image_class_pixal_dict_list.get())

            pool.close()
            pool.join()
            pbar.close()

            # 获取多进程结果
            for n in tqdm(total_image_count_pixel_dict_list,
                          desc='Collection multiprocessing result',
                          leave=False):
                for l in n[0]:
                    for key, value in l.items():
                        each_class_pixel_count_dataframe[divide_file_name][
                            key] += value
                for m in n[1]:
                    for key, value in m.items():
                        each_class_object_count_dataframe[divide_file_name][
                            key] += value

        each_class_object_count_dataframe[
            'total'] = each_class_object_count_dataframe.sum(axis=1)
        each_class_pixel_count_dataframe[
            'total'] = each_class_pixel_count_dataframe.sum(axis=1)

        # 类别占比统计
        for n in each_class_pixel_proportion_dataframe.keys():
            total_count = each_class_pixel_count_dataframe[n].sum()
            each_class_pixel_proportion_dataframe[
                n] = each_class_pixel_count_dataframe[n].apply(
                    lambda x: x / total_count)
        each_class_pixel_proportion_dataframe = each_class_pixel_proportion_dataframe.fillna(
            int(0))
        each_class_object_count_dataframe = each_class_object_count_dataframe.sort_index(
            ascending=False)
        each_class_pixel_count_dataframe = each_class_pixel_count_dataframe.sort_index(
            ascending=False)
        each_class_pixel_proportion_dataframe = each_class_pixel_proportion_dataframe.sort_index(
            ascending=False)
        self.temp_divide_object_count_dataframe_dict.update(
            {task: each_class_object_count_dataframe})
        self.temp_divide_each_class_pixel_count_dataframe_dict.update(
            {task: each_class_pixel_count_dataframe})
        self.temp_divide_each_pixel_proportion_dataframe_dict.update(
            {task: each_class_pixel_proportion_dataframe})
        # 记录类别分布
        each_class_object_count_dataframe.to_csv(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Semantic_segmentation_object_count.csv')))
        each_class_pixel_count_dataframe.to_csv(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Semantic_segmentation_pixel_count.csv')))
        each_class_pixel_proportion_dataframe.to_csv(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Semantic_segmentation_pixel_proportion.csv')))

        each_class_object_count_dataframe.plot(kind='bar')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Semantic_segmentation_object_count.png')))
        each_class_pixel_count_dataframe.plot(kind='bar')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Semantic_segmentation_pixel_count.png')))
        each_class_pixel_proportion_dataframe.plot()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Semantic_segmentation_pixel_proportion.png')))

        return

    def get_temp_segmentation_class_pixal(self,
                                          temp_annotation_path: str) -> list:
        """获取分割类别像素计数

        Args:
            temp_annotation_path (str): 暂存标注路径

        Returns:
            list: image_class_pixal_dict_list, total_annotation_class_count_dict_list
        """

        image_class_pixal_dict_list = []
        total_annotation_class_count_dict_list = []

        image = self.TEMP_LOAD(self, temp_annotation_path)
        image_pixal = image.height * image.width
        if image == None:
            print('Load erro: ', temp_annotation_path)
            return
        for object in image.object_list:
            if object.segmentation_exist_flag:
                area = polygon_area(object.segmentation[:-1])
                if object.segmentation_clss != 'unlabeled':
                    image_class_pixal_dict_list.append(
                        {object.segmentation_clss: area})
                    total_annotation_class_count_dict_list.append(
                        {object.segmentation_clss: 1})
                else:
                    image_pixal -= area
                    if 'unlabeled' in self.task_dict['Semantic_segmentation'][
                            'Target_dataset_class']:
                        total_annotation_class_count_dict_list.append(
                            {object.segmentation_clss: 1})
        image_class_pixal_dict_list.append({'unlabeled': image_pixal})

        return [
            image_class_pixal_dict_list, total_annotation_class_count_dict_list
        ]

    def keypoint_sample_statistics(self, task: str,
                                   task_class_dict: dict) -> None:
        """[数据集样本统计]

        Args:
            dataset (dict): [数据集信息字典]
        """

        # 分割后各数据集annotation文件路径
        print('Start statistic detection sample:')
        # 声明dataframe
        data = {}
        for divide_file_name in self.temp_divide_file_annotation_path_dict.keys(
        ):
            data.update({
                divide_file_name:
                [0 for _ in task_class_dict['Target_dataset_class']]
            })
        each_class_keypoints_count_dataframe = pd.DataFrame(
            data, index=[x for x in task_class_dict['Target_dataset_class']])
        each_class_keypoints_proportion_dataframe = pd.DataFrame(
            data, index=[x for x in task_class_dict['Target_dataset_class']])

        for divide_file_name, divide_annotation_list in tqdm(
                self.temp_divide_file_annotation_path_dict.items(),
                total=len(self.temp_divide_file_annotation_path_dict),
                desc='Statistic detection sample'):
            if divide_file_name == 'total':
                continue

            # 统计全部labels各类别像素点数量
            total_image_count_object_dict_list = []
            pbar, update = multiprocessing_list_tqdm(
                divide_annotation_list,
                desc='Count {} set class object'.format(divide_file_name),
                leave=False)
            pool = multiprocessing.Pool(self.workers)
            for temp_annotation_path in divide_annotation_list:
                image_keypoints_dict_list = pool.apply_async(
                    func=self.get_temp_annotations_keypoints_class_count,
                    args=(temp_annotation_path, ),
                    callback=update,
                    error_callback=err_call_back)
                total_image_count_object_dict_list.append(
                    image_keypoints_dict_list.get())

            pool.close()
            pool.join()
            pbar.close()

            # 获取多进程结果
            for n in tqdm(total_image_count_object_dict_list,
                          desc='Collection multiprocessing result',
                          leave=False):
                for m in n:
                    for key, value in m.items():
                        each_class_keypoints_count_dataframe[divide_file_name][
                            key] += value

        each_class_keypoints_count_dataframe[
            'total'] = each_class_keypoints_count_dataframe.sum(axis=1)

        # 类别占比统计
        for n in each_class_keypoints_proportion_dataframe.keys():
            total_count = each_class_keypoints_count_dataframe[n].sum()
            each_class_keypoints_proportion_dataframe[
                n] = each_class_keypoints_count_dataframe[n].apply(
                    lambda x: x / total_count)
        each_class_keypoints_proportion_dataframe = each_class_keypoints_proportion_dataframe.fillna(
            int(0))

        each_class_keypoints_count_dataframe = each_class_keypoints_count_dataframe.sort_index(
            ascending=False)
        each_class_keypoints_proportion_dataframe = each_class_keypoints_proportion_dataframe.sort_index(
            ascending=False)

        self.temp_divide_object_count_dataframe_dict.update(
            {task: each_class_keypoints_count_dataframe})
        self.temp_divide_each_pixel_proportion_dataframe_dict.update(
            {task: each_class_keypoints_proportion_dataframe})

        # 记录类别分布
        each_class_keypoints_count_dataframe.to_csv(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Detection_object_count.csv')))
        each_class_keypoints_proportion_dataframe.to_csv(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Detection_object_proportion.csv')))

        each_class_keypoints_count_dataframe.plot(kind='bar')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Detection_object_count.png')))
        each_class_keypoints_proportion_dataframe.plot()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            (os.path.join(self.temp_sample_objec_class_statistics_folder,
                          'Detection_object_proportion.png')))

        return

    def get_temp_annotations_keypoints_class_count(
            self, temp_annotation_path: str) -> None:
        """[获取暂存标签信息]

        Args:
            dataset (dict): [数据集信息字典]
            temp_annotation_path (str): [暂存标签路径]
            process_output (dict): [进程输出字典]
        """

        image = self.TEMP_LOAD(self, temp_annotation_path)
        total_annotation_keypoints_class_count_dict_list = []
        for object in image.object_list:
            total_annotation_keypoints_class_count_dict_list.append(
                {object.keypoints_clss: 1})

        return total_annotation_keypoints_class_count_dict_list

    def get_image_mean_std(self, img_filename: str) -> list:
        """[获取图片均值和标准差]

        Args:
            dataset (dict): [数据集信息字典]
            img_filename (str): [图片名]

        Returns:
            list: [图片均值和标准差列表]
        """

        img_type = os.path.splitext(img_filename)[-1].split('.')[-1]
        if img_type == 'png':
            img = Image.open(
                os.path.join(self.source_dataset_images_folder, img_filename))
            img = cv2.cvtColor(np.asarray(img.getdata(), dtype='uint8'),
                               cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(
                os.path.join(self.source_dataset_images_folder, img_filename))
        m, s = cv2.meanStdDev(img / 255.0)
        name = img_filename

        return m.reshape((3, )), s.reshape((3, )), name

    def get_dataset_image_mean_std(self) -> None:
        """[计算读取的数据集图片均值、标准差]

        Args:
            dataset (dict): [数据集信息字典]
        """

        img_filenames = os.listdir(self.source_dataset_images_folder)
        print('\nStart count images mean and std:')
        pbar, update = multiprocessing_list_tqdm(
            img_filenames, desc='Count images mean and std')
        pool = multiprocessing.Pool(self.workers)
        mean_std_list = []
        for img_filename in img_filenames:
            mean_std_list.append(
                pool.apply_async(func=self.get_image_mean_std,
                                 args=(img_filename, ),
                                 callback=update,
                                 error_callback=err_call_back))
        pool.close()
        pool.join()
        pbar.close()

        m_list, s_list = [], []
        for n in mean_std_list:
            m_list.append(n.get()[0])
            s_list.append(n.get()[1])
        m_array = np.array(m_list)
        s_array = np.array(s_list)
        m = m_array.mean(axis=0, keepdims=True)
        s = s_array.mean(axis=0, keepdims=True)

        mean_std_file_output_path = os.path.join(self.temp_informations_folder,
                                                 'mean_std.json')
        mean_std = [{'mean': m.tolist(), 'std': s.tolist()}]
        mean_std_json = json.dumps(mean_std)
        with open(mean_std_file_output_path, 'w') as f:
            f.write(mean_std_json)
            f.close()
        print(mean_std)
        print('Count images mean and std end.')

        return

    def per_image_standardization(self, img_filename: str, mean, std,
                                  check_dataset_image_folder) -> list:
        """对图片进行标准化

        Args:
            img_filename (str): _description_

        Returns:
            list: _description_
        """

        img_type = os.path.splitext(img_filename)[-1].split('.')[-1]
        if img_type == 'png':
            img = Image.open(
                os.path.join(self.source_dataset_images_folder, img_filename))
            img = cv2.cvtColor(np.asarray(img.getdata(), dtype='uint8'),
                               cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(
                os.path.join(self.source_dataset_images_folder, img_filename))

        img = (img - mean * 255) / std
        img_output_path = os.path.join(check_dataset_image_folder,
                                       img_filename)
        cv2.imwrite(img_output_path, img)

        return

    def check_dataset_image_mean_std(self) -> None:
        """检查数据集图片均值和方差
        """
        print('Start check dataset image mean std')
        mean_std_file_output_path = os.path.join(self.temp_informations_folder,
                                                 'mean_std.json')
        if not os.path.exists(mean_std_file_output_path):
            print('No mean std file, return!')
            return
        else:
            with open(mean_std_file_output_path, 'r', encoding='utf8') as f:
                mean_std = json.load(f)
                mean = np.array(mean_std[0]['mean'])
                std = np.array(mean_std[0]['std'])
        check_dataset_image_folder = os.path.join(
            self.temp_informations_folder, 'check_mean_std')
        if os.path.exists(check_dataset_image_folder):
            shutil.rmtree(check_dataset_image_folder)
        check_output_path(check_dataset_image_folder)
        img_filenames = os.listdir(self.temp_images_folder)
        if self.target_dataset_annotations_check_count < len(img_filenames):
            random_check_images_list = random.sample(
                img_filenames, self.target_dataset_annotations_check_count)
        else:
            random_check_images_list = img_filenames
        pbar, update = multiprocessing_list_tqdm(
            random_check_images_list, desc='Check images mean and std')
        pool = multiprocessing.Pool(self.workers)
        for img_filename in random_check_images_list:
            pool.apply_async(func=self.per_image_standardization,
                             args=(img_filename, mean, std,
                                   check_dataset_image_folder),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        return

    def plot_true_box(self, task: str, task_class_dict: dict) -> None:
        """[绘制每张图片的真实框检测图]

        Args:
            task (str): [任务类型]
            task_class_dict (dict): [任务类别字典]
        """

        # 类别色彩
        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in range(len(task_class_dict['Target_dataset_class']))]
        # 统计各个类别的框数
        nums = [[]
                for _ in range(len(task_class_dict['Target_dataset_class']))]
        image_count = 0
        plot_true_box_success = 0
        plot_true_box_fail = 0
        total_box = 0
        for image in tqdm(self.target_dataset_check_images_list,
                          desc='Output check detection images'):
            image_path = os.path.join(self.temp_images_folder,
                                      image.image_name)
            output_image = cv2.imread(image_path)  # 读取对应标签图片
            for object in image.object_list:  # 获取每张图片的bbox信息
                if not len(object.box_xywh):
                    continue
                try:
                    nums[task_class_dict['Target_dataset_class'].index(
                        object.box_clss)].append(object.box_clss)
                    color = colors[
                        task_class_dict['Target_dataset_class'].index(
                            object.box_clss)]
                    # if dataset['target_annotation_check_mask'] == False:
                    if self.source_dataset_style == 'CVAT_IMAGE_BEV':
                        cv2.polylines(output_image,
                                      object.rotated_rect_points,
                                      isClosed=True,
                                      color=color,
                                      thickness=2)
                    else:
                        cv2.rectangle(
                            output_image,
                            (int(object.box_xywh[0]), int(object.box_xywh[1])),
                            (int(object.box_xywh[0] + object.box_xywh[2]),
                             int(object.box_xywh[1] + object.box_xywh[3])),
                            color,
                            thickness=2)
                        plot_true_box_success += 1
                        # 绘制透明锚框
                        # else:
                        #     zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                        #     zeros1_mask = cv2.rectangle(zeros1, (box.xmin, box.ymin),
                        #                                 (box.xmax, box.ymax),
                        #                                 color, thickness=-1)
                        #     alpha = 1   # alpha 为第一张图片的透明度
                        #     beta = 0.5  # beta 为第二张图片的透明度
                        #     gamma = 0
                        #     # cv2.addWeighted 将原始图片与 mask 融合
                        #     mask_img = cv2.addWeighted(
                        #         output_image, alpha, zeros1_mask, beta, gamma)
                        #     output_image = mask_img
                        #     plot_true_box_success += 1
                        cv2.putText(
                            output_image, object.box_clss,
                            (int(object.box_xywh[0]), int(object.box_xywh[1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                except:
                    print(image.image_name + ' ' + str(object.box_clss) +
                          " is not in {} class list".format(task))
                    plot_true_box_fail += 1
                    continue
                total_box += 1
                # 输出图片
            path = os.path.join(
                self.target_dataset_annotation_check_output_folder,
                image.image_name)
            cv2.imwrite(path, output_image)
            image_count += 1

        # 输出检查统计
        print("Total check annotations count: \t%d" % image_count)
        print('Check annotation true box count:')
        print("Plot true box success image: \t%d" % plot_true_box_success)
        print("Plot true box fail image:    \t%d" % plot_true_box_fail)
        print('True box class count:')
        for i in nums:
            if len(i) != 0:
                print(i[0] + ':' + str(len(i)))

        with open(
                os.path.join(
                    self.target_dataset_annotation_check_output_folder,
                    'detect_class_count.txt'), 'w') as f:
            for i in nums:
                if len(i) != 0:
                    temp = i[0] + ':' + str(len(i)) + '\n'
                    f.write(temp)
            f.close()

        return

    def plot_true_box_bev(self, task: str, task_class_dict: dict) -> None:
        """[绘制每张图片的bev真实框检测图]

        Args:
            task (str): [任务类型]
            task_class_dict (dict): [任务类别字典]
        """

        # 类别色彩
        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in range(len(task_class_dict['Target_dataset_class']))]
        # 统计各个类别的框数
        nums = [[]
                for _ in range(len(task_class_dict['Target_dataset_class']))]
        image_count = 0
        plot_true_box_success = 0
        plot_true_box_fail = 0
        total_box = 0
        for image in tqdm(self.target_dataset_check_images_list,
                          desc='Output check detection images'):
            image_path = os.path.join(self.temp_images_folder,
                                      image.image_name)
            output_image = cv2.imread(image_path)  # 读取对应标签图片

            for object in image.object_list:  # 获取每张图片的bbox信息
                if not len(object.box_xywh):
                    continue
                try:
                    nums[task_class_dict['Target_dataset_class'].index(
                        object.box_clss)].append(object.box_clss)
                    color = colors[
                        task_class_dict['Target_dataset_class'].index(
                            object.box_clss)]
                    points = np.int32(
                        [np.array(object.box_rotated_rect_points)])
                    if self.target_dataset_annotation_check_mask == False:
                        cv2.polylines(output_image,
                                      points,
                                      isClosed=True,
                                      color=color,
                                      thickness=2)
                        plot_true_box_success += 1
                    # 绘制透明锚框
                    else:
                        zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                        zeros1_mask = cv2.drawContours(zeros1, [points],
                                                       -1,
                                                       color=color,
                                                       thickness=-1)
                        alpha = 1  # alpha 为第一张图片的透明度
                        beta = 0.5  # beta 为第二张图片的透明度
                        gamma = 0
                        # cv2.addWeighted 将原始图片与 mask 融合
                        mask_img = cv2.addWeighted(output_image, alpha,
                                                   zeros1_mask, beta, gamma)
                        output_image = mask_img
                        plot_true_box_success += 1
                    cv2.putText(
                        output_image, object.box_clss,
                        (int(object.box_xywh[0]), int(object.box_xywh[1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                except:
                    print(image.image_name + ' ' + " erro in {}!".format(task))
                    plot_true_box_fail += 1
                    continue
                total_box += 1
                # 输出图片
            path = os.path.join(
                self.target_dataset_annotation_check_output_folder,
                image.image_name)
            cv2.imwrite(path, output_image)
            image_count += 1

        # 输出检查统计
        print("Total check annotations count: \t%d" % image_count)
        print('Check annotation true box count:')
        print("Plot true box success image: \t%d" % plot_true_box_success)
        print("Plot true box fail image:    \t%d" % plot_true_box_fail)
        print('True box class count:')
        for i in nums:
            if len(i) != 0:
                print(i[0] + ':' + str(len(i)))

        with open(
                os.path.join(
                    self.target_dataset_annotation_check_output_folder,
                    'detect_class_count.txt'), 'w') as f:
            for i in nums:
                if len(i) != 0:
                    temp = i[0] + ':' + str(len(i)) + '\n'
                    f.write(temp)
            f.close()

        return

    def plot_true_segmentation(self, task, task_class_dict) -> None:
        """[绘制每张图片的真实分割检测图]

        Args:
            task (str): [任务类型]
            task_class_dict (dict): [任务类别字典]
        """

        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in range(len(task_class_dict['Target_dataset_class']))
                  ]  # 类别色彩
        # 统计各个类别的框数
        nums = [[]
                for _ in range(len(task_class_dict['Target_dataset_class']))]
        image_count = 0
        plot_true_box_success = 0
        plot_true_box_fail = 0
        total_box = 0
        for image in tqdm(self.target_dataset_check_images_list,
                          desc='Output check semantic segmentation images'):
            if task == 'Instance_segmentation' or\
                    self.task_dict['Detection'] is not None:
                image_path = os.path.join(
                    self.target_dataset_annotation_check_output_folder,
                    image.image_name)
            else:
                image_path = os.path.join(self.temp_images_folder,
                                          image.image_name)
            output_image = cv2.imread(image_path)  # 读取对应标签图片
            for object in image.object_list:  # 获取每张图片的bbox信息
                if not len(object.segmentation):
                    continue
                # try:
                nums[task_class_dict['Target_dataset_class'].index(
                    object.segmentation_clss)].append(object.segmentation_clss)
                class_color = colors[
                    task_class_dict['Target_dataset_class'].index(
                        object.segmentation_clss)]
                if self.target_dataset_annotation_check_mask == False:
                    points = np.array(object.segmentation)
                    cv2.polylines(output_image,
                                  pts=[points],
                                  isClosed=True,
                                  color=class_color,
                                  thickness=2)
                    plot_true_box_success += 1
                # 绘制透明真实框
                else:
                    zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                    points = np.array(object.segmentation)
                    zeros1_mask = cv2.fillPoly(zeros1,
                                               pts=[points],
                                               color=class_color)
                    alpha = 1  # alpha 为第一张图片的透明度
                    beta = 0.5  # beta 为第二张图片的透明度
                    gamma = 0
                    # cv2.addWeighted 将原始图片与 mask 融合
                    mask_img = cv2.addWeighted(output_image, alpha,
                                               zeros1_mask, beta, gamma)
                    output_image = mask_img
                    plot_true_box_success += 1

                cv2.putText(output_image, object.segmentation_clss,
                            (int(object.segmentation[0][0]),
                             int(object.segmentation[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                # except:
                #     print(image.image_name + ' ' +
                #         str(object.segmentation_clss) + " is not in {} class list".format(task))
                #     plot_true_box_fail += 1
                #     continue
                total_box += 1
                # 输出图片
            path = os.path.join(
                self.target_dataset_annotation_check_output_folder,
                image.image_name)
            cv2.imwrite(path, output_image)
            image_count += 1

        # 输出检查统计
        print("Total check annotations count: \t%d" % image_count)
        print('Check annotation true box count:')
        print("Plot true segment success image: \t%d" % plot_true_box_success)
        print("Plot true segment fail image:    \t%d" % plot_true_box_fail)
        for i in nums:
            if len(i) != 0:
                print(i[0] + ':' + str(len(i)))

        with open(
                os.path.join(
                    self.target_dataset_annotation_check_output_folder,
                    'class_count.txt'), 'w') as f:
            for i in nums:
                if len(i) != 0:
                    temp = i[0] + ':' + str(len(i)) + '\n'
                    f.write(temp)
            f.close()

        return

    def plot_segmentation_annotation(
            self, task: str, task_class_dict: dict, image: IMAGE,
            segment_annotation_output_path: str) -> None:
        """[绘制分割标签图]

        Args:
            task (str): [任务类型]
            task_class_dict (dict): [任务类别字典]
            image (IMAGE): [图片类实例]
            segment_annotation_output_path (str): [分割标签图输出路径]
        """

        zeros = np.zeros((image.height, image.width), dtype=np.uint8)
        if len(image.true_segmentation_list):
            for seg in image.true_segmentation_list:
                class_color = task_class_dict['Target_dataset_class'].index(
                    seg.clss)
                points = np.array(seg.segmentation)
                zeros_mask = cv2.fillPoly(zeros,
                                          pts=[points],
                                          color=class_color)
                cv2.imwrite(segment_annotation_output_path, zeros_mask)
        else:
            cv2.imwrite(segment_annotation_output_path, zeros)

        return

    @staticmethod
    def TEMP_LOAD(dataset_instance, temp_annotation_path: str) -> IMAGE:
        """[读取暂存annotation]

        Args:
            dataset (dict): [数据集信息字典]
            temp_annotation_path (str): [annotation路径]

        Returns:
            IMAGE: [输出IMAGE类变量]
        """

        with open(temp_annotation_path, 'r') as f:
            data = json.loads(f.read())
            image_name = temp_annotation_path.split(os.sep)[-1].replace(
                '.json', '.' + dataset_instance.temp_image_form)
            image_path = os.path.join(dataset_instance.temp_images_folder,
                                      image_name)
            if not dataset_instance.only_statistic:
                if os.path.splitext(image_path)[-1] == '.png':
                    img = Image.open(image_path)
                    height, width = img.height, img.width
                    channels = 3
                else:
                    image_size = cv2.imread(image_path).shape
                    height = int(image_size[0])
                    width = int(image_size[1])
                    channels = int(image_size[2])
            else:
                height = int(data['base_information']['height'])
                width = int(data['base_information']['width'])
                channels = int(data['base_information']['channels'])
            object_list = []
            for object in data['frames'][0]['objects']:
                try:
                    one_object = OBJECT(
                        object['id'],
                        object['object_clss'],
                        box_clss=object['box_clss'],
                        segmentation_clss=object['segmentation_clss'],
                        keypoints_clss=object['keypoints_clss'],
                        box_xywh=object['box_xywh'],
                        box_xtlytlxbrybr=object['box_xtlytlxbrybr'],
                        box_rotation=object['box_rotation'],
                        box_head_point=object['box_head_point'],
                        box_head_orientation=object['box_head_orientation'],
                        segmentation=object['segmentation'],
                        keypoints_num=int(object['keypoints_num'])
                        if object['keypoints_num'] != '' else 0,
                        keypoints=object['keypoints'],
                        need_convert=dataset_instance.need_convert,
                        box_color=object['box_color'],
                        box_tool=object['box_tool'],
                        box_difficult=int(object['box_difficult'])
                        if object['box_difficult'] != '' else 0,
                        box_distance=float(object['box_distance'])
                        if object['box_distance'] != '' else 0.0,
                        box_occlusion=float(object['box_occlusion'])
                        if object['box_occlusion'] != '' else 0.0,
                        segmentation_area=int(object['segmentation_area'])
                        if object['segmentation_area'] != '' else 0,
                        segmentation_iscrowd=int(
                            object['segmentation_iscrowd'])
                        if object['segmentation_iscrowd'] != '' else 0,
                    )
                except EOFError as e:
                    print('未知错误: %s', e)
                object_list.append(one_object)

            if dataset_instance.get_local_map:
                # image_ego_pose
                image_ego_pose = data['image_ego_pose']
                image_time_stamp = dataset_instance.image_time_stamp_dict[
                    image_name]
                laneline_list = []
                for laneline in data['frames'][0]['lanelines']:
                    laneline_list.append(
                        LANELINE(laneline_id_in=laneline['id'],
                                 laneline_class_in=laneline['laneline_class'],
                                 laneline_points_utm_in=laneline[
                                     'laneline_points_utm'],
                                 laneline_points_label_image_in=laneline[
                                     'laneline_points_label_image']))
                image = IMAGE(image_name, image_name, image_path, height,
                              width, channels, object_list, image_ego_pose,
                              image_time_stamp, laneline_list)
            else:
                image = IMAGE(image_name, image_name, image_path, height,
                              width, channels, object_list)
            f.close()

        return image

    def transform_to_target_dataset():
        # print('\nStart transform to target dataset:')
        raise NotImplementedError("ERROR: func not implemented!")

    def target_dataset_annotation_check(self) -> None:
        """[进行标签检测]
        """

        print('\nStart check target annotations:')
        self.target_dataset_check_images_list = dataset.__dict__[
            self.target_dataset_style].annotation_check(self)
        shutil.rmtree(self.target_dataset_annotation_check_output_folder)
        check_output_path(self.target_dataset_annotation_check_output_folder)
        if 0 == len(self.target_dataset_check_images_list):
            print('No target dataset check images list.')
            return
        for task, task_class_dict in self.task_dict.items():
            if task == 'Detection' and task_class_dict is not None:
                if self.target_dataset_view == 'perspective':
                    self.plot_true_box(task, task_class_dict)
                else:
                    self.plot_true_box_bev(task, task_class_dict)
            elif task == 'Semantic_segmentation' and task_class_dict is not None:
                self.plot_true_segmentation(task, task_class_dict)
            elif (task == 'Instance_segmentation'
                  or task == 'Multi_task') and task_class_dict is not None:
                self.plot_true_box(task, task_class_dict)
                self.plot_true_segmentation(task, task_class_dict)

        return

    def plot_true_segment(dataset: dict) -> None:
        """[绘制每张图片的真实分割检测图]

        Args:
            dataset (dict): [Dataset类实例]
        """

        colors = [[random.randint(0, 255) for _ in range(3)]
                  for _ in range(len(dataset['segment_class_list_new']))
                  ]  # 类别色彩
        # 统计各个类别的像素点
        nums = [[] for _ in range(len(dataset['segment_class_list_new']))]
        image_count = 0
        plot_true_box_success = 0

        print('Output check true segmentation annotation images:')
        for image in tqdm(dataset['check_images_list']):
            image_path = os.path.join(
                dataset['check_annotation_output_folder'], image.image_name)
            output_image = cv2.imread(image_path)  # 读取对应标签图片
            for object in image.true_segmentation_list:  # 获取每张图片的bbox信息
                nums[dataset['segment_class_list_new'].index(
                    object.clss)].append(object.clss)
                class_color = colors[dataset['segment_class_list_new'].index(
                    object.clss)]
                if dataset['target_segment_annotation_check_mask'] == False:
                    points = np.array(object.segmentation)
                    cv2.polylines(output_image,
                                  pts=[points],
                                  isClosed=True,
                                  color=class_color,
                                  thickness=2)
                    plot_true_box_success += 1
                # 绘制透明分割真实框
                else:
                    zeros1 = np.zeros((output_image.shape), dtype=np.uint8)
                    points = np.array(object.segmentation)
                    zeros1_mask = cv2.fillPoly(zeros1,
                                               pts=[points],
                                               color=class_color)
                    alpha = 1  # alpha 为第一张图片的透明度
                    beta = 0.3  # beta 为第二张图片的透明度
                    gamma = 0
                    # cv2.addWeighted 将原始图片与 mask 融合
                    mask_img = cv2.addWeighted(output_image, alpha,
                                               zeros1_mask, beta, gamma)
                    output_image = mask_img
                    plot_true_box_success += 1

                cv2.putText(output_image, object.clss,
                            (int(object.segmentation[0][0]),
                             int(object.segmentation[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0))
                # 输出图片
            path = os.path.join(dataset['check_annotation_output_folder'],
                                image.image_name)
            cv2.imwrite(path, output_image)
            image_count += 1

        # 输出检查统计
        print("\nTotal check annotations count: \t%d" % image_count)
        print('Check annotation true segment count:')
        print("Plot true segment success image: \t%d" % plot_true_box_success)
        print("Plot true segment fail image:    \t%d" %
              (len(dataset['check_images_list']) - image_count))
        print('True box class count:')
        for i in nums:
            if len(i) != 0:
                print(i[0] + ':' + str(len(i)))

        with open(
                os.path.join(dataset['check_annotation_output_folder'],
                             'segment_class_count.txt'), 'w') as f:
            for i in nums:
                if len(i) != 0:
                    temp = i[0] + ':' + str(len(i)) + '\n'
                    f.write(temp)
            f.close()

        return

    def plot_segment_annotation(dataset: dict, image: IMAGE,
                                segment_annotation_output_path: str) -> None:
        """[绘制分割标签图]

        Args:
            dataset (dict): [数据集信息字典]
            image (IMAGE): [图片类实例]
            segment_annotation_output_path (str): [分割标签图输出路径]
        """

        zeros = np.zeros((image.height, image.width), dtype=np.uint8)
        if len(image.true_segmentation_list):
            for seg in image.true_segmentation_list:
                class_color = dataset['segment_class_list_new'].index(seg.clss)
                points = np.array(seg.segmentation)
                zeros_mask = cv2.fillPoly(zeros,
                                          pts=[points],
                                          color=class_color)
                cv2.imwrite(segment_annotation_output_path, zeros_mask)
        else:
            cv2.imwrite(segment_annotation_output_path, zeros)

        return

    def plot_pick_class_segment_annotation(dataset: dict, image: IMAGE,
                                           segment_annotation_output_path: str,
                                           class_list: list,
                                           lane_color: int) -> None:
        """[绘制分割标签图]

        Args:
            dataset (dict): [数据集信息字典]
            image (IMAGE): [图片类实例]
            segment_annotation_output_path (str): [分割标签图输出路径]
        """

        zeros = np.zeros((image.height, image.width), dtype=np.uint8)
        if len(image.true_segmentation_list):
            plot_true_segmentation_count = 0
            for seg in image.true_segmentation_list:
                if seg.clss not in class_list:
                    continue
                class_color = lane_color
                points = np.array(seg.segmentation)
                zeros_mask = cv2.fillPoly(zeros,
                                          pts=[points],
                                          color=class_color)
                cv2.imwrite(segment_annotation_output_path, zeros_mask)
                plot_true_segmentation_count += 1
            if 0 == plot_true_segmentation_count:
                cv2.imwrite(segment_annotation_output_path, zeros)
        else:
            cv2.imwrite(segment_annotation_output_path, zeros)

        return

    def build_target_dataset_folder():
        # print('\nStart build target dataset folder:')
        raise NotImplementedError("ERROR: func not implemented!")

    def distance_pixel_rate(self, real_distance: int,
                            pixel_distance: int) -> float:
        """计算真实距离与像素距离的换算关系(m/pixel)

        Args:
            real_distance (int): 真实距离
            pixel_distance (int): 像素距离

        Returns:
            float: m/pixel
        """
        return float(real_distance) / float(pixel_distance)

    def self_position_parse(self, camera_image_height: int,
                            label_image_width: int, label_image_height: int,
                            front_range: int, back_range: int, left_range: int,
                            right_range: int) -> tuple:
        """计算自车中心像素坐标(x,y)，以真实前距、左距比例计算

        Args:
            camera_image_height (int): 相机图像高度
            label_image_width (int): label图像宽度
            label_image_height (int): label图像高度
            front_range (int): 车辆前方距离
            back_range (int): 车辆后方距离
            left_range (int): 车辆左侧距离
            right_range (int): 车辆右侧距离

        Returns:
            tuple: (x,y)
        """
        # 计算本车中心点像素坐标(图片由上camera+下label组成)
        self_ycenter = label_image_height * \
            (front_range/(front_range+back_range)) + camera_image_height
        self_xcenter = label_image_width * (left_range /
                                            (left_range + right_range))

        return (int(self_xcenter), int(self_ycenter))
