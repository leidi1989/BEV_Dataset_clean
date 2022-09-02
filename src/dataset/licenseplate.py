'''
Description: 
Version: 
Author: Leidi
Date: 2021-10-13 18:36:09
LastEditors: Leidi
LastEditTime: 2021-10-22 16:43:06
'''
import os
import cv2

from utils.utils import *
from base.image_base import *
from utils.convertion_function import revers_yolo
from annotation.annotation_temp import TEMP_OUTPUT
from utils.modify_class import modify_true_box_list


def load_annotation(dataset: dict, src_lab_path_one: str,
                    local_mask_key_list: list, code_mask_key_list: list,
                    process_output) -> None:
    """[输出转换后的目标标签]

    Args:
        dataset (dict): [数据集信息字典]
        source_annotation_path (str): [源标签路径]
        process_output ([dict]): [进程通信字典]
    """

    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        dataset['file_prefix'] + src_lab_path_one)
    src_lab_dir = os.path.join(
        dataset['source_annotations_folder'], src_lab_path_one)
    with open(src_lab_dir, 'r') as f:
        truebox_dict_list = []
        for one_bbox in f.read().splitlines():
            bbox = one_bbox.split(' ')[1:]
            image_name = (src_lab_dir.split(
                '/')[-1]).replace('.txt', '.jpg')
            image_name_new = dataset['file_prefix'] + (src_lab_dir.split(
                '/')[-1]).replace('.txt', '.jpg')
            image_path = os.path.join(
                dataset['temp_images_folder'], image_name_new)
            img = cv2.imread(image_path)
            if img is None:
                print('Can not load: {}'.format(image_name_new))
                continue
            size = img.shape
            width = int(size[1])
            height = int(size[0])
            channels = int(size[2])
            cls = dataset['source_class_list'][int(one_bbox.split(' ')[0])]
            cls = cls.strip(' ').lower()
            if cls not in dataset['source_class_list']:
                continue
            if cls == 'dontcare' or cls == 'misc':
                continue
            bbox = revers_yolo(size, bbox)
            xmin = min(
                max(min(float(bbox[0]), float(bbox[1])), 0.), float(width))
            ymin = min(
                max(min(float(bbox[2]), float(bbox[3])), 0.), float(height))
            xmax = max(
                min(max(float(bbox[1]), float(bbox[0])), float(width)), 0.)
            ymax = max(
                min(max(float(bbox[3]), float(bbox[2])), float(height)), 0.)
            truebox_dict_list.append(TRUE_BOX(
                cls, xmin, ymin, xmax, ymax))  # 将单个真实框加入单张图片真实框列表
    if 7 != len(truebox_dict_list):
        return
    truebox_dict_list.sort(key=lambda x: x.xmin)

    # 更换真实框类别为车牌真实值
    real_classes_list = list(
        map(int, src_lab_path_one.split('-')[4].split('_')))
    classes_decode_list = []
    classes_decode_list.append(local_mask_key_list[real_classes_list[0]])
    for one in real_classes_list[1:]:
        classes_decode_list.append(code_mask_key_list[one])
    for truebox, classes in zip(truebox_dict_list, classes_decode_list):
        truebox.clss = classes
    image = IMAGE(image_name, image_name_new, image_path, int(
        height), int(width), int(channels), truebox_dict_list)

    # 将单张图对象添加进全数据集数据列表中
    temp_annotation_output_path = os.path.join(
        dataset['temp_annotations_folder'],
        image.file_name_new + '.' + dataset['temp_annotation_form'])
    modify_true_box_list(image, dataset['modify_class_dict'])
    if dataset['class_pixel_distance_dict'] is not None:
        class_pixel_limit(dataset, image.true_box_list)
    if 0 == len(image.true_box_list):
        print('{} has not true box, delete!'.format(image.image_name_new))
        os.remove(image.image_path)
        process_output['no_true_box'] += 1
        process_output['fail_count'] += 1
        return
    if TEMP_OUTPUT(temp_annotation_output_path, image):
        process_output['temp_file_name_list'].append(image.file_name_new)
        process_output['success_count'] += 1
    else:
        process_output['fail_count'] += 1
        return

    return
