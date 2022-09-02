'''
Description: 
Version: 
Author: Leidi
Date: 2022-08-07 15:49:04
LastEditors: Leidi
LastEditTime: 2022-08-07 18:17:19
'''
import json
import os
from PIL import Image
import cv2
from Clean_up.base.dataset_base import Dataset_Base
from Clean_up.base.image_base import IMAGE, OBJECT


def TEMP_LOAD(dataset_instance: Dataset_Base, temp_annotation_path: str) -> IMAGE:
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
                    segmentation_iscrowd=int(object['segmentation_iscrowd'])
                    if object['segmentation_iscrowd'] != '' else 0,
                )
            except EOFError as e:
                print('未知错误: %s', e)
            object_list.append(one_object)
        image = IMAGE(image_name, image_name, image_path, height, width,
                      channels, object_list)
        f.close()

    return image
