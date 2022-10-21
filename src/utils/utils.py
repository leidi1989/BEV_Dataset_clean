'''
Description: 
Version: 
Author: Leidi
Date: 2021-04-26 20:59:03
LastEditors: Leidi
LastEditTime: 2022-10-20 20:15:12
'''
# -*- coding: utf-8 -*-
import math
import os
import random

import numpy as np
from tqdm import tqdm


def check_input_path(path: str) -> str:
    """[检查输入路径是否存在]

    Args:
        path (str): [输入路径]

    Returns:
        str: [输入路径]
    """

    if os.path.exists(path):
        return path
    else:
        # assert path was found
        assert os.path.exists(path), 'Input path Not Found: %s' % path


def check_output_path(path: str, attach: str = '') -> str:
    """[检查输出路径是否存在，若不存在则创建路径]

    Args:
        path (str): [输出路径]
        attach (str, optional): [输出路径后缀]. Defaults to ''.

    Returns:
        str: [添加后缀的输出路径]
    """

    if os.path.exists(os.path.join(path, attach)):

        return os.path.join(path, attach)
    else:
        print('Create folder: ', os.path.join(path, attach))
        os.makedirs(os.path.join(path, attach))

        return os.path.join(path, attach)


def check_output_path_(path: str, attach: str = '') -> str:
    """[检查输出路径是否存在，若不存在则创建路径]

    Args:
        path (str): [输出路径]
        attach (str, optional): [输出路径后缀]. Defaults to ''.

    Returns:
        str: [添加后缀的输出路径]
    """

    if os.path.exists(os.path.join(path, attach)):

        return os.path.join(path, attach)
    else:
        os.makedirs(os.path.join(path, attach))

        return os.path.join(path, attach)


def check_out_file_exists(output_file_path: str) -> str:
    """[检查路径是否存在，存在则删除已存在路径]

    Args:
        output_file_path (str): [输出路径]

    Returns:
        str: [输出路径]
    """

    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    return output_file_path


def check_in_file_exists(input_file_path: str) -> str:
    """[检查路径是否存在]

    Args:
        input_file_path (str): [输入路径]

    Returns:
        str: [输入路径]
    """

    if os.path.exists(input_file_path):
        return input_file_path
    # assert path was found
    assert os.path.exists(
        input_file_path), 'Input path Not Found: %s' % input_file_path


def check_prefix(prefix: str, delimiter: str) -> str:
    """[检查是否添加前缀]

    Args:
        prefix (str): [前缀字符串]
        delimiter (str): [前缀字符串分割字符]

    Returns:
        str: [前缀字符串]
    """

    if prefix == '' or prefix is None:
        name_prefix = ''

        return name_prefix
    else:
        name_prefix = prefix + delimiter

        return name_prefix


def get_class_list(source_dataset_class_file_path: str) -> list:
    """[读取指定的class文件至class_list]

    Args:
        source_class_path (str): [源数据集类别文件路径]

    Returns:
        list: [数据集类别列表]
    """
    if source_dataset_class_file_path == '' or \
            source_dataset_class_file_path is None:
        return None
    class_list = []  # 类别列表
    with open(source_dataset_class_file_path, 'r') as class_file:
        for one_line in class_file.read().splitlines():
            re_one_line = one_line.replace(' ', '').lower()
            re_one_line = ''.join(list(filter(str.isalpha, re_one_line)))
            class_list.append(re_one_line)

    return class_list


def get_modify_class_dict(modify_class_file_path: str) -> dict or None:
    """[按modify_class_dict_class_file文件对类别进行融合，生成标签融合字典]

    Args:
        modify_class_file_path (str): [修改类别文件路径]

    Returns:
        dict or None: [输出修改类别字典]
    """

    if modify_class_file_path == '' or \
            modify_class_file_path is None:
        return None
    modify_class_dict = {}  # 修改类别对应字典
    with open(modify_class_file_path, 'r') as class_file:
        for one_line in class_file.read().splitlines():  # 获取class修改文件内容
            one_line_list = one_line.rstrip(' ').lower().split(':')
            one_line_list[0] = ''.join(
                list(filter(str.isalpha, one_line_list[0])))
            one_line_list[-1] = one_line_list[-1].split(' ')  #
            # 细分类别插入修改后class类别，类别融合
            one_line_list[-1].insert(0, one_line_list[0])
            for one_class in one_line_list[1]:  # 清理列表中的空值
                if one_class == '':
                    one_line_list[1].pop(one_line_list[1].index(one_class))
            for one_class in one_line_list[1]:  
                one_line_list[1][one_line_list[1].index(one_class)] = ''.join(list(filter(str.isalpha, one_class)))
            key = one_line_list[0]
            modify_class_dict[key] = one_line_list[1]  # 将新类别添加至融合类别字典

    return modify_class_dict


def get_new_class_names_list(source_class_list: list,
                             modify_class_dict: dict) -> list:
    """[获取融合后的类别列表]

    Args:
        sr_classes_list (list): [源数据集类别列表]
        modify_class_dict (dict): [数据集类别修改字典]

    Returns:
        list: [目标数据集类别列表]
    """

    if modify_class_dict is None:
        return source_class_list
    else:
        new_class_names_list = []  # 新类别列表
        for key in modify_class_dict.keys():  # 依改后类别字典，生成融合后的类别列表
            new_class_names_list.append(key.replace(' ', ''))

        return new_class_names_list


def get_temp_annotations_name_list(temp_annotations_folder: str) -> list:
    """[获取暂存数据集全量文件名称列表]

    Args:
        temp_annotations_folder (str): [暂存数据集标签文件夹]

    Returns:
        list: [暂存数据集全量文件名称列表]
    """

    temp_file_name_list = []  # 暂存数据集全量文件名称列表
    print('Get temp file name list.')
    for n in os.listdir(temp_annotations_folder):
        temp_file_name_list.append(os.path.splitext(n.split(os.sep)[-1])[0])

    return temp_file_name_list


def class_box_pixel_limit(dataset: dict, true_box_list: list) -> None:
    """[按类别依据类类别像素值选择区间与距离文件对真实框进行筛选]

    Args:
        dataset (dict): [数据集信息字典]
        true_box_list (list): [真实框类实例列表]
    """

    for n in true_box_list:
        pixel = (n.xmax - n.xmin) * (n.ymax - n.ymin)
        if pixel < dataset['class_pixel_distance_dict'][n.clss][0] or \
                pixel > dataset['class_pixel_distance_dict'][n.clss][1]:
            true_box_list.pop(true_box_list.index(n))

    return


def polygon_area(points: list) -> int:
    """[返回多边形面积]

    Args:
        points (list): [多边形定点列表]

    Returns:
        int: [多边形面积]
    """

    area = 0  # 多边形面积
    q = points[-1]
    for p in points:
        area += p[0] * q[1] - p[1] * q[0]
        q = p

    return abs(int(area / 2))


def class_segmentation_pixel_limit(dataset: dict,
                                   true_segmentation_list: list) -> None:
    """[按类别依据类类别像素值选择区间与距离文件对真实框进行筛选]

    Args:
        dataset (dict): [数据集信息字典]
        true_segmentation_list (list): [真实分割列表]
    """

    for n in true_segmentation_list:
        pixel = polygon_area(n)
        if pixel < dataset['class_pixel_distance_dict'][n.clss][0] or \
                pixel > dataset['class_pixel_distance_dict'][n.clss][1]:
            true_segmentation_list.pop(true_segmentation_list.index(n))

    return


def temp_annotations_path_list(temp_annotations_folder: str) -> list:
    """[获取暂存数据集全量标签路径列表]

    Args:
        temp_annotations_folder (str): [暂存数据集标签文件夹路径]

    Returns:
        list: [暂存数据集全量标签路径列表]
    """

    temp_annotation_path_list = []  # 暂存数据集全量标签路径列表
    print('Get temp annotation path.')
    for n in os.listdir(temp_annotations_folder):
        temp_annotation_path_list.append(
            os.path.join(temp_annotations_folder, n))

    return temp_annotation_path_list


def total_file(temp_informations_folder: str) -> list:
    """[获取暂存数据集全量图片文件名列表]

    Args:
        temp_informations_folder (str): [暂存数据集信息文件夹]

    Returns:
        list: [暂存数据集全量图片文件名列表]
    """

    total_list = []  # 暂存数据集全量图片文件名列表
    print('Get total file name list.')
    try:
        with open(os.path.join(temp_informations_folder, 'total.txt'),
                  'r') as f:
            for n in f.read().splitlines():
                total_list.append(os.path.splitext(n.split(os.sep)[-1])[0])
            f.close()

        total_file_name_path = os.path.join(temp_informations_folder,
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


def annotations_path_list(total_file_name_path: str,
                          target_annotation_check_count: int) -> list:
    """[获取检测标签路径列表]

    Args:
        total_file_name_path (str): [标签文件路径]
        target_annotation_check_count (int): [检测标签文件数量]

    Returns:
        list: [检测标签路径列表]
    """

    print('Get check file name:')
    file_name = []  # 检测标签路径列表
    with open(total_file_name_path, 'r') as f:
        for n in tqdm(f.read().splitlines()):
            file_name.append(n)
        random.shuffle(file_name)

    return file_name[0:target_annotation_check_count]


def get_src_total_label_path_list(total_label_path_list: list,
                                  image_type: str) -> list:
    """[获取源数据集全部标签路径列表]

    Args:
        total_label_path_list (list): [源数据集全部标签路径]
        image_type (str): [图片类别]

    Returns:
        list: [数据集全部标签路径列表]
    """

    src_total_label_path_list = []  # 数据集全部标签路径列表
    with open(total_label_path_list, 'r') as total:  # 获取数据集全部源标签列表
        for one_line in total.read().splitlines():  # 获取源数据集labels路径列表
            src_total_label_path_list.append(
                one_line.replace('JPEGImages',
                                 'labels').replace('.' + image_type, '.txt'))

    return src_total_label_path_list


def get_dataset_scene(dateset_list: list) -> tuple:
    """[获取数据集场景元组]

    Args:
        dateset_list (list): [数据集annotation提取数据]

    Returns:
        tuple: [数据集按名称分类场景元组]
    """

    class_list = []  # 数据集按名称分类场景列表
    for one in dateset_list:
        image_name_list = (one.image_name).split('_')  # 对图片名称进行分段，区分场景
        image_name_str = ''
        for one_name in image_name_list[:-1]:  # 读取切分图片名称的值，去掉编号及后缀
            image_name_str += one_name  # target_image_name_str为图片包含场景的名称
        class_list.append(image_name_str)

    return set(class_list)


def cheak_total_images_data_list(total_images_data_list: list) -> list:
    """[检查总图片信息列表，若图片无真实框则剔除]

    Args:
        total_images_data_list ([list]): [总图片信息列表]

    Returns:
        [list]: [返回清理无真实框图片后的总图片列表]
    """

    new_total_images_data_list = []  # 返回清理无真实框图片后的总图片列表
    total = len(total_images_data_list)
    empty = 0
    print("\nStart cheak empty annotation.")
    for one_image_data in tqdm(total_images_data_list):
        if 0 != (len(one_image_data.true_box_list)):
            new_total_images_data_list.append(one_image_data)
        else:
            empty += 1
    total_last = total - empty
    print("\nUsefull images: {n} \nEmpty images: {m}\n".format(n=total_last,
                                                               m=empty))

    return new_total_images_data_list


def delete_empty_images(output_path: str, total_label_list: list) -> None:
    """[删除无labels的图片]

    Args:
        output_path ([str]): [输出数据集路径]
        total_label_list ([list]): [全部labels列表]
    """

    images_path = os.path.join(output_path, 'JPEGImages')
    total_images_list = os.listdir(images_path)
    labels_to_images_list = []
    for n in total_label_list:
        n += '.jpg'
        labels_to_images_list.append(n)
    need_delete_images_list = [
        y for y in total_images_list if y not in labels_to_images_list
    ]
    print('\nStart delete no ture box images:')
    delete_images = 0
    if 0 != len(need_delete_images_list):
        for delete_one in tqdm(need_delete_images_list):
            delete_image_path = os.path.join(images_path, delete_one)
            if os.path.exists(delete_image_path):
                os.remove(delete_image_path)
                delete_images += 1
    else:
        print("\nNo images need delete.")
    print("\nDelete images: %d" % delete_images)

    return


def delete_empty_ann(output_path: str, total_label_list: list) -> None:
    """[删除无labels的annotations]

    Args:
        output_path ([str]): [输出数据集路径]
        total_label_list ([list]): [全部labels列表]
    """

    annotations_path = os.path.join(output_path, 'Annotations')
    total_images_list = os.listdir(annotations_path)
    labels_to_annotations_list = []

    for n in total_label_list:
        n += '.xml'
        labels_to_annotations_list.append(n)
    need_delete_annotations_list = [
        y for y in total_images_list if y not in labels_to_annotations_list
    ]
    print('\nStart delete no ture box images:')
    delete_annotations = 0
    if 0 != len(need_delete_annotations_list):
        for delete_one in tqdm(need_delete_annotations_list):
            delete_annotation_path = os.path.join(annotations_path, delete_one)
            if os.path.exists(delete_annotation_path):
                os.remove(delete_annotation_path)
                delete_annotations += 1
    else:
        print("\nNo annotation need delete.")
    print("\nDelete annotations: %d" % delete_annotations)

    return


def get_class_pixel_limit(class_pixel_distance_file_path: str) -> dict:
    """[读取类别像素大小限制文件]

    Args:
        class_pixel_distance_file_path (str): [类别像素大小限制文件路径]

    Returns:
        dict or None: [类别像素大小限制字典]
    """

    if class_pixel_distance_file_path == '' \
            or class_pixel_distance_file_path == None:
        return None

    class_pixel_limit_dict = {}  # 类别像素大小限制字典
    with open(class_pixel_distance_file_path, 'r') as f:
        for n in f.read().splitlines():
            key = n.split(':')[0]
            value = n.split(':')[1]
            pixel_rang = list(map(int, value.split(',')))
            if pixel_rang[0] < pixel_rang[1]:
                class_pixel_limit_dict[key] = list(map(int, value.split(',')))
            else:
                print('Class pixel distance file wrong!')
                print('Unlimit true box pixel.')
                return None

    return class_pixel_limit_dict


def bdd100k_line_sort(line_1: list, line_2: list) -> int:
    """[bdd100k提取车道线列表排序]

    Args:
        line_point_list_1 (list): [车道线1]
        line_point_list_2 (list): [车道线2]

    Returns:
        int: [排序方式]
    """
    return line_1['line_point_list'][0][0] - line_2['line_point_list'][0][0]


def calNextPoints(points: list, rate: float) -> list:
    """[通过递归构造贝塞尔曲线，如果给定了具体的n， 那么可以直接得到计算方程]

    Args:
        points (list): [贝塞尔曲线起点及3个控制点]
        rate (float): [出点率]

    Returns:
        list: [曲线上的点x，y列表]
    """

    if len(points) == 1:
        return points

    left = points[0]
    ans = []
    # 根据比例计算当前的点的坐标，一层层的推进
    for i in range(1, len(points)):
        right = points[i]
        disX = right[0] - left[0]
        disY = right[1] - left[1]

        nowX = left[0] + disX * rate
        nowY = left[1] + disY * rate
        ans.append([nowX, nowY])

        # 更新left
        left = right

    return calNextPoints(ans, rate)


def err_call_back(err):
    """[报错回调函数]

    Args:
        err ([type]): [错误原因]
    """

    print(f'error：{str(err)}')


def dist(a: np.array, b: np.array) -> np.array:
    """[计算向量欧式距离]

    Args:
        a (np.array): [向量a]
        b (np.array): [向量b]

    Returns:
        np.array: [向量欧式距离]
    """

    return np.sqrt(sum((a - b)**2))


def RGB_to_Hex(tmp: int) -> str:
    """[RGB转换为16进制色彩]

    Args:
        tmp (int): [单值输入]

    Returns:
        str: [16进制色彩]
    """
    # 将RGB格式划分开来
    rgb = tmp.split(',')
    strs = '#'
    for i in rgb:
        num = int(i)  # 将str转int
        # 将R、G、B分别转化为16进制拼接转换并大写
        strs += str(hex(num))[-2:].replace('x', '0').upper()

    return strs


def multiprocessing_list_tqdm(file_list: list,
                              desc: str = '',
                              position: int = None,
                              leave: bool = True):
    """[多进程列表tqdm]

    Args:
        file_list (list): [计数文件列表]
        topic (str, optional): [主题]. Defaults to ''.
        position (int, optional): [位置]. Defaults to None.
        leave (bool, optional): [是否留屏]. Defaults to True.

    Returns:
        [type]: [description]
    """

    pbar = tqdm(total=len(file_list),
                position=position,
                leave=leave,
                desc=desc)

    return pbar, lambda *args: pbar.update()


def multiprocessing_object_tqdm(count: int,
                                desc: str = '',
                                position: int = None,
                                leave: bool = True):
    """[多进程计数tqdm]

    Args:
        count (int): [计数总量]
        topic (str, optional): [主题]. Defaults to ''.
        position (int, optional): [位置]. Defaults to None.
        leave (bool, optional): [是否留屏]. Defaults to True.

    Returns:
        [type]: [pbar, lambda *args: pbar.update()]
    """

    pbar = tqdm(total=count, position=position, leave=leave, desc=desc)

    return pbar, lambda *args: pbar.update()


def calculate_distance(a_x: float, a_y: float, b_x: float,
                       b_y: float) -> float:
    """计算两点之间的距离

    Args:
        a_x (float): a点x
        a_y (float): a点y
        b_x (float): b点x
        b_y (float): b点x

    Returns:
        float: a, b两点距离
    """

    return ((a_x - b_x)**2 + (a_y - b_y)**2)**0.5


def utm_to_bev(lane_lines_utm: dict, utm_x: float, utm_y: float, att_z: float,
               label_image_wh: list, label_range: list,
               label_image_self_car_uv: list) -> dict:
    """utm坐标转换为检测距离内的bev图像坐标

    Args:
        lane_lines_utm (dict): utm线段坐标字典
        utm_x (float): utm_x
        utm_y (float): utm_y
        att_z (float): att_z
        label_image_wh (list): 标注图片宽高
        label_range (list): 标注范围前后左右
        label_image_self_car_uv (list): 租车图像坐标

    Returns:
        dict: lane_lines_bev_image
    """

    lane_lines_bev_image = {}
    for id, line in lane_lines_utm.items():
        temp_line = []
        for point in line:
            # pose[3], pose[4], pose[8]对应车辆后轴中心的x, y, position_type(即车辆的转向角theta)
            x_ = (point[0] - utm_x) * np.cos(att_z) + (point[1] -
                                                       utm_y) * np.sin(att_z)
            y_ = (point[1] - utm_y) * np.cos(att_z) - (point[0] -
                                                       utm_x) * np.sin(att_z)
            #车身坐标系转像素坐标系
            u = int(label_image_self_car_uv[0] - y_ * label_image_wh[0] /
                    (label_range[2] + label_range[3]))
            v = int(label_image_self_car_uv[1] - x_ * label_image_wh[1] /
                    (label_range[0] + label_range[1]))
            # clip
            # u = int(np.clip(u, 0, label_image_wh[0]))
            # v = int(np.clip(v, 0, label_image_wh[1]))

            temp_line.append([u, v])
        lane_lines_bev_image[id] = temp_line

    return lane_lines_bev_image