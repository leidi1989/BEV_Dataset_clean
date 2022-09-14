#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import hashlib
import json
import math
import os
import re
import shutil
import uuid
import xml.etree.ElementTree as ET

import cv2
import lanelet2
import numpy as np
from tqdm import tqdm

# from data.const import Lat_Lon_Origin

Lat_Lon_Origin = {
    'wuhan': {
        'lat': 30.425457029885372151,
        'lon': 114.09623523096009023
    },
    'shenzhen': {
        'lat': 22.6860589,
        'lon': 114.3779897
    },
}

# TODO:根据需要增加传感器
level_01_cameras_name = [
    'cam_left_front', 'cam_front_center', 'cam_right_front', 'cam_left_back',
    'cam_back', 'cam_right_back'
]
level_02_cameras_name = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
    'CAM_BACK', 'CAM_BACK_RIGHT'
]
# TODO:根据需要增加类别
box_types = [
    'mtruck', 'truck', 'ltruck', 'wtruck', 'sixtruck', 'seventruck',
    'eighttruck', 'ninetruck', 'tentruck', 'twelvetruck', 'fourteentruck',
    'sixteentruck', 'seventeentruck', 'eighteentruck', 'twentytruck',
    'twentyonetruck', 'twentythreetruck', 'van', 'mvan', 'lvan', 'pickup',
    'bus', 'twelvebus', 'elevenbus', 'tenbus', 'ninebus', 'eightbus',
    'sevenbus', 'sixbus', 'cyclist', 'car', 'suvcar', 'mini', 'vehicles',
    'person', 'three-wheel', 'handcart', 'oil-tanker', 'sixoil-tanker',
    'sevenoil-tanker', 'tenoil-tanker', 'fifteenoil-tanker', 'cement-tanker',
    'msprinkler', 'sprinkler', 'eight-sprinkler', 'cleaning-truck',
    'garbage-truck', 'mdigging', 'digging', 'concrete-mixer',
    'emergency-rescue', 'ambulance', 'engineering-vehicle', 'tram',
    'bicycleGroup', 'conebarrel', 'nineteentruck', 'thirteentruck'
]

osm_names = [
    '/maps/osm/df2xc.osm',
    '/maps/osm/xc2df.osm',
]  #'/maps/osm/jszx2wd/jszx2wd.osm', '/maps/osm/wd2jszx/wd2jszx.osm', '/maps/osm/dfgs_jszx/jszx2dfgs.osm',  '/maps/osm/wxc2wd/nad.osm',]
# osm_names = ['/maps/osm/byd_zhoubian.osm']
map_path = '/mnt/data_1/Dataset/dataset_temp/to_nuscenes/demo_level02'
proj = lanelet2.projection.UtmProjector(
    lanelet2.io.Origin(Lat_Lon_Origin['wuhan']['lat'],
                       Lat_Lon_Origin['wuhan']['lon']))
lanelet_layers = {}
for osm_name in osm_names:
    lanelet_layers[osm_name] = lanelet2.io.load(map_path + osm_name,
                                                proj).laneletLayer

lanelet_lineStringLayer = {}
for osm_name in osm_names:
    lanelet_lineStringLayer[osm_name] = lanelet2.io.load(
        map_path + osm_name, proj).lineStringLayer

pass


# ------------------------------------------------------01 tools
# 图像加上mask（红旗1号车）
def img_mask(cam, img_path_):
    img = cv2.imread(img_path_)
    if cam == 'CAM_FRONT_LEFT':
        pts = np.array([[[1034.98, 719], [1103.76, 507.58], [1162.10, 415.65],
                         [1219.50, 412.20], [1233.70, 395.30],
                         [1279.00, 398.48], [1279, 719]]], np.int32)
        pts2 = np.array([[[0, 0], [12.16, 4.64], [39.56, 206.17],
                          [53.70, 348.48], [73.15, 525.25], [97.01, 603.04],
                          [114.69, 619.83], [370.38, 719.00], [0, 719]]],
                        np.int32)
        pts = pts.reshape((-1, 1, 2))
        pts2 = pts2.reshape((-1, 1, 2))
        img = cv2.fillConvexPoly(img, pts, (0, 0, 0))
        img = cv2.fillConvexPoly(img, pts2, (0, 0, 0))
    elif cam == 'CAM_FRONT':
        pts1 = np.array(
            [[[0, 719], [0.00, 699.80], [1.14, 681.89], [120.78, 663.95],
              [171.03, 661.56], [282.29, 663.35], [467.74, 669.93],
              [569.43, 668.73], [572.38, 719.00]]], np.int32)
        pts2 = np.array(
            [[[955.69, 719.00], [963.35, 662.87], [1103.93, 655.09],
              [1220.58, 641.93], [1279.00, 640.75], [1279, 719]]], np.int32)
        pts3 = np.array(
            [[[569.37, 719.00], [569.37, 600.54], [593.90, 562.25],
              [907.96, 556.87], [964.79, 590.97], [961.22, 719.00]]], np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        pts2 = pts2.reshape((-1, 1, 2))
        pts3 = pts3.reshape((-1, 1, 2))
        img = cv2.fillConvexPoly(img, pts1, (0, 0, 0))
        img = cv2.fillConvexPoly(img, pts2, (0, 0, 0))
        img = cv2.fillConvexPoly(img, pts3, (0, 0, 0))
    elif cam == 'CAM_FRONT_RIGHT':
        pts = np.array(
            [[[0, 719], [0.14, 415.79], [20.76, 432.73], [68.64, 437.15],
              [67.20, 437.80], [95.15, 479.87], [106.94, 502.70],
              [124.62, 547.63], [165.13, 677.27], [182.07, 719.99]]], np.int32)
        pts2 = np.array([[[917.91, 717.78], [1140.35, 641.18],
                          [1240.53, 585.93], [1279.00, 428.73], [1279, 719]]],
                        np.int32)
        pts = pts.reshape((-1, 1, 2))
        pts2 = pts2.reshape((-1, 1, 2))
        img = cv2.fillConvexPoly(img, pts, (0, 0, 0))
        img = cv2.fillConvexPoly(img, pts2, (0, 0, 0))
    elif cam == 'CAM_BACK_LEFT':
        pts = np.array([[[0, 719], [0.21, 520.01], [25.25, 542.84],
                         [68.71, 577.46], [102.59, 615.03], [120.30, 647.40],
                         [126.16, 667.32], [353.03, 719.62]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.fillConvexPoly(img, pts, (0, 0, 0))
    elif cam == 'CAM_BACK_RIGHT':
        pts = np.array([[[1279, 719], [862.66, 719.03], [1102.79, 660.11],
                         [1133.72, 570.24], [1149.93, 531.21],
                         [1214.00, 477.40], [1279.00, 396.24]]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        img = cv2.fillConvexPoly(img, pts, (0, 0, 0))
    cv2.imwrite(img_path_, img)


# 根据标注结果生成标注框的四个顶点
def rotated_rec(xtl, ytl, xbr, ybr, angle):
    width, height = abs(xtl - xbr), abs(ytl - ybr)
    if width < height:
        width, height = height, width
    center_x, center_y = (xtl + xbr) / 2, (ytl + ybr) / 2
    angle = (270 - angle) * math.pi / 180  # 弧度
    xo = np.cos(angle)
    yo = np.sin(angle)
    y1 = center_y + height / 2 * yo
    x1 = center_x - height / 2 * xo
    y2 = center_y - height / 2 * yo
    x2 = center_x + height / 2 * xo
    return [
        [(x2 + width / 2 * yo), (y2 + width / 2 * xo)],  #对应于旋转后车头左边，左上
        [(x1 + width / 2 * yo), (y1 + width / 2 * xo)],  #对应于旋转后车头右边，右上
        [(x1 - width / 2 * yo), (y1 - width / 2 * xo)],  #对应于旋转后车尾右边，右下
        [(x2 - width / 2 * yo), (y2 - width / 2 * xo)],  #对应于旋转后车尾左边，左下
    ]


# 解析xml
def parse_xml(xml_path):  # TODO: 待完善，目前只有解析车辆障碍物Box部分
    """
    解析xml文件
        参数:
            xml_path:xml文件路径
        返回:
            image:读取xml文件生成的字典
    """
    images_anno = {}
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for child in root:
        for point in child.findall('box'):
            visibility, name = '', ''
            for attribute in point.findall('attribute'):
                if attribute.attrib['name'] == 'visibility':
                    visibility = attribute.text
                else:
                    name = attribute.text
            if not name:
                continue
            if child.get('name') not in images_anno:
                images_anno[child.get('name')] = {}
            if name not in images_anno[child.get('name')]:
                images_anno[child.get('name')][name] = {}
            images_anno[child.get('name')][name]['key_points'] = [[
                float(point.get('xtl')),
                float(point.get('ytl'))
            ], [float(point.get('xbr')),
                float(point.get('ybr'))]]
            if point.get('rotation'):
                images_anno[child.get('name')][name]['rotation'] = float(
                    point.get('rotation'))
            if not point.get('rotation'):
                images_anno[child.get(
                    'name')][name]['rotation'] = 0.0  #标注时没有旋转
                # print(f'rotation doesnot exist! in {name} ', child.get('name'))
            if visibility:
                images_anno[
                    child.attrib['name']][name]['visibility'] = visibility
            for point in child.findall('points'):
                for attribute in point.findall('attribute'):
                    if attribute.attrib['name'] == 'visibility':
                        # visibility = attribute.text
                        continue
                    else:
                        p_name = attribute.text
                if not p_name:
                    continue
                if len(re.split('([0-9]+)', name)) == 1:
                    img_name = child.get('name')
                    # print(f'can not split {name} in {img_name}')
                    continue
                if re.split('([0-9]+)', name)[1] == p_name.split('-')[0]:
                    p_head = point.get('points').split(',')
                    images_anno[child.get('name')][name]['head'] = [
                        float(p_head[0]), float(p_head[1])
                    ]
                else:
                    continue
    # images_anno = dict(sorted(images_anno.items(), key=lambda x: x[0]))
    # out_json = open(os.path.join(data_path + 'anno.json'), 'w')
    # json.dump(images_anno, out_json, indent=4)
    # out_json.close()
    return images_anno


def split_xml(xml_path, save_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    task_dic = {}
    child = root[1]
    for project in child.findall('project'):
        for tasks in project.findall('tasks'):
            for task in tasks.findall('task'):
                task_dic[task.find('id').text] = task.find('name').text
    print(task_dic)
    for value in task_dic.values():
        #初始化xml
        temp_tree = ET.parse(
            '/mnt/data_1/Dataset/dataset_temp/map_c/dftc2ysg2wd_20220607_hq1/annotations.xml'
        )
        changjing = value.split("_")[-4]
        bag_range = value.split("_")[-3]
        # changjing=value.split("_")[-2]          #wd2jszx0730
        # bag_range=value.split("_")[-1]        #wd2jszx0730
        temp_tree.write(
            os.path.join(save_path, changjing + "_" + bag_range + ".xml"))
    temp_dic = {}
    for child in root:
        if child.tag != 'image':
            continue
        task_name = task_dic[child.get('task_id')]
        changjing = task_name.split("_")[-4]
        bag_range = task_name.split("_")[-3]
        # changjing=task_name.split("_")[-2]          #wd2jszx0730
        # bag_range=task_name.split("_")[-1]        #wd2jszx0730
        xml_path_1 = os.path.join(save_path,
                                  changjing + "_" + bag_range + ".xml")
        if xml_path_1 not in temp_dic:
            temp_dic[xml_path_1] = []
            temp_dic[xml_path_1].append(child)
        else:
            temp_dic[xml_path_1].append(child)
    for key, values in temp_dic.items():
        new_tree = ET.parse(key)
        new_root = new_tree.getroot()
        for value in values:
            new_root.append(value)
        new_tree.write(key)


# ------------------------------------------------------02 从原始数据生成AthenaDataset格式的数据
# 0.将图像数据加上mask，重命名后copy到level_02的对应路径下
def img_copy(level_01_data_path, outer_scene, level_02_data_path):
    # 在level_02的sample下生成各个传感器的目录
    for i in range(6):
        os.makedirs(level_02_data_path + '/' + level_02_cameras_name[i],
                    exist_ok=True)
    if not os.path.isdir(level_01_data_path + outer_scene + '/sync_data/'):
        print('(img_copy)Error:  ' + level_01_data_path + outer_scene +
              '/sync_data/  does not exist !')
        return
    scenes = sorted(
        os.listdir(level_01_data_path + outer_scene + '/sync_data/'))
    scenes = sorted([int(s) for s in scenes if len(s) < 3])
    print(outer_scene, 'start to mask and copy images ! ')
    # for scene in tqdm.tqdm(scenes):
    for scene in scenes:
        for i in range(6):
            if scene != ".DS_Store" and os.path.isdir(
                    level_01_data_path + outer_scene + '/sync_data/' +
                    str(scene) + '/' + level_01_cameras_name[i]):
                samples = sorted(
                    os.listdir(level_01_data_path + outer_scene +
                               '/sync_data/' + str(scene) + '/' +
                               level_01_cameras_name[i]))
                for sample in samples:
                    if sample[-1] == 'g':
                        #规范命名后直接使用outer_scene（场景路线_采集日期_采集车型）后续可加入天气状况
                        new_img_name = sample.split(
                            ".")[0] + "_" + outer_scene + "_" + str(
                                scene) + ".jpg"  #目前的outer_scene为采集路线+采集日期+采集车型
                        # copy and rename img
                        shutil.copyfile(
                            level_01_data_path + outer_scene + '/sync_data/' +
                            str(scene) + '/' + level_01_cameras_name[i] + "/" +
                            sample, level_02_data_path + '/' +
                            level_02_cameras_name[i] + "/" + new_img_name)
                        # mask img
                        img_mask(
                            level_02_cameras_name[i],
                            level_02_data_path + '/' +
                            level_02_cameras_name[i] + "/" + new_img_name)
        print("success " + outer_scene + " " + str(scene))
    print("mask, copy and rename img done")


# 1.0 BEV像素空间转换为车体坐标系(2D,xy对应于前左)
def uv_to_xy(points):
    """
    像素坐标系到自车坐标系的转换.
    points: list, shape=(n, 2), [[u1, v1], [u2, v2], ...]
    """
    ans = []
    for u, v in points:
        x = (1200 + 4000 - v) * 0.025
        y = (1600 - u) * 0.025

        # x = (4000 - v) * 0.025
        # y = (1600 - u) * 0.025
        ans.append([x, y])
    return ans


# 1.根据annotation.xml生成vehicle.json
def generate_vehicle_json(data_path, images, outer_scene, scene_id):
    vehicle_json = {}
    for key_id, image in tqdm(images.items(), desc='Get image', leave=False):
        point_temp = {}
        for name, obstacle in image.items():
            xtl, ytl = obstacle['key_points'][0]
            xbr, ybr = obstacle['key_points'][1]
            # if (obstacle['rotation']-0.0)< 1e-3:
            #     poly_region = rotated_rec(xtl,ytl,xbr,ybr,-90.0)
            # else:
            poly_region = rotated_rec(xtl, ytl, xbr, ybr, obstacle['rotation'])
            ufl, vfl = poly_region[0]
            ufr, vfr = poly_region[1]
            head_uv = [(ufl + ufr) / 2, (vfl + vfr) / 2]
            # head_uv = [round((ufl+ufr)/2),round((vfl+vfr)/2)]
            poly_region_np = np.array(poly_region, np.float32)
            center_uv = poly_region_np.mean(axis=0)
            # center_uv = np.asarray(poly_region_np.mean(axis=0)).astype(np.int32)
            #计算yaw角(法1)
            u1, v1 = -(np.array(head_uv, np.float32) - center_uv)
            cal_yaw = math.atan2(u1, v1)
            #计算yaw角(法2), 使用此法需先将rotation的范围限制在0-360°
            obstacle['rotation'] = obstacle['rotation'] % 360
            if 0 <= obstacle['rotation'] < 270:
                rot_yaw = (90 - obstacle['rotation'])
            elif 270 <= obstacle['rotation'] <= 360:
                rot_yaw = (450 - obstacle['rotation'])

            if (cal_yaw - rot_yaw * math.pi / 180.0) > 1e-5:
                print(f'angle error in {name} {key_id}', cal_yaw,
                      rot_yaw * math.pi / 180.0,
                      cal_yaw - rot_yaw * math.pi / 180.0)
            point_temp[name] = {'key_points': poly_region, 'angle': rot_yaw}

        vehicle_json[key_id] = point_temp

    vehicle_out_json = open(
        os.path.join(data_path, outer_scene, 'sync_data', scene_id,
                     'vehicle.json'), 'w')
    json.dump(vehicle_json, vehicle_out_json, indent=4)
    vehicle_out_json.close()
    # print(outer_scene + " annotation_" + scene_id +
    #       ".xml 已转换为 vehicle.json \n")


# ------------------------------------------------------03 生成AthenaDataset所需的jsons
# 生成timestamp.json
def generate_timestamp_json(level_01_data_path, outer_scene):
    timestamp_json = {}
    scene_num = -1
    if os.path.isdir(os.path.join(level_01_data_path, outer_scene)):
        scenes = sorted(
            os.listdir(
                os.path.join(level_01_data_path, outer_scene, "sync_data")))
        scenes = sorted([int(s) for s in scenes
                         if len(s) < 4])  #将scene：0，1，2文件夹筛选出来，排序后转为int
        for scene in scenes:
            if scene == ".DS_Store" or not os.path.isdir(
                    os.path.join(level_01_data_path, outer_scene, "sync_data",
                                 str(scene), 'cam_back')):
                continue
            scene_num += 1
            timestamp_json[scene_num] = {}
            with open(
                    os.path.join(level_01_data_path, outer_scene, "sync_data",
                                 str(scene), 'times.txt'), "r") as f:
                times = f.read()
            times = times.split("\n")
            samples = sorted(
                os.listdir(
                    os.path.join(level_01_data_path, outer_scene, 'sync_data',
                                 str(scene), 'cam_back')))
            int_sample_name = [k for k in samples if k[-1] == 'g']
            for sample in range(len(int_sample_name)):
                if sample == ".DS_Store":
                    continue
                timestamp_json[scene_num][int_sample_name[sample]] = {
                    "timestamp": times[sample * 5]
                }
    with open(os.path.join(level_01_data_path, outer_scene, 'timestamp.json'),
              "w") as json_file:
        json.dump(timestamp_json, json_file)
    return timestamp_json


# 生成AthenaDataset所需的json
def generate_athena_dataset_jsons(data_path, training_path, timestamps):
    # 0.生成scene的token、sample的token、instance的token和annotation的token
    scenes_token = {}
    sample_token = {}
    instances_token = {}
    annotation_token = {}
    scene_num = -1
    if not os.path.isdir(os.path.join(data_path, outer_scene, 'sync_data')):
        print('(generate_athena_dataset_jsons)Error:  \n' + data_path +
              outer_scene + '/sync_data/  does not exist !')
        return
    scenes = sorted(
        os.listdir(os.path.join(data_path, outer_scene, 'sync_data')))
    scenes = sorted([int(s) for s in scenes if len(s) < 3])
    for scene in scenes:  # 遍历原始文件夹目录，每个子文件夹的名字就是一个scene
        if os.path.isdir(
                os.path.join(
                    data_path, outer_scene, 'sync_data', str(scene),
                    'cam_back')) and scene != ".DS_Store":  # 为每个scene生成一个token
            scene_num += 1
            uid = uuid.uuid1()
            scenes_token[scene_num] = uid.hex
            samples = sorted(
                os.listdir(
                    os.path.join(data_path, outer_scene, 'sync_data',
                                 str(scene), 'cam_back')))
            sample_token[scene_num] = {}
            for sample in samples:  # 遍历每个scene的文件夹目录，每个文件就是一个sample
                # 为每个sample生成一个token
                if not os.path.isdir(
                        os.path.join(data_path, outer_scene, 'sync_data',
                                     str(scene), 'cam_back',
                                     sample)) and sample[-1] == 'g':
                    with open(
                            os.path.join(data_path, outer_scene, 'sync_data',
                                         str(scene), 'cam_back', sample),
                            'rb') as fp:
                        sample_get_md5 = fp.read()
                    file_md5 = hashlib.md5(sample_get_md5).hexdigest()
                    sample_token[scene_num][sample] = file_md5
            # 生成instance的token和annotation的token
            instances = {}
            scene_annotation = {}

            with open(
                    os.path.join(data_path, outer_scene, 'sync_data',
                                 str(scene), "vehicle.json"), 'r') as f:
                vehicles = json.load(f)

            # 遍历samples
            for sample in samples:
                if sample in vehicles:
                    sample_annotation = {}
                    for k in vehicles[sample].keys(
                    ):  # vehicle.json的第二层key是各个instance的名字，如car1
                        if k not in instances.keys(
                        ):  # 每个场景中出现的car1都是同一个instance，每个instance只生成一个token
                            uid = uuid.uuid1()
                            instances[k] = uid.hex
                        uid = uuid.uuid1()
                        sample_annotation[k] = uid.hex
                    scene_annotation[sample] = sample_annotation
            annotation_token[
                scene_num] = scene_annotation  # sample_annotation需要存储全部标注框的信息，有三层key，scene-sample-instance
            instances_token[
                scene_num] = instances  # instance_json只有两层key，scene-instance(scene下的所有instance(唯一))
    with open(os.path.join(training_path, 'scene_token.json'),
              "w") as json_file:
        json.dump(scenes_token, json_file)
    with open(os.path.join(training_path, 'sample_token.json'),
              "w") as json_file:
        json.dump(sample_token, json_file)
    with open(os.path.join(training_path, 'instances_token.json'),
              "w") as json_file:
        json.dump(instances_token, json_file)
    with open(os.path.join(training_path, 'annotation_token.json'),
              "w") as json_file:
        json.dump(annotation_token, json_file)

    # 1.生成scene.json
    scenes_json = []
    for key in scenes_token.keys():
        token = scenes_token[key]
        # TODO:根据地图信息定义log_token
        log_token = "7e25a2c8ea1f41c5b0da1e69ecfa71a2"
        nbr_samples = len(sample_token[key])
        # sample要按照时间排顺序，需要存储时间上的前后关系
        int_sample_name = {int(k[:6]): k for k in sample_token[key].keys()}
        sort_sample_name = sorted(int_sample_name.keys())
        name = "scene-" + str(key).zfill(4)
        description = "vehicles"
        scenes_json.append({
            "token":
            token,
            "log_token":
            log_token,
            "nbr_samples":
            nbr_samples,
            "first_sample_token":
            sample_token[key][int_sample_name[sort_sample_name[0]]],
            "last_sample_token":
            sample_token[key][int_sample_name[sort_sample_name[-1]]],
            "name":
            name,
            "description":
            description
        })
    with open(os.path.join(training_path, 'scene.json'), "w") as json_file:
        json.dump(scenes_json, json_file)
    print('1.生成scene.json done.\n')

    # 2.生成sample.json
    samples_json = []
    for scene in scenes_token.keys():
        int_sample_name = {int(k[:6]): k for k in sample_token[scene].keys()}
        sort_sample_name = sorted(int_sample_name.keys())

        for i in range(0, len(sort_sample_name)):
            token = sample_token[scene][int_sample_name[sort_sample_name[i]]]
            timestamp = timestamps[scene][int_sample_name[
                sort_sample_name[i]]]['timestamp']
            sample_prev = ""
            sample_next = ""
            scene_token = scenes_token[scene]
            if i > 0:
                sample_prev = sample_token[scene][int_sample_name[
                    sort_sample_name[i - 1]]]
            if i < len(sort_sample_name) - 1:
                sample_next = sample_token[scene][int_sample_name[
                    sort_sample_name[i + 1]]]
            samples_json.append({
                "token": token,
                "timestamp": timestamp,
                "prev": sample_prev,
                "next": sample_next,
                "scene_token": scene_token
            })
    with open(os.path.join(training_path, 'sample.json'), "w") as json_file:
        json.dump(samples_json, json_file)
    print('2.生成sample.json done.\n')

    # 3.生成instance.json
    instances_json = []
    # 需要先把category.json拷贝进目标文件夹
    with open(training_path + 'category.json', 'r') as cate_data:
        cates = json.load(cate_data)

    for scene_instance in instances_token.keys():
        for instance_name in instances_token[scene_instance].keys():
            token = instances_token[scene_instance][instance_name]

            category_token = ""
            class_n = re.split('([0-9]+)', instance_name)[0]
            for cate in cates:
                if cate['name'] == class_n:
                    category_token = cate['token']
                    break

                # 遍历这个scene的所有sample中这个instance_name的数量
            instance_annotation = []
            int_sample_name = {
                int(k[:6]): k
                for k in annotation_token[scene_instance].keys()
            }
            # 例sort_sample_name={0, 5}
            sort_sample_name = sorted(int_sample_name.keys())

            for i in range(0, len(sort_sample_name)):
                for anno_instance in annotation_token[scene_instance][
                        int_sample_name[sort_sample_name[i]]].keys():
                    if anno_instance == instance_name:
                        instance_annotation.append(
                            annotation_token[scene_instance][int_sample_name[
                                sort_sample_name[i]]][anno_instance])
                        # 每个sample中每次instance只会出现一次
                        break
            if category_token == '':
                print('Error: category_token==\'\'  ', instance_name)
                print(
                    class_n, "  in outer_scene: " + outer_scene + "    " +
                    str(scene_instance))
                continue
            instances_json.append({
                "token":
                token,
                "category_token":
                category_token,
                "nbr_annotations":
                len(instance_annotation),
                "first_annotation_token":
                instance_annotation[0],
                "last_annotation_token":
                instance_annotation[-1]
            })
    with open(os.path.join(training_path, 'instance.json'), "w") as json_file:
        json.dump(instances_json, json_file)
    print('3.生成instance.json done.\n')

    # 4.生成sample_annotation.json
    sample_annotations = []
    for scene_instance in instances_token.keys(
    ):  # 遍历一个scene中的所有instance，然后按顺序记录该instance的annotation
        for instance_name in instances_token[scene_instance].keys():
            instance_annotation = []

            int_sample_name = {
                int(k[:6]): k
                for k in annotation_token[scene_instance].keys()
            }
            sort_sample_name = sorted(int_sample_name.keys())

            sample_squence = []
            for i in range(0, len(sort_sample_name)):
                for anno_instance in annotation_token[scene_instance][
                        int_sample_name[sort_sample_name[i]]].keys():
                    if anno_instance == instance_name:
                        sample_squence.append(
                            int_sample_name[sort_sample_name[i]])
                        instance_annotation.append(
                            annotation_token[scene_instance][int_sample_name[
                                sort_sample_name[i]]][anno_instance])
                        break
            translation = [0, 0, 0]
            size = [0, 0, 0]
            rotation = [0, 0, 0, 0]
            visibility_token = 4
            attribute_tokens = "cb5118da1ab342aa947717dc53544259"
            anno_prev = ""
            anno_next = ""
            for i in range(0, len(instance_annotation)):
                token = instance_annotation[i]
                s_token = sample_token[scene_instance][sample_squence[i]]
                ins_token = instances_token[scene_instance][instance_name]
                if i > 0:
                    anno_prev = instance_annotation[i - 1]
                if i < len(instance_annotation) - 1:
                    anno_next = instance_annotation[i + 1]
                sample_annotations.append({
                    "token": token,
                    "sample_token": s_token,
                    "instance_token": ins_token,
                    "visibility_token": visibility_token,
                    "attribute_tokens": attribute_tokens,
                    "translation": translation,
                    "size": size,
                    "rotation": rotation,
                    "prev": anno_prev,
                    "next": anno_next,
                    "num_lidar_pts": 0,
                    "num_radar_pts": 0
                })
    with open(os.path.join(training_path, 'sample_annotation.json'),
              "w") as json_file:
        json.dump(sample_annotations, json_file)
    print('4.生成sample_annotation.json done.\n')

    # 5.生成sensor的token和sensor.json
    sensors_token = {}
    sensors_json = []
    # TODO:根据需要增加传感器
    sensors = [
        "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK",
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ]
    for sensor in sensors:
        uid = uuid.uuid1()
        code = uid.hex
        sensors_token[sensor] = code
        token = code
        channel = sensor
        modality = "camera"
        sensor_json = {
            "token": token,
            "channel": channel,
            "modality": modality
        }
        sensors_json.append(sensor_json)
    with open(os.path.join(training_path, 'sensor_token.json'),
              "w") as json_file:
        json.dump(sensors_token, json_file)
    with open(os.path.join(training_path, 'sensor.json'), "w") as json_file:
        json.dump(sensors_json, json_file)
    print('5.生成sensor.json done.\n')

    # 6.生成calibrated_sensor的token和calibrated_sensor.json
    calibrated_sensor_token = {}
    calibrated_sensor_json = []
    for scene in scenes_token.keys():
        calibrated_sensor_token[scene] = {}
        for sensor in sensors_token.keys():
            uid = uuid.uuid1()
            code = uid.hex
            calibrated_sensor_token[scene][sensor] = code
            token = code
            sensor_token = sensors_token[sensor]
            translation = []
            rotation = []
            camera_intrinsic = []
            calibrated_sensor = {
                'token': token,
                'sensor_token': sensor_token,
                'translation': translation,
                'rotation': rotation,
                'camera_intrinsic': camera_intrinsic
            }
            calibrated_sensor_json.append(calibrated_sensor)
    with open(os.path.join(training_path, 'calibrated_sensor_token.json'),
              "w") as json_file:
        json.dump(calibrated_sensor_token, json_file)
    with open(os.path.join(training_path, 'calibrated_sensor.json'),
              "w") as json_file:
        json.dump(calibrated_sensor_json, json_file)
    print('6.生成calibrated_sensor的token和calibrated_sensor.json done.\n')

    # 7.生成sample_data的token、sample_data.json和ego_pose.json
    sample_data_token = {}
    sample_data_json = []
    ego_pose_json = []
    camera = [
        'CAM_BACK', 'CAM_FRONT', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT'
    ]
    for scene in sample_token.keys():
        sample_data_token[scene] = {}
        for cam in camera:
            sample_data_token[scene][cam] = {}
            for sample in sample_token[scene].keys():
                uid = uuid.uuid1()
                sample_data_token[scene][cam][sample] = uid.hex

            int_sample_name = {
                int(k[:6]): k
                for k in sample_token[scene].keys()
            }
            sort_sample_name = sorted(int_sample_name.keys())

            for i in range(0, len(sort_sample_name)):
                token = sample_data_token[scene][cam][int_sample_name[
                    sort_sample_name[i]]]
                s_token = sample_token[scene][int_sample_name[
                    sort_sample_name[i]]]
                c_sensor_token = calibrated_sensor_token[scene][cam]
                timestamp = timestamps[scene][int_sample_name[
                    sort_sample_name[i]]]['timestamp']
                fileformat = 'jpg'
                is_key_frame = True
                height = 720
                width = 1280
                # filename的内容需要根据场景设置，已规范为'000000_shuiguohutoxiongchu_20220526_hongqi1_0.jpg'的格式
                filename = 'samples/' + cam + '/' + int_sample_name[
                    sort_sample_name[i]].split(
                        ".")[0] + "_" + outer_scene + "_" + str(scene) + ".jpg"

                sample_data_prev = ""
                sample_data_next = ""
                if i > 0:
                    sample_data_prev = sample_data_token[scene][cam][
                        int_sample_name[sort_sample_name[i - 1]]]
                if i < len(sort_sample_name) - 1:
                    sample_data_next = sample_data_token[scene][cam][
                        int_sample_name[sort_sample_name[i + 1]]]
                sample_data_json.append({
                    'token': token,
                    'sample_token': s_token,
                    'ego_token': token,
                    'calibrated_sensor_token': c_sensor_token,
                    'timestamp': timestamp,
                    'fileformat': fileformat,
                    'is_key_frame': is_key_frame,
                    'height': height,
                    'width': width,
                    'filename': filename,
                    'prev': sample_data_prev,
                    'next': sample_data_next
                })
                # ego_pose_json
                rotation = []
                translation = []
                ego_pose_json.append({
                    'token': token,
                    'timestamp': timestamp,
                    'rotation': rotation,
                    'translation': translation
                })
    with open(os.path.join(training_path, 'sample_data.json'),
              "w") as json_file:
        json.dump(sample_data_json, json_file)
    with open(os.path.join(training_path, 'ego_pose.json'), "w") as json_file:
        json.dump(ego_pose_json, json_file)
    print('7.生成sample_data的token、sample_data.json和ego_pose.json done.\n')

    # 8.生成hy_sample_annotation.json
    # 通过hy_sample_annotation[scene_token][sample_token][instance_token]获取到四个点的坐标
    hy_sample_annotation = {}
    scene_num = -1
    for scene in scenes:
        if os.path.isdir(
                os.path.join(data_path, outer_scene, 'sync_data', str(scene),
                             'cam_back')):
            with open(
                    os.path.join(data_path, outer_scene, 'sync_data',
                                 str(scene), "vehicle.json"), 'r') as f:
                vehicles = json.load(f)

            scene_num += 1
            scene_annotation = {}
            samples = sorted(
                os.listdir(
                    os.path.join(data_path, outer_scene, 'sync_data',
                                 str(scene), 'cam_back')))
            for sample in samples:
                if sample in vehicles:
                    sample_annotation = {}
                    for ins in vehicles[sample].keys():
                        sample_annotation[instances_token[scene_num][ins]] = {
                            "key_points":
                            uv_to_xy(vehicles[sample][ins]
                                     ["key_points"]),  #BEV像素坐标系(标注)转为车体坐标系
                            # "bbox": uv_to_xy(vehicles[sample][ins]["bbox"]),
                            # BEV像素坐标系(标注)转为车体坐标系
                            "angle":
                            vehicles[sample][ins]["angle"] * math.pi / 180.0
                        }  # 把yaw角转化为弧度
                    scene_annotation[sample_token[scene_num]
                                     [sample]] = sample_annotation
            if scene_annotation == []:
                print("Error : scene_annotation == []  in scene ", scene_num)
                continue
            hy_sample_annotation[scenes_token[scene_num]] = scene_annotation
    with open(os.path.join(training_path, "hy_sample_annotation.json"),
              "w") as json_file:
        json.dump(hy_sample_annotation, json_file)
    print('8.生成hy_sample_annotation.json done.\n')

    # 9.生成hy_ego_pose.json
    # [latitude longitude elevation utm_position.x utm_position.y utm_position.z attitude.x attitude.y attitude.z position_type]
    hy_ego_pose = {}
    scne_num = -1
    # for scene in tqdm.tqdm(scenes):
    for scene in scenes:
        if scene != '.DS_Store' and os.path.isdir(
                os.path.join(data_path, outer_scene, 'sync_data', str(scene),
                             'cam_back')):
            scne_num += 1
            hy_ego_pose[scenes_token[scne_num]] = {}
            poses = []
            with open(
                    os.path.join(data_path, outer_scene, 'sync_data',
                                 str(scene), 'pose.txt'), "r") as f:
                for line in f:
                    data = line.split(" ")
                    poses.append(data)
            samples = sorted(
                os.listdir(
                    os.path.join(data_path, outer_scene, 'sync_data',
                                 str(scene), 'cam_back')))
            int_sample_name = [k for k in samples if k[-1] == 'g']
            for sample in range(len(int_sample_name)):
                if sample == ".DS_Store":
                    continue
                no_ = sample * 5
                utm_x, utm_y, utm_z = float(poses[no_][3]), float(
                    poses[no_][4]), float(poses[no_][5])
                box = lanelet2.core.BoundingBox2d(
                    lanelet2.core.BasicPoint2d(utm_x - 150, utm_y - 150),
                    lanelet2.core.BasicPoint2d(utm_x + 150, utm_y + 150))
                osmname = ""
                for osm_name, lanelet_layer in lanelet_layers.items():
                    lanelets_inRegion = lanelet_layer.search(box)
                    for elem in lanelets_inRegion:
                        # if lanelet2.geometry.distance(elem,lanelet2.core.BasicPoint3d(utm_x, utm_y,utm_z)) == 0:
                        if lanelet2.geometry.distance(
                                elem, lanelet2.core.BasicPoint2d(utm_x,
                                                                 utm_y)) == 0:
                            # cur_lane = elem
                            osmname = osm_name
                            break
                    if osmname != "":
                        break
                hy_ego_pose[scenes_token[scne_num]][timestamps[scne_num][
                    int_sample_name[sample]]['timestamp']] = {
                        "latitude": poses[no_][0],
                        "longitude": poses[no_][1],
                        "elevation": poses[no_][2],
                        "utm_position.x": poses[no_][3],
                        "utm_position.y": poses[no_][4],
                        "utm_position.z": poses[no_][5],
                        "attitude.x": poses[no_][6],
                        "attitude.y": poses[no_][7],
                        "attitude.z": poses[no_][8],
                        "position_type": poses[no_][9],
                        "osmname": osmname,
                    }
    with open(os.path.join(training_path, 'hy_ego_pose.json'),
              "w") as json_file:
        json.dump(hy_ego_pose, json_file)
    print('9.生成hy_ego_pose.json done.\n')


if __name__ == '__main__':
    # 定义原始文件夹根目录
    # level_01_data_path = '/media/hy/889b0bd5-777a-4ee4-bef4-171d2359de14/dataset/HY_dataset/level01/'
    level_01_data_path = '/mnt/data_1/Dataset/dataset_temp/map_c/'
    outer_scenes = os.listdir(level_01_data_path)
    # 生成v1.0-trainval/category.json
    category_json = []
    for box_type in box_types:
        uid = uuid.uuid1()
        token = uid.hex  #去掉uid中的'-'
        type_json = {"token": token, "name": box_type, "description": box_type}
        category_json.append(type_json)
    with open(
            '/mnt/data_1/Dataset/dataset_temp/map_c/dftc2ysg2wd_20220607_hq1/trainval/v1.0-trainval/category.json',
            "w") as json_file:
        json.dump(category_json, json_file)

    for outer_scene in tqdm(outer_scenes):
        if os.path.isdir(os.path.join(level_01_data_path, outer_scene)):
            level_02_data_path = os.path.join(level_01_data_path, outer_scene)
            os.makedirs(level_02_data_path, exist_ok=True)
            # 在level_02目录下生成所需文件夹
            # os.makedirs(level_02_data_path + '/maps', exist_ok=True)
            # #后面会直接将nuscenes的maps文件夹copy过来
            os.makedirs(os.path.join(level_02_data_path, 'sweeps'),
                        exist_ok=True)
            os.makedirs(os.path.join(level_02_data_path, 'samples'),
                        exist_ok=True)
            os.makedirs(os.path.join(level_02_data_path, 'v1.0-trainval'),
                        exist_ok=True)

            # 0.将图像数据加上mask，重命名后copy到level_02的对应路径下
            # img_copy(level_01_data_path, outer_scene, level_02_data_path + 'samples')

            # 1.根据annotaion.xml得到vehicle.json
            # xmls_path = os.path.join(level_01_data_path, outer_scene,
            #                          'xmls_128')
            # if os.path.isfile(
            #         os.path.join(level_01_data_path, outer_scene,
            #                      'annotations.xml')):
            #     os.makedirs(xmls_path, exist_ok=True)
            #     split_xml(
            #         os.path.join(level_01_data_path, outer_scene,
            #                      'annotations.xml'), xmls_path)
            # xmls = sorted(os.listdir(xmls_path))
            # for xml in tqdm(xmls, desc='xml to vehicle.json', leave=True):
            #     xml_path = os.path.join(xmls_path, xml)
            #     images = parse_xml(xml_path)
            #     images = dict(sorted(images.items(), key=lambda x: x[0]))
            #     os.makedirs(os.path.join(level_01_data_path, outer_scene,
            #                              'annos'),
            #                 exist_ok=True)
            #     out_json = open(
            #         os.path.join(
            #             level_01_data_path, outer_scene, 'annos',
            #             'anno_' + xml.split(".")[0].split("_")[1] + '.json'),
            #         'w')
            #     json.dump(images, out_json, indent=4)
            #     out_json.close()
            #     generate_vehicle_json(level_01_data_path, images, outer_scene,
            #                           xml.split("_")[-1].split(".")[0])

            # 2.生成timestamp.json
            timestamps = generate_timestamp_json(level_01_data_path,
                                                 outer_scene)

            # TODO:log.json和map.json也可以根据需要生成
            # 3.将nuscenes数据集中的部分json文件copy到训练目录
            file = [
                "log.json", "map.json", "attribute.json", "visibility.json",
                "category.json"
            ]
            for f in file:
                shutil.copy(
                    "/mnt/data_1/Dataset/dataset_temp/to_nuscenes/demo_level02/v1.0-trainval/"
                    + f, level_02_data_path + '/v1.0-trainval/' + f)

            # 将nuscenes数据集中map部分文件copy对应的map文件夹下
            if os.path.isdir(os.path.join(level_02_data_path, 'maps')):
                shutil.rmtree(os.path.join(level_02_data_path, 'maps'))
            shutil.copytree(
                "/mnt/data_1/Dataset/dataset_temp/to_nuscenes/demo_level02/maps",
                os.path.join(level_02_data_path, 'maps'))

            # 3.根据公司数据生成NUSCENES格式剩余的json文件
            generate_athena_dataset_jsons(
                level_01_data_path, level_02_data_path + '/v1.0-trainval/',
                timestamps)
            print(outer_scene + " generated ! \n")

    print("all scenes generated !")
