'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-04 16:13:19
LastEditors: Leidi
LastEditTime: 2022-09-14 13:53:03
'''
import json
import math
import os

import cv2
import numpy as np
from utils.utils import dist


class BOX:
    """真实框类"""

    def __init__(
        self,
        box_clss: str = '',
        box_xywh: list = None,
        box_xtlytlxbrybr: list = None,
        box_rotation: float = 0,
        box_rotated_rect_points: list = None,
        box_head_point: list = None,
        box_head_orientation: float = None,
        box_size_erro: float = 0,
        box_color: str = '',
        box_tool: str = '',
        box_difficult: int = 0,
        box_distance: float = 0,
        box_occlusion: float = 0,
    ) -> None:
        """真实框类

        Args:
            box_clss (str, optional): 类别. Defaults to ''.
            box_xywh (list, optional): bbox左上点和宽高. Defaults to None.
            box_xtlytlxbrybr (list, optional): 真实框左上右下对角点列表. Defaults to None.
            box_rotation (float, optional): 真实框旋转角. Defaults to 0.
            box_rotated_rect_points (list, optional): 真实框矩形角点. Defaults to None.
            box_head_point (list, optional): 头指向标志点. Defaults to None.
            box_head_orientation (float, optional): 头指向角度. Defaults to None.
            box_size_erro (float, optional): bbox大小错误标志位. Defaults to 0.
            box_color (str, optional): 真实框目标颜色. Defaults to ''.
            box_tool (str, optional): bbox工具. Defaults to ''.
            box_difficult (int, optional): 检测困难度. Defaults to 0.
            box_distance (float, optional): 真实框中心点距离. Defaults to 0.
            box_occlusion (float, optional): 真实框遮挡率. Defaults to 0.
        """

        self.box_clss = box_clss
        if box_xywh == None:
            self.box_xywh = []
        else:
            self.box_xywh = box_xywh
        self.box_rotation = box_rotation
        self.box_size_erro = box_size_erro
        if box_xtlytlxbrybr == None:
            self.box_xtlytlxbrybr = []
            self.box_rotated_rect_points = []
        else:
            if box_rotated_rect_points == None:
                self.box_xtlytlxbrybr = box_xtlytlxbrybr
                self.box_rotated_rect_points, self.box_size_erro = self.rotated_rect_point(
                    self.box_xtlytlxbrybr[0], self.box_xtlytlxbrybr[1],
                    self.box_xtlytlxbrybr[2], self.box_xtlytlxbrybr[3],
                    self.box_rotation)
            else:
                self.box_xtlytlxbrybr = box_xtlytlxbrybr
                self.box_rotated_rect_points = box_rotated_rect_points
        if box_head_point == None:
            self.box_head_point = []
            self.box_head_orientation = 0
        else:
            if box_head_orientation == None:
                self.box_head_point = box_head_point
                self.box_head_orientation = self.get_head_orientation()
            else:
                self.box_head_point = box_head_point
                self.box_head_orientation = box_head_orientation
        self.box_color = box_color
        self.box_tool = box_tool
        self.box_difficult = box_difficult
        self.box_distance = box_distance
        self.box_occlusion = box_occlusion
        if len(self.box_xywh):
            self.box_exist_flag = True
        else:
            self.box_exist_flag = False

    def box_get_area(self) -> float:
        """[获取box面积]

        Returns:
            float: [box面积]
        """

        return self.box_xywh[2] * self.box_xywh[3]

    def true_box_to_true_segmentation(self) -> list:
        """[获取由真实框转换来的分割包围框]

        Returns:
            list: [真实框转换来的分割包围框]
        """

        point_0 = [self.box_xywh[0], self.box_xywh[1]]
        point_1 = [self.box_xywh[0] + self.box_xywh[2], self.box_xywh[1]]
        point_2 = [
            self.box_xywh[0] + self.box_xywh[2],
            self.box_xywh[1] + self.box_xywh[3]
        ]
        point_3 = [self.box_xywh[0], self.box_xywh[1] + self.box_xywh[3]]

        return [point_0, point_1, point_2, point_3]

    def rotated_rect_point(self,
                           xtl: float,
                           ytl: float,
                           xbr: float,
                           ybr: float,
                           angle: float,
                           min_width: float = 0,
                           max_width: float = 0,
                           min_height: float = 0,
                           max_heigh: float = 0) -> list:
        """由box左上右下点及旋转角求box四个角点坐标。

        Args:
            xtl (float): box左上点x坐标
            ytl (float): box左上点y坐标
            xbr (float): box右下点x坐标
            ybr (float): box右下点y坐标
            angle (float): 旋转角度
            min_width (float, optional): box最小宽度. Defaults to 0.
            max_width (float, optional): box最大宽度. Defaults to 0.
            min_height (float, optional): box最小高度. Defaults to 0.
            max_heigh (float, optional): box最大高度. Defaults to 0.

        Returns:
            list: point_array.tolist(), size_err
        """

        width, height = abs(xtl - xbr), abs(ytl - ybr)
        if width < height:
            width, height = height, width
        size_err = 0
        if width > max_width or height > max_heigh or width < min_width or height < min_height:
            size_err = 1
        center_x, center_y = (xtl + xbr) / 2, (ytl + ybr) / 2
        angle = (-angle) * math.pi / 180  # 矩形旋转弧度
        xo = np.cos(angle)
        yo = np.sin(angle)
        y1 = center_y + height / 2 * yo
        x1 = center_x - height / 2 * xo
        y2 = center_y - height / 2 * yo
        x2 = center_x + height / 2 * xo

        point_array = np.array([
            [x1 - width / 2 * yo, y1 - width / 2 * xo],
            [x2 - width / 2 * yo, y2 - width / 2 * xo],
            [x2 + width / 2 * yo, y2 + width / 2 * xo],
            [x1 + width / 2 * yo, y1 + width / 2 * xo],
        ]).astype(np.int32)

        point_list = []
        for point in point_array.tolist():
            point_list.append(point)

        return point_list, size_err

    def rotated_rect_point_no_orin_point(self, xtl, ytl, xbr, ybr, angle):
        """不打头指向点, 由box左上右下点及旋转角求box四个角点坐标

        Args:
            xtl (float): box左上点x坐标
            ytl (float): box左上点y坐标
            xbr (float): box右下点x坐标
            ybr (float): box右下点y坐标
            angle (_type_): 旋转角度

        Returns:
            _type_: _description_
        """
        
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

    def get_head_orientation(self) -> float:
        """获取目标标注包围框头指向（标注图片水平向右为正方向）

        Returns:
            float: 目标标注包围框头指向角度（标注图片水平向右为正方向）
        """
        head_point_np = np.array(self.box_head_point)
        head_point_to_corner_distance_dict = {}
        for n, corner_point in enumerate(self.box_rotated_rect_points):
            corner_point_np = np.array(corner_point)
            distance = dist(head_point_np, corner_point_np)
            head_point_to_corner_distance_dict.update({n: distance})
        head_point_to_corner_distance_list = sorted(
            head_point_to_corner_distance_dict.items(), key=lambda d: d[1])
        head_corner_point_0_xy = self.box_rotated_rect_points[
            head_point_to_corner_distance_list[0][0]]
        head_corner_point_1_xy = self.box_rotated_rect_points[
            head_point_to_corner_distance_list[1][0]]
        head_corner_point_center_xy = [
            (head_corner_point_0_xy[0] + head_corner_point_1_xy[0]) / 2,
            (head_corner_point_0_xy[1] + head_corner_point_1_xy[1]) / 2
        ]
        box_center_xy = [
            (self.box_xtlytlxbrybr[0] + self.box_xtlytlxbrybr[2]) / 2,
            (self.box_xtlytlxbrybr[1] + self.box_xtlytlxbrybr[3]) / 2
        ]
        head_orientation = math.atan2(
            head_corner_point_center_xy[1] - box_center_xy[1],
            head_corner_point_center_xy[0] - box_center_xy[0]) / math.pi * 180

        return head_orientation


class SEGMENTATION:
    """真分割类"""

    def __init__(
        self,
        segmentation_clss: str = '',
        segmentation: list = None,
        segmentation_area: int = None,
        segmentation_iscrowd: int = 0,
    ) -> None:
        """[真分割]

        Args:
            segmentation_clss (str): [类别]
            segmentation (list): [分割区域列表]
            area (float, optional): [像素区域大小]. Defaults to 0.
            iscrowd (int, optional): [是否使用coco2017中的iscrowd格式]. Defaults to 0.
        """

        self.segmentation_clss = segmentation_clss
        if segmentation == None:
            self.segmentation = []
        else:
            self.segmentation = segmentation
        if segmentation_area == None:
            if len(self.segmentation):
                self.segmentation_area = int(
                    cv2.contourArea(np.array(self.segmentation)))
            else:
                self.segmentation_area = 0
        else:
            self.segmentation_area = segmentation_area
        self.segmentation_iscrowd = int(segmentation_iscrowd)
        if len(self.segmentation):
            self.segmentation_exist_flag = True
        else:
            self.segmentation_exist_flag = False

    def segmentation_get_bbox_area(self) -> int:
        """[获取语义分割外包围框面积]

        Returns:
            int: [语义分割外包围框面积]
        """

        segmentation = np.asarray(self.segmentation)
        min_x = int(np.min(segmentation[:, 0]))
        min_y = int(np.min(segmentation[:, 1]))
        max_x = int(np.max(segmentation[:, 0]))
        max_y = int(np.max(segmentation[:, 1]))
        box_xywh = [min_x, min_y, max_x - min_x, max_y - min_y]

        return box_xywh[2] * box_xywh[3]

    def true_segmentation_to_true_box(self) -> list:
        """[将分割按最外围矩形框转换为bbox]

        Returns:
            list: [转换后真实框左上点坐标、宽、高]
        """

        segmentation = np.asarray(self.segmentation)
        min_x = np.min(segmentation[:, 0])
        min_y = np.min(segmentation[:, 1])
        max_x = np.max(segmentation[:, 0])
        max_y = np.max(segmentation[:, 1])
        width = max_x - min_x
        hight = max_y - min_y
        bbox = [int(min_x), int(min_y), int(width), int(hight)]

        return bbox


class KEYPOINTS:
    """真实关键点类"""

    def __init__(
        self,
        keypoints_clss: str = '',
        keypoints_num: int = 0,
        keypoints: list = None,
    ) -> None:
        """[真实关键点类]

        Args:
            clss (str): [类别]
            num_keypoints (int): [关键点数量]
            keypoints (list): [关键点坐标列表]
        """

        self.keypoints_clss = keypoints_clss
        self.keypoints_num = keypoints_num
        if keypoints == None:
            self.keypoints = []
        else:
            self.keypoints = keypoints
        if keypoints is not None and len(keypoints):
            self.keypoints_exist_flag = True
        else:
            self.keypoints_exist_flag = False


class OBJECT(BOX, SEGMENTATION, KEYPOINTS):
    """标注物体类"""

    def __init__(
        self,
        object_id: int,
        object_clss: str,
        box_clss: str = '',
        segmentation_clss: str = '',
        keypoints_clss: str = '',
        box_xywh: list = None,
        box_xtlytlxbrybr: list = None,
        box_rotation: float = 0,
        box_rotated_rect_points: list = None,
        box_head_point: list = None,
        box_head_orientation: float = None,
        segmentation: list = None,
        keypoints_num: int = 0,
        keypoints: list = None,
        need_convert: str = None,
        box_color: str = '',
        box_tool: str = '',
        box_difficult: int = 0,
        box_distance: float = 0,
        box_occlusion: float = 0,
        segmentation_area: int = None,
        segmentation_iscrowd: int = 0,
    ) -> None:
        """[标注物体类初始化]

        Args:
            object_id (int): [标注目标id]
            object_clss (str): [标注目标类别]
            box_clss (str): [真实框类别]
            segmentation_clss (str): [分割区域类别]
            keypoints_clss (str): [关键点类别]
            box_xywh (list): [真实框x, y, width, height列表]
            box_xtlytlxbrybr (list): [真实框左上右下对角点列表]
            box_rotation (float): [真实框旋转角]
            segmentation (list): [分割多边形点列表]
            keypoints_num (int): [关键点个数]
            keypoints (list): [关键点坐标]
            need_convert (str): [标注目标任务转换形式]
            box_color (str, optional): [真实框颜色]. Defaults to ''.
            box_tool (str, optional): [真实框标注工具]. Defaults to ''.
            box_difficult (int, optional): [真实框困难程度]. Defaults to 0.
            box_distance (float, optional): [真实框距离]. Defaults to 0.
            box_occlusion (float, optional): [真实框遮挡比例]. Defaults to 0.
            segmentation_area (int, optional): [分割区域像素大小]. Defaults to None.
            segmentation_iscrowd (int, optional): [是否使用coco2017中的iscrowd格式]. Defaults to 0.
        """

        BOX.__init__(self,
                     box_clss,
                     box_xywh,
                     box_xtlytlxbrybr=box_xtlytlxbrybr,
                     box_rotation=box_rotation,
                     box_rotated_rect_points=box_rotated_rect_points,
                     box_head_point=box_head_point,
                     box_head_orientation=box_head_orientation,
                     box_color=box_color,
                     box_tool=box_tool,
                     box_difficult=box_difficult,
                     box_distance=box_distance,
                     box_occlusion=box_occlusion)
        SEGMENTATION.__init__(self,
                              segmentation_clss,
                              segmentation,
                              segmentation_area=segmentation_area,
                              segmentation_iscrowd=segmentation_iscrowd)
        KEYPOINTS.__init__(self, keypoints_clss, keypoints_num, keypoints)
        self.object_id = object_id
        self.object_clss = object_clss
        self.object_convert_flag = ''
        if need_convert == None:
            self.need_convert = None
        else:
            self.need_convert = need_convert
        if 0 == len(self.box_xywh)\
                and 0 != len(self.segmentation)  \
        and self.need_convert == 'segmentation_to_box':
            self.box_xywh = self.true_segmentation_to_true_box()
            self.box_clss = self.segmentation_clss
            self.box_exist_flag = True
            self.object_convert_flag = 'segmentation_to_box'
        if 0 == len(self.segmentation)\
                and 0 != len(self.box_xywh) \
            and self.need_convert == 'box_to_segmentation':
            self.segmentation = self.true_box_to_true_segmentation()
            self.segmentation_clss = self.box_clss
            self.segmentation_exist_flag = True
            self.object_convert_flag = 'box_to_segmentation'

    def delete_box_information(self) -> None:
        """[清除object中box信息]
        """

        self.box_clss = ''
        self.box_xywh = []
        self.box_color = ''
        self.box_tool = ''
        self.box_difficult = ''
        self.box_distance = ''
        self.box_occlusion = ''
        self.box_exist_flag = False

        return

    def delete_segmentation_information(self) -> None:
        """[清除object中segmentation信息]
        """

        self.segmentation_clss = ''
        self.segmentation = []
        self.segmentation_area = ''
        self.segmentation_iscrowd = ''
        self.segmentation_exist_flag = False

        return

    def delete_keypoints_information(self) -> None:
        """[清除object中keypoint信息]
        """

        self.keypoints_clss = ''
        self.keypoints_num = ''
        self.keypoints = []
        self.keypoints_exist_flag = False

        return


class EGO_POSE:
    def __init__(self) -> None:
        pass

class IMAGE:
    """图片类"""

    def __init__(
        self,
        image_name_in: str = '',
        image_name_new_in: str = '',
        image_path_in: str = '',
        height_in: int = 0,
        width_in: int = 0,
        channels_in: int = 0,
        object_list_in: list = None,
    ) -> None:
        """[图片类]

        Args:
            image_name_in (str): [图片名称]
            image_name_new_in (str): [图片新名称]
            image_path_in (str): [图片路径]
            height_in (int): [图片高]
            width_in (int): [图片宽]
            channels_in (int): [图片通道数]
            object_list_in (list): [标注目标列表]
        """

        self.image_name = image_name_in  # 图片名称
        self.image_name_new = image_name_new_in  # 修改后图片名称
        self.file_name = os.path.splitext(self.image_name)[0]
        self.file_name_new = os.path.splitext(self.image_name_new)[0]
        self.image_path = image_path_in  # 图片地址
        self.height = height_in  # 图片高
        self.width = width_in  # 图片宽
        self.channels = channels_in  # 图片通道数
        if object_list_in == None:
            self.object_list = []
        else:
            self.object_list = object_list_in
        if len(self.object_list):
            self.object_exist_flag = True
        else:
            self.object_exist_flag = False

    def object_class_modify_and_pixel_limit(self,
                                            dataset_instance: object) -> None:
        """清理无目标标注

        Args:
            dataset_instance (object): _description_
        """

        self.object_class_modify(dataset_instance)
        self.object_list_check_empty()
        self.object_pixel_limit(dataset_instance)
        self.object_list_check_empty()

        return

    def object_class_modify(self, dataset_instance: object) -> None:
        """[修改真实框类别]

        Args:
            dataset_instance (object): [数据集实例]
        """
        # 需补充3个类型均存在情况
        if dataset_instance.task_dict['Detection'] is None \
                and dataset_instance.task_dict['Semantic_segmentation'] is None \
                and dataset_instance.task_dict['Keypoints'] is None \
                and dataset_instance.task_dict['Instance_segmentation'] is not None \
                and dataset_instance.task_dict['Instance_segmentation']['Source_dataset_class'] is not None:
            for one_object in self.object_list:
                if one_object.object_clss not in dataset_instance.task_dict[
                        'Instance_segmentation']['Source_dataset_class']:
                    one_object.delete_box_information()
                    one_object.delete_segmentation_information()
                    one_object.delete_keypoints_information()
            if dataset_instance.task_dict['Instance_segmentation'][
                    'Modify_class_dict'] is not None:
                for one_object in self.object_list:
                    # 遍历融合类别文件字典，完成label中的类别修改，
                    # 若此bbox类别属于混合标签类别列表，则返回该标签在混合类别列表的索引值，修改类别。
                    not_in_modify_class_dict_box = True
                    not_in_modify_class_dict_segmentation = True
                    for key, value in dataset_instance.task_dict[
                            'Instance_segmentation'][
                                'Modify_class_dict'].items():
                        if one_object.box_clss in set(value):
                            one_object.box_clss = key
                            one_object.object_clss = one_object.box_clss
                            not_in_modify_class_dict_box = False
                        if one_object.segmentation_clss in set(value):
                            one_object.segmentation_clss = key
                            one_object.object_clss = one_object.segmentation_clss
                            not_in_modify_class_dict_segmentation = False
                    if not_in_modify_class_dict_box and not_in_modify_class_dict_segmentation:
                        one_object.delete_box_information()
                        one_object.delete_segmentation_information()
        else:
            for task, task_class_dict in dataset_instance.task_dict.items():
                if task_class_dict is None:
                    for one_object in self.object_list:
                        if task == 'Detection':
                            one_object.delete_box_information()
                        elif task == 'Semantic_segmentation':
                            one_object.delete_segmentation_information()
                        elif task == 'Keypoints':
                            one_object.delete_keypoints_information()
                else:
                    if task_class_dict['Source_dataset_class'] is not None:
                        for one_object in self.object_list:
                            if one_object.object_clss not in task_class_dict[
                                    'Source_dataset_class']:
                                one_object.delete_box_information()
                                one_object.delete_segmentation_information()
                                one_object.delete_keypoints_information()
                    if task_class_dict['Modify_class_dict'] is not None:
                        for one_object in self.object_list:
                            # 遍历融合类别文件字典，完成label中的类别修改，
                            # 若此bbox类别属于混合标签类别列表，则返回该标签在混合类别列表的索引值，修改类别。
                            if task == 'Detection':
                                not_in_modify_class_dict = True
                                for key, value in task_class_dict[
                                        'Modify_class_dict'].items():
                                    if one_object.box_clss in set(value):
                                        one_object.box_clss = key
                                        one_object.object_clss = one_object.box_clss
                                        not_in_modify_class_dict = False
                                if not_in_modify_class_dict:
                                    one_object.delete_box_information()
                            elif task == 'Semantic_segmentation':
                                not_in_modify_class_dict = True
                                for key, value in task_class_dict[
                                        'Modify_class_dict'].items():
                                    if one_object.segmentation_clss in set(
                                            value):
                                        one_object.segmentation_clss = key
                                        one_object.object_clss = one_object.segmentation_clss
                                        not_in_modify_class_dict = False
                                if not_in_modify_class_dict:
                                    one_object.delete_segmentation_information(
                                    )
                            elif task == 'Instance_segmentation':
                                not_in_modify_class_dict_box = True
                                not_in_modify_class_dict_segmentation = True
                                for key, value in task_class_dict[
                                        'Modify_class_dict'].items():
                                    if one_object.box_clss in set(value):
                                        one_object.box_clss = key
                                        one_object.object_clss = one_object.box_clss
                                        not_in_modify_class_dict_box = False
                                    if one_object.segmentation_clss in set(
                                            value):
                                        one_object.segmentation_clss = key
                                        one_object.object_clss = one_object.segmentation_clss
                                        not_in_modify_class_dict_segmentation = False
                                if not_in_modify_class_dict_box and not_in_modify_class_dict_segmentation:
                                    one_object.delete_box_information()
                                    one_object.delete_segmentation_information(
                                    )
                            elif task == 'Keypoint':
                                not_in_modify_class_dict = True
                                for key, value in task_class_dict[
                                        'Modify_class_dict'].items():
                                    if one_object.keypoints_class in set(
                                            value):
                                        one_object.keypoints_class = key
                                        not_in_modify_class_dict = False
                                if not_in_modify_class_dict:
                                    one_object.delete_keypoints_information()

        return

    def object_pixel_limit(self, dataset_instance: object) -> None:
        """[对标注目标进行像素大小筛选]

        Args:
            dataset_instance (object): [数据集实例]
        """

        for task, task_class_dict in dataset_instance.task_dict.items():
            if task_class_dict is None:
                continue
            if task_class_dict['Target_object_pixel_limit_dict'] is not None:
                for n, one_object in enumerate(self.object_list):
                    if task == 'Detection' or task == 'Instance_segmentation' or task == 'Keypoint':
                        if one_object.box_clss not in task_class_dict[
                                'Target_object_pixel_limit_dict']:
                            one_object.delete_box_information()
                            continue
                        pixel = one_object.box_get_area()
                        if pixel < task_class_dict['Target_object_pixel_limit_dict'][one_object.box_clss][0] or \
                                pixel > task_class_dict['Target_object_pixel_limit_dict'][one_object.box_clss][1]:
                            one_object.delete_box_information()
                    elif task == 'Semantic_segmentation':
                        if one_object.segmentation_clss not in task_class_dict[
                                'Target_object_pixel_limit_dict']:
                            one_object.delete_segmentation_information()
                            continue
                        pixel = one_object.segmentation_get_bbox_area()
                        if pixel < task_class_dict['Target_object_pixel_limit_dict'][one_object.segmentation_clss][0] or \
                                pixel > task_class_dict['Target_object_pixel_limit_dict'][one_object.segmentation_clss][1]:
                            one_object.delete_segmentation_information()

        return

    def object_list_check_empty(self) -> None:
        """清理object_list中无标注目标
        """

        temp_object_list = []
        for n in self.object_list:
            if n.box_exist_flag \
                or n.segmentation_exist_flag \
                    or n.keypoints_exist_flag:
                temp_object_list.append(n)
        self.object_list = temp_object_list

        return

    def output_temp_annotation(self, temp_annotation_output_path):
        """[输出temp dataset annotation]

        Args:
            annotation_output_path (str): [temp dataset annotation输出路径]

        Returns:
            bool: [输出是否成功]
        """

        if self == None:
            return False
        annotation = {
            'name': self.file_name_new,
            'base_information': {
                'width': self.width,
                'height': self.height,
                'channels': self.channels
            },
            'frames': [{
                'timestamp': 10000,
                'objects': []
            }],
            'attributes': {
                'weather': 'undefined',
                'scene': 'city street',
                'timeofday': 'daytime'
            }
        }
        id = 1
        object_count = 0
        for object in self.object_list:
            # 真实框
            if object.box_exist_flag == False \
                and object.segmentation_exist_flag == False \
                    and object.keypoints_exist_flag == False:
                continue
            object_dict = {
                'id': id,
                'object_clss': object.object_clss,
                'box_clss': object.box_clss,
                'box_color': object.box_color,
                'box_difficult': object.box_difficult,
                'box_distance': object.box_distance,
                'box_occlusion': object.box_occlusion,
                'box_tool': object.box_tool,
                'box_xywh': object.box_xywh,
                'box_xtlytlxbrybr': object.box_xtlytlxbrybr,
                'box_rotation': object.box_rotation,
                'box_rotated_rect_points': object.box_rotated_rect_points,
                'box_head_point': object.box_head_point,
                'box_head_orientation': object.box_head_orientation,
                'box_size_erro': object.box_size_erro,
                'box_exist_flag': object.box_exist_flag,
                'keypoints_clss': object.keypoints_clss,
                'keypoints_num': object.keypoints_num,
                'keypoints': object.keypoints,
                'keypoints_exist_flag': object.keypoints_exist_flag,
                'segmentation_clss': object.segmentation_clss,
                'segmentation': object.segmentation,
                'segmentation_area': object.segmentation_area,
                'segmentation_iscrowd': object.segmentation_iscrowd,
                'segmentation_exist_flag': object.segmentation_exist_flag,
            }
            annotation['frames'][0]['objects'].append(object_dict)
            id += 1
            object_count += 1
        if 0 == object_count:
            return True
        # 输出json文件
        json.dump(annotation, open(temp_annotation_output_path, 'w'))

        return True
