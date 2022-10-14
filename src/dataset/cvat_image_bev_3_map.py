'''
Description: 数据集图片由6相机图像及激光点云bev图分开组成
Version:
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-10-14 11:32:45
'''
import multiprocessing
import shutil
import xml.etree.ElementTree as ET

import cv2
import lanelet2
import open3d as o3d
import utm
from base.dataset_base import Dataset_Base
from base.image_base import *
from pyproj import Proj
from utils import image_form_transform
from utils.utils import *

import dataset


class CVAT_IMAGE_BEV_3_MAP(Dataset_Base):
    """CVAT平台使用拆分camera和点云bev标注方法

    Args:
        Dataset_Base (_type_): 数据集基础类
    """

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_annotation_image_folder = check_output_path(
            os.path.join(opt['Dataset_output_folder'],
                         'source_dataset_annotation_images'))
        self.source_dataset_map_folder = check_output_path(
            os.path.join(opt['Dataset_output_folder'], 'source_dataset_maps'))
        self.source_dataset_pose_folder = check_output_path(
            os.path.join(opt['Dataset_output_folder'], 'source_dataset_poses'))
        self.source_dataset_time_folder = check_output_path(
            os.path.join(opt['Dataset_output_folder'], 'source_dataset_times'))
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_image_form = 'jpg'
        self.source_dataset_annotation_form = 'xml'
        self.source_dataset_map_form = 'osm'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count(
        )
        self.source_dataset_map_count = self.get_source_dataset_map_count()
        self.image_ego_pose_dict = self.get_image_ego_pose()
        self.image_time_stamp_dict = self.get_image_time_stamp()
        self.lat_lon_origin = opt['Lat_lon_origin_point']

        self.origin_utm = {}
        self.utm_offset = int(
            math.sqrt(
                math.pow((self.label_range[0] + self.label_range[1]), 2) +
                math.pow((self.label_range[2] + self.label_range[3]), 2)))
        # self.camera_name_list = [
        #     'cam_back', 'cam_front_center', 'cam_left_back', 'cam_left_front',
        #     'cam_right_back', 'cam_right_front'
        # ]
        if self.get_dense_pcd_map_bev_image:
            self.dense_pcd_map_bev_image_folder = check_output_path(
                os.path.join(opt['Dataset_output_folder'],
                             'dense_pcd_map_bev_images'))
            dense_pcd_map_bev_image_path = os.path.join(
                self.dense_pcd_map_bev_image_folder,
                'dense_pcd_map_bev_image_location.json')
            if os.path.exists(dense_pcd_map_bev_image_path):
                with open(dense_pcd_map_bev_image_path, 'r',
                          encoding='utf8') as fp:
                    self.dense_pcd_map_location_dict_list = json.load(fp)
                self.dense_pcd_map_location_total_dict = {}
                for dense_map_dict in self.dense_pcd_map_location_dict_list:
                    self.dense_pcd_map_location_total_dict.update(
                        {dense_map_dict['name']: dense_map_dict})

    def get_utm_origin(self) -> dict:
        """计算utm起始点

        Returns:
            dict: utm起始点字典
        """

        origin_x, origin_y, zone_number, zone_letter = utm.from_latlon(
            self.lat_lon_origin[self.lat_lon_origin_city]['lat'],
            self.lat_lon_origin[self.lat_lon_origin_city]['lon'])
        utm_proj = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(
                self.lat_lon_origin[self.lat_lon_origin_city]['lat'],
                self.lat_lon_origin[self.lat_lon_origin_city]['lon']))
        origin_utm = {
            "origin_x": origin_x,
            "origin_y": origin_y,
            "zone_number": zone_number,
            "zone_letter": zone_letter,
            "utm_proj": utm_proj
        }

        return origin_utm

    def get_source_dataset_image_count(self) -> int:
        """[获取源数据集图片数量]

        Returns:
            int: [源数据集图片数量]
        """

        image_count = 0
        sync_data_folder = os.path.join(self.dataset_input_folder, 'sync_data')
        for n in os.listdir(sync_data_folder):
            batch_folder = os.path.join(sync_data_folder, n)
            if os.path.isdir(batch_folder):
                for m in os.listdir(
                        os.path.join(batch_folder, 'annotation_bev_image')):
                    if os.path.splitext(m)[-1].replace('.', '') in \
                                self.source_dataset_image_form_list:
                        image_count += 1

        return image_count

    def get_source_dataset_annotation_count(self) -> int:
        """[获取源数据集标注文件数量]

        Returns:
            int: [源数据集标注文件数量]
        """

        annotation_count = 0
        xml_root = os.path.join(self.dataset_input_folder, 'xml')
        for n in os.listdir(xml_root):
            pitch_xml = os.path.join(self.dataset_input_folder, 'xml', n)
            if os.path.isdir(pitch_xml):
                for m in os.listdir(pitch_xml):
                    if m == 'annotations.xml':
                        annotation_count += 1

        return annotation_count

    def get_source_dataset_map_count(self) -> int:
        """[获取源数据集标注文件数量]

        Returns:
            int: [源数据集标注文件数量]
        """

        map_count = 0
        for n in os.listdir(os.path.join(self.dataset_input_folder, 'osm')):
            if os.path.splitext(n)[-1].replace(
                    '.', '') == self.source_dataset_map_form:
                map_count += 1

        return map_count

    def get_image_ego_pose(self) -> dict:
        """获取图片坐标位置

        Returns:
            dict: 图片坐标位置字典
        """
        print('\n Start get image ego pose:')
        sync_data_folder = os.path.join(self.dataset_input_folder, 'sync_data')
        image_ego_pose_dict = {}
        for n in tqdm(os.listdir(sync_data_folder), desc='Get image ego pose'):
            batch_folder = os.path.join(sync_data_folder, n)
            if os.path.isdir(batch_folder):
                annotation_bev_image_list = sorted(
                    os.listdir(
                        os.path.join(batch_folder, 'annotation_bev_image')))
                for index, m in enumerate(annotation_bev_image_list):
                    poses = []
                    with open(
                            os.path.join(os.path.join(batch_folder,
                                                      'pose.txt')), "r") as f:
                        for line in f:
                            data = line.split(" ")
                            poses.append(data)
                    if os.path.splitext(m)[-1].replace('.', '') in \
                                self.source_dataset_image_form_list:
                        image_name = str(m).split(os.sep)[-1].split(
                            '.')[0] + '.' + self.temp_image_form
                        image_name_new = self.file_prefix + image_name
                        if 0 == index:
                            image_ego_pose_dict.update({
                                image_name_new:
                                list(map(float, poses[index]))
                            })
                        else:
                            image_ego_pose_dict.update({
                                image_name_new:
                                list(map(float, poses[index * 5 - 1]))
                            })

        image_ego_pose_json_path = os.path.join(
            self.source_dataset_pose_folder, 'image_ego_pose.json')
        with open(image_ego_pose_json_path, "w+") as j:
            j.write(json.dumps(image_ego_pose_dict))

        return image_ego_pose_dict

    def get_image_time_stamp(self) -> dict:
        """获取图片时间戳

        Returns:
            dict: 图片时间戳字典
        """
        print('\n Start get image time:')
        sync_data_folder = os.path.join(self.dataset_input_folder, 'sync_data')
        image_time_stamp_dict = {}
        for n in tqdm(os.listdir(sync_data_folder), desc='Get image time'):
            batch_folder = os.path.join(sync_data_folder, n)
            if os.path.isdir(batch_folder):
                annotation_bev_image_list = sorted(
                    os.listdir(
                        os.path.join(batch_folder, 'annotation_bev_image')))
                for index, m in enumerate(annotation_bev_image_list):
                    time_stamp_list = []
                    with open(
                            os.path.join(
                                os.path.join(batch_folder, 'times.txt')),
                            "r") as f:
                        for line in f:
                            data = line.replace('\n', '')
                            time_stamp_list.append(data)
                    if os.path.splitext(m)[-1].replace('.', '') in \
                                self.source_dataset_image_form_list:
                        image_name = str(m).split(os.sep)[-1].split(
                            '.')[0] + '.' + self.temp_image_form
                        image_name_new = self.file_prefix + image_name
                        if 0 == index:
                            image_time_stamp_dict.update(
                                {image_name_new: time_stamp_list[index]})
                        else:
                            image_time_stamp_dict.update({
                                image_name_new:
                                time_stamp_list[index * 5 - 1]
                            })

        image_time_stamp_json_path = os.path.join(
            self.source_dataset_time_folder, 'image_time_stamp.json')
        with open(image_time_stamp_json_path, "w+") as j:
            j.write(json.dumps(image_time_stamp_dict))

        return image_time_stamp_dict

    def get_lanelet_layers(self) -> dict:
        """获取lanelet地图

        Returns:
            dict: _description_
        """

        proj = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(
                self.lat_lon_origin[self.lat_lon_origin_city]['lat'],
                self.lat_lon_origin[self.lat_lon_origin_city]['lon']))
        lanelet_layers = {}
        for n in os.listdir(self.source_dataset_map_folder):
            lanelet_layers[n] = lanelet2.io.load(
                os.path.join(self.source_dataset_map_folder, n),
                proj).laneletLayer

        return lanelet_layers

    def get_lanelet_linestringlayer(self) -> dict:
        """获取lanelet地图

        Returns:
            dict: _description_
        """

        proj = lanelet2.projection.UtmProjector(
            lanelet2.io.Origin(
                self.lat_lon_origin[self.lat_lon_origin_city]['lat'],
                self.lat_lon_origin[self.lat_lon_origin_city]['lon']))
        lanelet_linestringlayer = {}
        for n in os.listdir(self.source_dataset_map_folder):
            lanelet_linestringlayer[n] = lanelet2.io.load(
                os.path.join(self.source_dataset_map_folder, n),
                proj).lineStringLayer

        return lanelet_linestringlayer

    def source_dataset_copy_image_and_annotation(self) -> None:
        """拷贝图片和标注文件
        """

        print('\nStart source dataset copy image, annotation and map:')
        if not self.only_statistic:
            sync_data_folder = os.path.join(self.dataset_input_folder,
                                            'sync_data')
            pbar, update = multiprocessing_object_tqdm(
                self.source_dataset_image_count, 'Copy images', leave=True)
            for n in os.listdir(sync_data_folder):
                batch_folder = os.path.join(sync_data_folder, n)
                if os.path.isdir(batch_folder):
                    pool = multiprocessing.Pool(self.workers)
                    annotation_bev_image_folder = os.path.join(
                        batch_folder, 'annotation_bev_image')
                    for m in os.listdir(annotation_bev_image_folder):
                        pool.apply_async(self.source_dataset_copy_image,
                                         args=(annotation_bev_image_folder, m),
                                         callback=update,
                                         error_callback=err_call_back)
                    pool.close()
                    pool.join()
            pbar.close()

        xml_root = os.path.join(self.dataset_input_folder, 'xml')
        for n in os.listdir(xml_root):
            pitch_xml = os.path.join(self.dataset_input_folder, 'xml', n)
            if os.path.isdir(pitch_xml):
                for m in os.listdir(pitch_xml):
                    if m == 'annotations.xml':
                        annotation_path = os.path.join(pitch_xml, m)
                        temp_annotation_path = os.path.join(
                            self.source_dataset_annotations_folder,
                            n + '_' + m)
                        shutil.copy(annotation_path, temp_annotation_path)

        map_osm_folder = os.path.join(self.dataset_input_folder, 'osm')
        for n in tqdm(os.listdir(map_osm_folder), desc='Copy maps'):
            if os.path.splitext(n)[-1].replace('.', '') == \
                                self.source_dataset_map_form:
                map_osm_path = os.path.join(map_osm_folder, n)
                temp_map_osm_path = os.path.join(
                    self.source_dataset_map_folder, self.file_prefix + n)
                shutil.copy(map_osm_path, temp_map_osm_path)

        print('Copy copy images, annotations and maps end.')

        return

    def draw_mask(self, ann_img):
        if self.annotation_car == 'hq1':
            # 给红旗1数据加mask
            pts = np.array(
                [[[2851.58, 1200], [2929.35, 1169.51], [3050.25, 1145.83],
                  [3088.43, 1049.67], [3116.36, 1013.61], [3123.43, 1007.25],
                  [3143.58, 994.52], [3197.67, 927.00], [3199.43, 1200],
                  [2853.34, 1200]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array([[[0, 1028.33], [21.59, 1054.00], [51.05, 1071.38],
                             [93.76, 1125.00], [107.02, 1155.64],
                             [293.85, 1200], [1.32, 1200], [0, 1090.86]]],
                           np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[2898.67, 600], [3161.83, 490.60], [3198.49, 341.79],
                  [3198.49, 600], [2900.87, 600]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[2133, 347.36], [2188.71, 365.69], [2242.95, 469.05],
                  [2282.54, 600.26], [2133, 600.26], [2133, 350.29]]],
                np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[1066, 565.81], [1114.79, 557.01], [1232.81, 548.95],
                  [1540.69, 556.28], [1539.96, 498.37], [1556.82, 467.58],
                  [1822.18, 463.18], [1870.56, 491.04], [1870.56, 548.95],
                  [2019.37, 543.08], [2133.00, 531.36], [2133, 600],
                  [1066, 600], [1066, 568.01]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[861.15, 600], [911.00, 436.79], [968.18, 345.89],
                  [1010.70, 338.56], [1028.29, 326.83], [1066.41, 331.23],
                  [1064.94, 600], [864.82, 600]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[0, 0], [22.40, 0], [49.52, 342.01], [81.78, 505.48],
                  [325.15, 600], [0, 600], [0, 271.16]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array([[[1893.17, 546.85], [2133.50, 526.10],
                             [2133.69, 549.58], [1893.35, 548.13]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[1129.20, 556.10], [1207.11, 549.80], [1249.32, 547.71],
                  [1413.71, 552.69], [1495.25, 554.26], [1517.54, 556.36],
                  [1129.80, 557.10]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))

        return ann_img

    def source_dataset_copy_image(self, annotation_bev_image_folder: str,
                                  n: str) -> None:
        """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

        Args:
            root (str): [文件所在目录]
            n (str): [文件名]
        """

        if os.path.splitext(n)[-1].replace('.', '') not in \
                                self.source_dataset_image_form_list:
            return
        image_path = os.path.join(annotation_bev_image_folder, n)
        temp_image_path = os.path.join(self.source_dataset_images_folder,
                                       self.file_prefix + n)
        temp_annotation_image_path = os.path.join(
            self.source_dataset_annotation_image_folder, self.file_prefix + n)
        image_suffix = os.path.splitext(n)[-1].replace('.', '')

        # 拷贝image
        if image_suffix != self.target_dataset_image_form:
            image_transform_type = image_suffix + \
                '_' + self.target_dataset_image_form
            image_form_transform.__dict__[image_transform_type](
                image_path, temp_image_path)
            if self.draw_car_mask:
                image = cv2.imread(temp_image_path)
                image = self.draw_mask(image)
                camera = image[0:self.camera_image_wh[1], :]
                annotation_image = image[self.camera_image_wh[1]:, :]
                cv2.imwrite(temp_image_path, camera)
                cv2.imwrite(temp_annotation_image_path, annotation_image)
        else:
            if self.draw_car_mask:
                image = cv2.imread(image_path)
                image = self.draw_mask(image)
                camera = image[0:self.camera_image_wh[1], :]
                annotation_image = image[self.camera_image_wh[1]:, :]
                cv2.imwrite(temp_image_path, camera)
                cv2.imwrite(temp_annotation_image_path, annotation_image)
            else:
                image = cv2.imread(image_path)
                camera = image[0:self.camera_image_wh[1], :]
                annotation_image = image[self.camera_image_wh[1]:, :]
                cv2.imwrite(temp_image_path, camera)
                cv2.imwrite(temp_annotation_image_path, annotation_image)

        return

    def create_dense_pcd_map_bev_image(self):
        """获取稠密点云地图bev图片
        """

        meter_per_pixel = self.pcd_meter_per_pixel
        pixel_per_meter = 1 / meter_per_pixel
        pcd_dir = os.path.join(self.dense_pcd_map_folder,
                               self.lat_lon_origin_city)
        pic_dir = self.dense_pcd_map_bev_image_folder
        location_path = os.path.join(pic_dir,
                                     'dense_pcd_map_bev_image_location.json')
        scale = {}
        scale['meter_per_pixel'] = meter_per_pixel

        file_name_dict = {}
        count = 0
        for folderName, _, filenames in tqdm(
                os.walk(pcd_dir),
                desc='Create dense pcd map bev image',
                leave=True):
            names = sorted(filenames)
            for filename in tqdm(names, desc='Pcd file', leave=False):
                pic_name = str(count)
                count += 1
                file_name_dict.update({filename: pic_name})

        if self.map_type_transform_style is not None:
            for folderName, _, filenames in tqdm(
                    os.walk(pcd_dir),
                    desc='Create dense pcd map bev image',
                    leave=True):
                names = sorted(filenames)
                pool = multiprocessing.Pool(self.workers if 1== self.workers else 4)
                pbar, update = multiprocessing_list_tqdm(names,
                                                         desc='Total pcd',
                                                         leave=False)
                for filename in tqdm(names, desc='Pcd file', leave=False):
                    pcd_input_path = os.path.join(folderName, filename)
                    pcd_output_path = os.path.join(
                        self.source_dense_pcd_map_folder,
                        file_name_dict[filename] + '.pcd')
                    # cloud_mercator_2_utm_map(pcd_input_path, pcd_output_path, mercator, origin_utm)
                    if self.map_type_transform_style == 'utm':
                        pool.apply_async(func=self.cloud_mercator_2_utm_map,
                                        args=(
                                            pcd_input_path,
                                            pcd_output_path,
                                        ),
                                        callback=update,
                                        error_callback=err_call_back)

                pool.close()
                pool.join()
                pbar.close()

        all_locations = []

        names = sorted(os.listdir(self.source_dense_pcd_map_folder))
        pool = multiprocessing.Pool(self.workers if 1== self.workers else 4)
        pbar, update = multiprocessing_list_tqdm(names,
                                                 desc='Total images',
                                                 leave=False)
        for filename in tqdm(names, desc='Pcd file', leave=False):
            all_locations.append(
                pool.apply_async(func=self.create_bev_pointcloud_image,
                                 args=(filename, pic_dir, meter_per_pixel,
                                       pixel_per_meter),
                                 callback=update,
                                 error_callback=err_call_back))
        pool.close()
        pool.join()
        pbar.close()

        all_locations_output = []
        if len(all_locations):
            for n in all_locations:
                all_locations_output.append(n.get())
            location_file = open(location_path, 'w+')
            json.dump(all_locations_output, location_file)
            location_file.close()

        self.dense_pcd_map_location_dict_list = all_locations_output
        self.dense_pcd_map_location_total_dict = {}
        for dense_map_dict in self.dense_pcd_map_location_dict_list:
            self.dense_pcd_map_location_total_dict.update(
                {dense_map_dict['name']: dense_map_dict})

        return

    def xy2ll_Mercator_map(self, x: float, y: float, mercator: Proj):
        lon, lat = mercator(x, y, inverse=True)
        return lat, lon

    def ll2xy_Utm_map(self, lat: float, lon: float, origin_utm: dict):
        """latitude and longtitude to x and y"""
        # global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
        easting, northing, _, _ = utm.from_latlon(lat, lon,
                                                  origin_utm['zone_number'],
                                                  origin_utm['zone_letter'])
        x = easting - origin_utm['origin_x']
        y = northing - origin_utm['origin_y']
        return x, y

    def cloud_mercator_2_utm_map(self, pcd_input_path: str,
                                 pcd_output_path: str) -> None:
        """将mercator点云转换为utm点云

        Args:
            pcd_input_path (str): 点云输入路径
            pcd_output_path (str): 点云输出路径
        """

        origin_utm = self.get_utm_origin()
        lat_string = str(self.lat_lon_origin[self.lat_lon_origin_city]['lat'])
        lon_string = str(self.lat_lon_origin[self.lat_lon_origin_city]['lon'])
        proj_string = '+proj=tmerc +lat_0=' + lat_string + ' +lon_0=' + lon_string + ' +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
        mercator = Proj(proj_string, preserve_units=False)

        pcd = o3d.io.read_point_cloud(pcd_input_path)
        pcd_points = np.asarray(pcd.points)
        lat, lon = self.xy2ll_Mercator_map(pcd_points[:, 0], pcd_points[:, 1],
                                           mercator)
        pcd_points[:, 0], pcd_points[:, 1] = self.ll2xy_Utm_map(
            lat, lon, origin_utm)
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        o3d.io.write_point_cloud(pcd_output_path, pcd)

        return

    def create_bev_pointcloud_image(self, filename: str, pic_dir: str,
                                    meter_per_pixel: float,
                                    pixel_per_meter: float) -> dict:
        """创建稠密点云鸟瞰图

        Args:
            filename (str): 点云文件名
            pic_dir (str): 图片输出路径
            meter_per_pixel (float): 每米像素个数
            pixel_per_meter (float): 每像素距离为多少米

        Returns:
            dict: 地图像素与坐标关系信息
        """
        
        file_path = os.path.join(self.source_dense_pcd_map_folder, filename)
        pic_name = filename.split('.')[0]
        pic_path = os.path.join(pic_dir, pic_name + '.jpg')
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        max_x = np.max(points[:, 0])
        min_x = np.min(points[:, 0])
        max_y = np.max(points[:, 1])
        min_y = np.min(points[:, 1])
        location = {}
        location['name'] = pic_name
        location['min_x'] = min_x
        location['max_x'] = max_x
        location['min_y'] = min_y
        location['max_y'] = max_y
        colors = np.asarray(pcd.colors) * 255
        points = np.column_stack((points, colors))
        pic_width = int((max_x - min_x) / meter_per_pixel)
        pic_height = int((max_y - min_y) / meter_per_pixel)
        points[:, 0] = np.floor((points[:, 0] - min_x) * pixel_per_meter)
        points[:, 1] = np.floor((max_y - points[:, 1]) * pixel_per_meter)
        mask_w = (points[:, 0] >= 0) & (points[:, 0] < pic_width)
        mask_h = (points[:, 1] >= 0) & (points[:, 1] < pic_height)
        mask = mask_w & mask_h
        points = points[mask]
        bev_pointcloud_image_np = np.zeros([pic_height, pic_width, 3],
                                           np.uint8)
        u = points[:, 0].astype(np.int)
        v = points[:, 1].astype(np.int)
        cloud_color = points[:, 3:6].astype(np.uint8)
        bev_pointcloud_image_np[v, u] = cloud_color
        bev_pointcloud_image_np = bev_pointcloud_image_np[:, :, ::-1]
        cv2.imwrite(pic_path, bev_pointcloud_image_np)  # 保存；

        return location

    def transform_to_temp_dataset(self) -> None:
        """[转换标注文件为暂存标注]
        """

        print('\nStart transform to temp dataset:')
        success_count = 0
        fail_count = 0
        no_object = 0
        temp_file_name_list = []
        total_source_dataset_annotations_list = os.listdir(
            self.source_dataset_annotations_folder)
        if not self.only_local_map:
            for source_annotation_name in tqdm(
                    total_source_dataset_annotations_list,
                    desc='Total annotations'):
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
                source_annotations_path = os.path.join(
                    self.source_dataset_annotations_folder,
                    source_annotation_name)
                tree = ET.parse(source_annotations_path)
                root = tree.getroot()
                pbar, update = multiprocessing_list_tqdm(root,
                                                         desc='Total images',
                                                         leave=False)
                for annotation in root:
                    if annotation.tag != 'image':
                        continue
                    pool.apply_async(func=self.load_image_annotation_map,
                                     args=(annotation, process_output),
                                     callback=update,
                                     error_callback=err_call_back)
                pool.close()
                pool.join()
                pbar.close()

                # debug
                # for annotation in tqdm(root, desc='Total images', leave=False):
                #     if annotation.tag != 'image':
                #         continue
                #     self.load_image_annotation(annotation, process_output)
        else:
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
            pbar, update = multiprocessing_list_tqdm(os.listdir(
                self.temp_images_folder),
                                                     desc='Total images',
                                                     leave=False)
            for image_name_new in os.listdir(self.temp_images_folder):
                pool.apply_async(func=self.load_image_map,
                                 args=(image_name_new, process_output),
                                 callback=update,
                                 error_callback=err_call_back)
            pool.close()
            pool.join()
            pbar.close()

            # debug
            # for image_name_new in tqdm(os.listdir(self.temp_images_folder),
            #                            desc='Total images',
            #                            leave=True):
            #     self.load_image_map(image_name_new, process_output)

            success_count += process_output['success_count']
            fail_count += process_output['fail_count']
            no_object = process_output['no_object']
            temp_file_name_list += process_output['temp_file_name_list']

        # 输出读取统计结果
        print('\nSource dataset convert to temp dataset file count: ')
        print('Total annotations:         \t {} '.format(
            len(total_source_dataset_annotations_list)))
        print('Convert fail:              \t {} '.format(fail_count))
        print('No object delete images: \t {} '.format(no_object))
        print('Convert success:           \t {} '.format(success_count))
        self.temp_annotation_name_list = temp_file_name_list
        print('Source dataset annotation transform to temp dataset end.')

        return

    def load_image_annotation_map(self, annotation,
                                  process_output: dict) -> None:
        """将源标注转换为暂存标注

        Args:
            source_annotation_name (str): 源标注文件名称
            process_output (dict): 进程间通信字典
        """

        image_name = str(annotation.attrib['name']).split(
            os.sep)[-1].split('.')[0] + '.' + self.temp_image_form
        image_name_new = self.file_prefix + image_name
        image_time_stamp = self.image_time_stamp_dict[image_name_new]
        image_path = os.path.join(self.temp_images_folder, image_name_new)
        channels = 0
        if not self.only_statistic:
            if not os.path.exists(image_path):
                print('\n {} not exist.'.format(image_name_new))
                process_output['fail_count'] += 1
                return
            img = cv2.imread(image_path)
            if img is None:
                print('Can not load: {}'.format(image_name_new))
                process_output['fail_count'] += 1
                return
            channels = img.shape[-1]
        else:
            channels = 3
        width = int(annotation.attrib['width'])
        if not self.camera_label_image_concat:
            height = int(annotation.attrib['height'])
        else:
            height = int(
                int(annotation.attrib['height']) - self.camera_image_wh[1])
        object_list = []
        annotation_children_node = annotation.getchildren()
        # get object box head orin
        head_points_dict = {}
        for obj in annotation_children_node:
            if obj.tag != 'points':  # 只处理points标签
                continue
            obj_children_node = obj.getchildren()
            if obj_children_node[0].text is None:
                continue
            head_point_id = ''.join(
                list(filter(str.isnumeric, obj_children_node[0].text)))
            point_list = list(map(float, obj.attrib['points'].split(',')))
            head_points_dict.update({head_point_id: point_list})
        # get object information
        for n, obj in enumerate(annotation_children_node):
            if obj.tag != 'box':  # 只处理box标签
                continue
            # if 'group_id' not in obj.attrib:
            #     continue
            obj_children_node = obj.getchildren()
            for one_obj_children_node in obj_children_node:
                if one_obj_children_node.attrib['name'] == 'visibility':
                    continue
                if one_obj_children_node.text is None:
                    object_head_point_id = None
                    clss = obj.attrib['label']
                else:
                    object_head_point_id = ''.join(
                        list(filter(str.isnumeric,
                                    one_obj_children_node.text)))
                    clss = ''.join(
                        list(filter(str.isalpha, one_obj_children_node.text)))
            clss = clss.replace(' ', '').lower()
            if clss not in self.total_task_source_class_list:
                continue
            box_xywh = []
            x = int(float(obj.attrib['xtl']))
            if not self.camera_label_image_concat:
                y = int(float(obj.attrib['ytl']))
                w = int(float(obj.attrib['xbr']) - float(obj.attrib['xtl']))
                h = int(float(obj.attrib['ybr']) - float(obj.attrib['ytl']))
                box_xywh = [x, y, w, h]
                box_xtlytlxbrybr = [
                    float(obj.attrib['xtl']),
                    float(obj.attrib['ytl']),
                    float(obj.attrib['xbr']),
                    float(obj.attrib['ybr'])
                ]
            else:
                y = int(float(obj.attrib['ytl']) - self.camera_image_wh[1])
                w = int(float(obj.attrib['xbr']) - float(obj.attrib['xtl']))
                h = int(float(obj.attrib['ybr']) - float(obj.attrib['ytl']))
                box_xywh = [x, y, w, h]
                box_xtlytlxbrybr = [
                    float(obj.attrib['xtl']),
                    float(obj.attrib['ytl']) - self.camera_image_wh[1],
                    float(obj.attrib['xbr']),
                    float(obj.attrib['ybr']) - self.camera_image_wh[1]
                ]
            if 'rotation' in obj.attrib:
                box_rotation = float(obj.attrib['rotation'])
            else:
                box_rotation = 0
            # get head orientation
            if object_head_point_id is not None and object_head_point_id in head_points_dict:
                if not self.camera_label_image_concat:
                    box_head_point = head_points_dict[object_head_point_id]
                else:
                    box_head_point = head_points_dict[object_head_point_id]
                    box_head_point[
                        1] = box_head_point[1] - self.camera_image_wh[1]
            else:
                box_head_point = None
            object_list.append(
                OBJECT(n,
                       clss,
                       box_clss=clss,
                       box_xywh=box_xywh,
                       box_xtlytlxbrybr=box_xtlytlxbrybr,
                       box_rotation=box_rotation +
                       self.label_object_rotation_angle,
                       box_head_point=box_head_point,
                       need_convert=self.need_convert))

        # local map
        image_ego_pose = None
        laneline_list = None
        if self.get_local_map:
            origin_utm = self.get_utm_origin()
            lanelet_layers = self.get_lanelet_layers()
            lanelet_linestringlayer = self.get_lanelet_linestringlayer()
            # osm name
            osm_file_name = ''
            local_map_vision_box = lanelet2.core.BoundingBox2d(
                lanelet2.core.BasicPoint2d(utm_x - self.utm_offset,
                                           utm_y - self.utm_offset),
                lanelet2.core.BasicPoint2d(utm_x + self.utm_offset,
                                           utm_y + self.utm_offset))
            for temp_osm_name, lanelet_layer in (lanelet_layers).items():
                lanelets_inRegion = lanelet_layer.search(local_map_vision_box)
                if len(lanelets_inRegion):
                    for elem in lanelets_inRegion:
                        if lanelet2.geometry.distance(
                                elem, lanelet2.core.BasicPoint2d(utm_x,
                                                                 utm_y)) == 0:
                            osm_file_name = temp_osm_name
                            break

            # for temp_osm_name, lanelet_layer in (lanelet_linestringlayer).items():
            #     lanelets_inRegion = lanelet_layer.search(local_map_vision_box)
            #     if len(lanelets_inRegion):
            #         for elem in lanelets_inRegion:
            #             if lanelet2.geometry.distance(
            #                     elem, lanelet2.core.BasicPoint2d(utm_x,
            #                                                      utm_y)) == 0:
            #                 osm_file_name = temp_osm_name
            #                 break

            # if osm_file_name == '':
            #     print('Do not find osm file name!')
            #     return

            # image_ego_pose
            lat, lon, el = float(
                self.image_ego_pose_dict[image_name_new][0]), float(
                    self.image_ego_pose_dict[image_name_new][1]), float(
                        self.image_ego_pose_dict[image_name_new][2])
            utm_xy = utm.from_latlon(lat, lon, origin_utm["zone_number"],
                                     origin_utm["zone_letter"])
            utm_x, utm_y = utm_xy[0] - origin_utm["origin_x"], utm_xy[
                1] - origin_utm["origin_y"]
            utm_z = el
            image_ego_pose = {
                "latitude": self.image_ego_pose_dict[image_name_new][0],
                "longitude": self.image_ego_pose_dict[image_name_new][1],
                "elevation": self.image_ego_pose_dict[image_name_new][2],
                "utm_position.x": utm_x,
                "utm_position.y": utm_y,
                "utm_position.z": utm_z,
                "attitude.x": self.image_ego_pose_dict[image_name_new][6],
                "attitude.y": self.image_ego_pose_dict[image_name_new][7],
                "attitude.z": self.image_ego_pose_dict[image_name_new][8] +
                self.attitude_z_offset,
                "position_type":
                int(self.image_ego_pose_dict[image_name_new][9]),
                "osm_file_name": osm_file_name,
            }

            # 获取local map
            laneline_list = []
            linestring_inregion_src = lanelet_linestringlayer[
                osm_file_name].search(local_map_vision_box)
            local_map_total_line_id = []
            for n, elem in enumerate(
                    linestring_inregion_src):  #提取所有车道线id和点集（无重复）
                if elem.id not in local_map_total_line_id:
                    laneline_points_utm = []
                    if 'roadside' in elem.attributes.keys():
                        if elem.attributes['roadside'] == 'true':
                            laneline_clss = 'roadside'
                        elif elem.attributes['roadside'] == 'false':
                            laneline_clss = 'line'
                        for point in elem:
                            laneline_points_utm.append([point.x, point.y])
                        # elif 'vguideline' in elem.attributes.keys():
                        #     pass
                    if laneline_clss not in self.task_dict['Laneline'][
                            'Source_dataset_class']:
                        continue
                    temp_laneline = LANELINE(
                        laneline_id_in=n,
                        laneline_clss_in=laneline_clss,
                        laneline_points_utm_in=laneline_points_utm,
                        utm_x=utm_x,
                        utm_y=utm_y,
                        att_z=self.image_ego_pose_dict[image_name_new][8] +
                        self.attitude_z_offset,
                        label_image_wh=self.label_image_wh,
                        label_range=self.label_range,
                        label_image_self_car_uv=self.label_image_self_car_uv)
                    if 0 == len(temp_laneline.laneline_points_label_image):
                        continue
                    laneline_list.append(temp_laneline)
                    local_map_total_line_id.append(elem.id)

        image = IMAGE(image_name, image_name_new, image_path, height, width,
                      channels, object_list, image_ego_pose, image_time_stamp,
                      laneline_list)
        # 读取目标标注信息，输出读取的source annotation至temp annotation
        if image is None:
            process_output['fail_count'] += 1
            return
        temp_annotation_output_path = os.path.join(
            self.temp_annotations_folder,
            image.file_name_new + '.' + self.temp_annotation_form)
        image.object_class_modify(self)
        image.laneline_class_modify(self)
        image.object_pixel_limit(self)
        # image.laneline_curvature_limit(self)
        # if 0 == len(image.object_list) and not self.keep_no_object:
        #     print('{} no object, has been delete.'.format(
        #         image.image_name_new))
        #     if not self.only_statistic:
        #         os.remove(image.image_path)
        #     process_output['no_object'] += 1
        #     process_output['fail_count'] += 1
        #     return
        if image.output_temp_annotation(temp_annotation_output_path):
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            print('errow output temp annotation: {}'.format(
                image.file_name_new))
            process_output['fail_count'] += 1

        return

    def load_image_map(self, image_name_new, process_output: dict) -> None:
        """将源标注转换为暂存标注

        Args:
            source_annotation_name (str): 源标注文件名称
            process_output (dict): 进程间通信字典
        """

        image_path = os.path.join(self.temp_images_folder, image_name_new)
        annotation_image_path = os.path.join(
            self.source_dataset_annotation_image_folder, image_name_new)
        image_time_stamp = self.image_time_stamp_dict[image_name_new]
        channels = 0
        if not os.path.exists(annotation_image_path):
            print('\n {} not exist.'.format(image_name_new))
            process_output['fail_count'] += 1
            return
        img = cv2.imread(annotation_image_path)
        if img is None:
            print('Can not load: {}'.format(image_name_new))
            process_output['fail_count'] += 1
            return
        channels = img.shape[-1]
        width = int(img.shape[1])
        if not self.camera_label_image_concat:
            height = int(img.shape[0] + self.camera_image_wh[1])
        else:
            height = int(img.shape[0])

        # get object annotation
        object_list = []

        # local map
        image_ego_pose = None
        laneline_list = None
        origin_utm = self.get_utm_origin()
        lanelet_layers = self.get_lanelet_layers()
        lanelet_linestringlayer = self.get_lanelet_linestringlayer()

        # image_ego_pose
        lat, lon, el = float(
            self.image_ego_pose_dict[image_name_new][0]), float(
                self.image_ego_pose_dict[image_name_new][1]), float(
                    self.image_ego_pose_dict[image_name_new][2])
        utm_xy = utm.from_latlon(lat, lon, origin_utm["zone_number"],
                                 origin_utm["zone_letter"])
        utm_x, utm_y = utm_xy[0] - origin_utm["origin_x"], utm_xy[
            1] - origin_utm["origin_y"]
        utm_z = el

        # osm name
        osm_file_name = ''
        local_map_vision_box = lanelet2.core.BoundingBox2d(
            lanelet2.core.BasicPoint2d(utm_x - self.utm_offset,
                                       utm_y - self.utm_offset),
            lanelet2.core.BasicPoint2d(utm_x + self.utm_offset,
                                       utm_y + self.utm_offset))
        for temp_osm_name, lanelet_layer in (lanelet_layers).items():
            lanelets_inRegion = lanelet_layer.search(local_map_vision_box)
            if len(lanelets_inRegion):
                for elem in lanelets_inRegion:
                    if lanelet2.geometry.distance(
                            elem, lanelet2.core.BasicPoint2d(utm_x,
                                                             utm_y)) == 0:
                        osm_file_name = temp_osm_name
                        break

        # for temp_osm_name, lanelet_layer in (lanelet_linestringlayer).items():
        #     lanelets_inRegion = lanelet_layer.search(local_map_vision_box)
        #     if len(lanelets_inRegion):
        #         for elem in lanelets_inRegion:
        #             if lanelet2.geometry.distance(
        #                     elem, lanelet2.core.BasicPoint2d(utm_x,
        #                                                      utm_y)) == 0:
        #                 osm_file_name = temp_osm_name
        #                 break

        if osm_file_name == '':
            print('Do not find osm file name!')
            return

        image_ego_pose = {
            "latitude": self.image_ego_pose_dict[image_name_new][0],
            "longitude": self.image_ego_pose_dict[image_name_new][1],
            "elevation": self.image_ego_pose_dict[image_name_new][2],
            "utm_position.x": utm_x,
            "utm_position.y": utm_y,
            "utm_position.z": utm_z,
            "attitude.x": self.image_ego_pose_dict[image_name_new][6],
            "attitude.y": self.image_ego_pose_dict[image_name_new][7],
            "attitude.z": self.image_ego_pose_dict[image_name_new][8] +
            self.attitude_z_offset,
            "position_type": int(self.image_ego_pose_dict[image_name_new][9]),
            "osm_file_name": osm_file_name,
        }

        # 获取local map
        laneline_list = []
        linestring_inregion_src = lanelet_linestringlayer[
            osm_file_name].search(local_map_vision_box)
        local_map_total_line_id = []
        for n, elem in enumerate(linestring_inregion_src):  #提取所有车道线id和点集（无重复）
            if elem.id not in local_map_total_line_id:
                laneline_points_utm = []
                laneline_clss = ''
                if 'roadside' in elem.attributes.keys():
                    if elem.attributes['roadside'] == 'true':
                        laneline_clss = 'roadside'
                    elif elem.attributes['roadside'] == 'false':
                        laneline_clss = 'line'
                    for point in elem:
                        laneline_points_utm.append([point.x, point.y])
                    # elif 'vguideline' in elem.attributes.keys():
                    #     pass
                if laneline_clss not in self.task_dict['Laneline'][
                        'Source_dataset_class']:
                    continue
                temp_laneline = LANELINE(
                    laneline_id_in=n,
                    laneline_clss_in=laneline_clss,
                    laneline_points_utm_in=laneline_points_utm,
                    utm_x=utm_x,
                    utm_y=utm_y,
                    att_z=self.image_ego_pose_dict[image_name_new][8] +
                    self.attitude_z_offset,
                    label_image_wh=self.label_image_wh,
                    label_range=self.label_range,
                    label_image_self_car_uv=self.label_image_self_car_uv)
                if 0 == len(temp_laneline.laneline_points_label_image):
                    continue
                laneline_list.append(temp_laneline)
                local_map_total_line_id.append(elem.id)
        image = IMAGE(image_name_new, image_name_new, image_path, height,
                      width, channels, object_list, image_ego_pose,
                      image_time_stamp, laneline_list)
        # 读取目标标注信息，输出读取的source annotation至temp annotation
        if image is None:
            process_output['fail_count'] += 1
            return
        temp_annotation_output_path = os.path.join(
            self.temp_annotations_folder,
            image.file_name_new + '.' + self.temp_annotation_form)
        image.object_class_modify(self)
        image.laneline_class_modify(self)
        image.object_pixel_limit(self)
        # image.laneline_curvature_limit(self)
        if 0 == len(image.laneline_list) and self.delete_no_map:
            print('{} no aneline, has been delete.'.format(
                image.image_name_new))
            # if not self.only_statistic:
            #     os.remove(image.image_path)
            process_output['no_object'] += 1
            process_output['fail_count'] += 1
            return
        if image.output_temp_annotation(temp_annotation_output_path):
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            print('errow output temp annotation: {}'.format(
                image.file_name_new))
            process_output['fail_count'] += 1

        return

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base) -> None:
        """输出target annotation

        Args:
            dataset_instance (Dataset_Base): 数据集实例
        """

        print('\nStart transform to target dataset:')
        class_color_encode_dict = {}
        for task_class_dict in dataset_instance.task_dict.values():
            if task_class_dict is not None:
                for n in task_class_dict['Target_dataset_class']:
                    class_color_encode_dict.update({n: 0})
        for n, key in zip(
                random.sample([x for x in range(255)],
                              len(class_color_encode_dict)),
                class_color_encode_dict.keys()):
            class_color_encode_dict[key] = RGB_to_Hex(
                str(n) + ',' + str(n) + ',' + str(n))

        # 生成空基本信息xml文件
        annotations = dataset.__dict__[
            dataset_instance.target_dataset_style].annotation_creat_root(
                dataset_instance, class_color_encode_dict)
        # 获取全部图片标签信息列表
        for task, task_class_dict in tqdm(dataset_instance.task_dict.items(),
                                          desc='Load each task annotation'):
            if task_class_dict is None \
                    or task_class_dict['Target_dataset_class'] is None:
                continue
            total_image_xml = []
            pbar, update = multiprocessing_list_tqdm(
                dataset_instance.temp_annotations_path_list,
                desc='transform to target dataset',
                leave=False)
            pool = multiprocessing.Pool(dataset_instance.workers)
            for temp_annotation_path in dataset_instance.temp_annotations_path_list:
                total_image_xml.append(
                    pool.apply_async(func=dataset.__dict__[
                        dataset_instance.target_dataset_style].
                                     annotation_get_temp,
                                     args=(
                                         dataset_instance,
                                         temp_annotation_path,
                                         task,
                                         task_class_dict,
                                     ),
                                     callback=update,
                                     error_callback=err_call_back))
            pool.close()
            pool.join()
            pbar.close()

            # 将image标签信息添加至annotations中
            for n, image in enumerate(total_image_xml):
                annotation = image.get()
                annotation.attrib['id'] = str(n)
                annotations.append(annotation)

            tree = ET.ElementTree(annotations)

            annotation_output_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder,
                'annotatons.' +
                dataset_instance.target_dataset_annotation_form)
            tree.write(annotation_output_path,
                       encoding='utf-8',
                       xml_declaration=True)

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
            image_path = os.path.join(
                self.source_dataset_annotation_image_folder, image.image_name)
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

    @staticmethod
    def annotation_creat_root(dataset_instance: Dataset_Base,
                              class_color_encode_dict: dict) -> object:
        """创建xml根节点

        Args:
            dataset_instance (dict): 数据集信息字典
            class_color_encode_dict (dict): 色彩字典

        Returns:
            object: ET.Element格式标注信息
        """

        class_id = 0
        annotations = ET.Element('annotations')
        version = ET.SubElement(annotations, 'version')
        version.text = '1.1'
        meta = ET.SubElement(annotations, 'meta')
        task = ET.SubElement(meta, 'task')
        # ET.SubElement(task, 'id')
        # ET.SubElement(task, 'name')
        # ET.SubElement(task, 'size')
        # mode = ET.SubElement(task, 'mode')
        # mode.text = 'annotation'
        # overlap = ET.SubElement(task, 'overlap')
        # ET.SubElement(task, 'bugtracker')
        # ET.SubElement(task, 'created')
        # ET.SubElement(task, 'updated')
        # subset = ET.SubElement(task, 'subset')
        # subset.text = 'default'
        # start_frame = ET.SubElement(task, 'start_frame')
        # start_frame.text='0'
        # ET.SubElement(task, 'stop_frame')
        # ET.SubElement(task, 'frame_filter')
        # segments = ET.SubElement(task, 'segments')
        # segment = ET.SubElement(segments, 'segment')
        # ET.SubElement(segments, 'id')
        # start = ET.SubElement(segments, 'start')
        # start.text='0'
        # ET.SubElement(segments, 'stop')
        # ET.SubElement(segments, 'url')
        # owner = ET.SubElement(task, 'owner')
        # ET.SubElement(owner, 'username')
        # ET.SubElement(owner, 'email')
        # ET.SubElement(task, 'assignee')
        labels = ET.SubElement(task, 'labels')

        class_dict_list_output_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            'class_dict_list.txt')
        with open(class_dict_list_output_path, 'w') as f:
            for n, c in class_color_encode_dict.items():
                label = ET.SubElement(labels, 'label')
                name = ET.SubElement(label, 'name')
                name.text = n
                color = ET.SubElement(label, 'color')
                color.text = c
                attributes = ET.SubElement(label, 'attributes')
                attribute = ET.SubElement(attributes, 'attribute')
                name = ET.SubElement(attribute, 'name')
                name.text = '1'
                mutable = ET.SubElement(attribute, 'mutable')
                mutable.text = 'False'
                input_type = ET.SubElement(attribute, 'input_type')
                input_type.text = 'text'
                default_value = ET.SubElement(attribute, 'default_value')
                default_value.text = None
                values = ET.SubElement(attribute, 'values')
                values.text = None
                # 输出标签色彩txt
                s = '  {\n    "name": "'+n+'",\n    "color": "' + \
                    str(c)+'",\n    "attributes": []\n  },\n'
                f.write(s)
                class_id += 1

            # ET.SubElement(task, 'dumped')
        return annotations

    @staticmethod
    def annotation_get_temp(dataset_instance: Dataset_Base,
                            temp_annotation_path: str, task: str,
                            task_class_dict: dict) -> object:
        """[获取temp标签信息]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]
            temp_annotation_path (str): [暂存标签路径]
            task (str): 任务类型
            task_class_dict (dict): 任务对应类别字典

        Returns:
            object: [ET.Element格式标注信息]
        """

        image = dataset_instance.TEMP_LOAD(dataset_instance,
                                           temp_annotation_path)
        if image == None:
            return
        image_xml = ET.Element(
            'image', {
                'id': '',
                'name': image.image_name_new,
                'width': str(image.width),
                'height': str(image.height)
            })
        for object in image.object_list:
            if task == 'Detection':
                clss = object.box_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                if object.box_exist_flag:
                    box = ET.SubElement(
                        image_xml, 'box', {
                            'label': object.box_clss,
                            'occluded': '0',
                            'source': 'manual',
                            'xtl': str(object.box_xywh[0]),
                            'ytl': str(object.box_xywh[1]),
                            'xbr':
                            str(object.box_xywh[0] + object.box_xywh[2]),
                            'ybr':
                            str(object.box_xywh[1] + object.box_xywh[3]),
                            'z_order': "0",
                            'group_id': str(object.object_id)
                        })
                    attribute = ET.SubElement(box, 'attribute', {'name': '1'})
                    attribute.text = object.box_clss + str(object.object_id)
            elif task == 'Semantic_segmentation':
                segmentation = np.asarray(
                    object.segmentation).flatten().tolist()
                clss = object.segmentation_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                if object.segmentation_exist_flag:
                    point_list = []
                    for x in object.segmentation:
                        point_list.append(str(x[0]) + ',' + str(x[1]))
                    if 2 == len(point_list):
                        continue
                    polygon = ET.SubElement(
                        image_xml, 'polygon', {
                            'label': object.segmentation_clss,
                            'occluded': '0',
                            'source': 'manual',
                            'points': ';'.join(point_list),
                            'z_order': "0",
                            'group_id': str(object.object_id)
                        })
                    attribute = ET.SubElement(polygon, 'attribute',
                                              {'name': '1'})
                    attribute.text = object.segmentation_clss + \
                        str(object.object_id)
            elif task == 'Instance_segmentation':
                segmentation = np.asarray(
                    object.segmentation).flatten().tolist()
                clss = object.segmentation_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                box = ET.SubElement(
                    image_xml, 'box', {
                        'label': object.box_clss,
                        'occluded': '0',
                        'source': 'manual',
                        'xtl': str(object.box_xywh[0]),
                        'ytl': str(object.box_xywh[1]),
                        'xbr': str(object.box_xywh[0] + object.box_xywh[2]),
                        'ybr': str(object.box_xywh[1] + object.box_xywh[3]),
                        'z_order': "0",
                        'group_id': str(object.object_id)
                    })
                attribute = ET.SubElement(box, 'attribute', {'name': '1'})
                attribute.text = object.box_clss + str(object.object_id)
                point_list = []
                for x in object.segmentation:
                    point_list.append(str(x[0]) + ',' + str(x[1]))
                if 2 == len(point_list):
                    continue
                polygon = ET.SubElement(
                    image_xml, 'polygon', {
                        'label': object.segmentation_clss,
                        'occluded': '0',
                        'source': 'manual',
                        'points': ';'.join(point_list),
                        'z_order': "0",
                        'group_id': str(object.object_id)
                    })
                attribute = ET.SubElement(polygon, 'attribute', {'name': '1'})
                attribute.text = object.segmentation_clss + str(
                    object.object_id)
            elif task == 'Keypoint':
                clss = object.keypoints_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                if object.keypoints_exist_flag:
                    for m, xy in object.keypoints:
                        points = ET.SubElement(
                            image_xml, 'points', {
                                'label': object.keypoints_clss,
                                'occluded': '0',
                                'source': 'manual',
                                'points': str(xy[0]) + ',' + str(xy[1]),
                                'z_order': "0",
                                'group_id': str(object.object_id)
                            })
                        attribute = ET.SubElement(points, 'attribute',
                                                  {'name': '1'})
                        attribute.text = str(object.object_id) + '-' + m

        return image_xml

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取CVAT_IMAGE_BEV_2数据集图片类检测列表]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []
        dataset_instance.total_file_name_path = total_file(
            dataset_instance.temp_informations_folder)
        dataset_instance.target_check_file_name_list = os.listdir(
            dataset_instance.target_dataset_annotations_folder
        )  # 读取target_annotations_folder文件夹下的全部文件名

        print('Start load target annotations:')
        for n in tqdm(dataset_instance.target_check_file_name_list,
                      desc='Load target dataset annotation'):
            if os.path.splitext(n)[-1] != '.xml':
                continue
            source_annotations_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder, n)
        tree = ET.parse(source_annotations_path)
        root = tree.getroot()
        random.shuffle(root)
        root = root[0:dataset_instance.target_dataset_annotations_check_count]
        for annotation in tqdm(root,
                               desc='Load object annotation',
                               leave=False):
            if annotation.tag != 'image':
                continue
            image_name = str(annotation.attrib['name'])
            image_name_new = image_name
            image_path = os.path.join(dataset_instance.temp_images_folder,
                                      image_name_new)
            img = cv2.imread(image_path)
            if img is None:
                print('Can not load: {}'.format(image_name_new))
                return
            width = int(annotation.attrib['width'])
            height = int(annotation.attrib['height'])
            channels = img.shape[-1]
            object_list = []
            for n, obj in enumerate(annotation):
                if obj.tag == 'box':
                    clss = str(obj.attrib['label'])
                    clss = clss.replace(' ', '').lower()
                    box_xywh = [
                        int(obj.attrib['xtl']),
                        int(obj.attrib['ytl']),
                        int(obj.attrib['xbr']) - int(obj.attrib['xtl']),
                        int(obj.attrib['ybr']) - int(obj.attrib['ytl'])
                    ]
                    object_list.append(
                        OBJECT(n,
                               clss,
                               box_clss=clss,
                               box_xywh=box_xywh,
                               need_convert=dataset_instance.need_convert))
                elif obj.tag == 'polygon':
                    clss = str(obj.attrib['label'])
                    clss = clss.replace(' ', '').lower()
                    segment = []
                    for seg in obj.attrib['points'].split(';'):
                        x, y = seg.split(',')
                        x = float(x)
                        y = float(y)
                        segment.append(list(map(int, [x, y])))
                    object_list.append(
                        OBJECT(n,
                               clss,
                               segmentation_clss=clss,
                               segmentation=segment,
                               need_convert=dataset_instance.need_convert))
            image = IMAGE(image_name, image_name, image_path, int(height),
                          int(width), int(channels), object_list)
            check_images_list.append(image)

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成CVAT_IMAGE_BEV_2组织格式的数据集]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVAT_IMAGE_BEV_3_MAP'))
        shutil.rmtree(output_root)
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVAT_IMAGE_BEV_3_MAP'))
        annotations_output_folder = check_output_path(
            os.path.join(output_root, 'annotations'))

        print('Start copy images:')
        image_list = []
        image_output_folder = check_output_path(
            os.path.join(output_root, 'images'))
        with open(dataset_instance.temp_divide_file_list[0], 'r') as f:
            for n in f.readlines():
                image_list.append(n.replace('\n', ''))
        pbar, update = multiprocessing_list_tqdm(image_list,
                                                 desc='Copy images',
                                                 leave=False)
        pool = multiprocessing.Pool(dataset_instance.workers)
        for image_input_path in image_list:
            image_output_path = image_input_path.replace(
                dataset_instance.temp_images_folder, image_output_folder)
            pool.apply_async(func=shutil.copy,
                             args=(
                                 image_input_path,
                                 image_output_path,
                             ),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        print('Start copy annotations:')
        for root, dirs, files in os.walk(
                dataset_instance.target_dataset_annotations_folder):
            for n in tqdm(files, desc='Copy annotations'):
                annotations_input_path = os.path.join(root, n)
                annotations_output_path = annotations_input_path.replace(
                    dataset_instance.target_dataset_annotations_folder,
                    annotations_output_folder)
                shutil.copy(annotations_input_path, annotations_output_path)

        return
