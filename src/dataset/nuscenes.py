'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-09-13 10:09:12
'''
import multiprocessing
import shutil
from PIL import Image
import time

from base.dataset_base import Dataset_Base
from base.image_base import *
from utils.utils import *

import dataset


class NUSCENES(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'json'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count(
        )

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base) -> None:
        """[输出target annotation]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart transform to target dataset:')
        total_annotation_path_list = []
        for dataset_temp_annotation_path_list in tqdm(
                dataset_instance.temp_divide_file_list[1:-1],
                desc='Get total annotation path list'):
            with open(dataset_temp_annotation_path_list, 'r') as f:
                for n in f.readlines():
                    total_annotation_path_list.append(
                        n.replace('\n', '').replace(
                            dataset_instance.source_dataset_images_folder,
                            dataset_instance.temp_annotations_folder).replace(
                                dataset_instance.target_dataset_image_form,
                                dataset_instance.temp_annotation_form))

        pbar, update = multiprocessing_list_tqdm(
            total_annotation_path_list,
            desc='Output target dataset annotation')
        pool = multiprocessing.Pool(dataset_instance.workers)
        for temp_annotation_path in total_annotation_path_list:
            pool.apply_async(func=dataset.__dict__[
                dataset_instance.target_dataset_style].annotation_output,
                             args=(
                                 dataset_instance,
                                 temp_annotation_path,
                             ),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        return

    @staticmethod
    def annotation_output(dataset_instance: Dataset_Base,
                          temp_annotation_path: str) -> None:
        """[读取暂存annotation]

        Args:
            dataset_instance (): [数据集信息字典]
            temp_annotation_path (str): [annotation路径]

        Returns:
            IMAGE: [输出IMAGE类变量]
        """

        image = dataset_instance.TEMP_LOAD(dataset_instance,
                                           temp_annotation_path)
        if image == None:
            return
        # 图片基础信息
        target_annotation_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            temp_annotation_path.split(os.sep)[-1])
        shutil.copy(temp_annotation_path, target_annotation_path)

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取nuscenes数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []
        dataset_instance.total_file_name_path = total_file(
            dataset_instance.temp_informations_folder)
        dataset_instance.target_check_file_name_list = os.listdir(
            dataset_instance.target_dataset_annotations_folder
        )  # 读取target_annotations_folder文件夹下的全部文件名
        dataset_instance.target_dataset_check_file_name_list = annotations_path_list(
            dataset_instance.total_file_name_path,
            dataset_instance.target_dataset_annotations_check_count)
        print('Start load target annotations:')
        for n in tqdm(dataset_instance.target_dataset_check_file_name_list,
                      desc='Load target dataset annotation'):
            target_annotation_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder,
                n + '.' + dataset_instance.target_dataset_annotation_form)
            image = dataset_instance.TEMP_LOAD(dataset_instance,
                                               target_annotation_path)
            if image == None:
                continue
            check_images_list.append(image)

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成NUSCENES组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        # generate nuscenes folders
        nuscenes_output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'nuScenes'))  # 输出数据集文件夹
        print('Clean dataset folder!')
        shutil.rmtree(nuscenes_output_root)
        print('Create new folder:')
        nuscenes_output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'nuScenes'))  # 输出数据集文件夹
        maps_folder_path = check_output_path(
            os.path.join(nuscenes_output_root, 'maps'))
        samples_folder_path = check_output_path(
            os.path.join(nuscenes_output_root, 'samples'))
        v1_0_trainval_folder_path = check_output_path(
            os.path.join(nuscenes_output_root, 'v1.0-trainval'))

        # define base data
        # timestamp
        base_timestamp = int(round(time.time() * 1000))
        # token
        base_token = '_'.join([
            dataset_instance.dataset_output_folder.split(os.sep)[-2],
            str(base_timestamp)
        ])
        scene_token = '_'.join([base_token, 'scene'])
        object_class_list = dataset_instance.task_dict['Detection'][
            'Target_dataset_class']
        camera_name_list = [
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT',
            'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
        ]

        camera_image_folder_list = []
        for folder_name in camera_name_list:
            camera_image_folder_list.append(
                check_output_path(
                    os.path.join(samples_folder_path, folder_name)))

        total_annotation_path_list = []
        for dataset_temp_annotation_path_list in tqdm(
                dataset_instance.temp_divide_file_list[1:-1],
                desc='Get total annotation path list'):
            with open(dataset_temp_annotation_path_list, 'r') as f:
                for n in f.readlines():
                    total_annotation_path_list.append(
                        n.replace('\n', '').replace(
                            dataset_instance.source_dataset_images_folder,
                            dataset_instance.temp_annotations_folder).replace(
                                dataset_instance.target_dataset_image_form,
                                dataset_instance.temp_annotation_form))

        # 1. attribute.json, 因自定义数据集暂无attribute属性，使用类别替代,
        # token: '_'.join([nuscenes_base_token, str(token)]
        nuscenes_attribute_list = []
        attribute_token_list = []
        for token, one_class in enumerate(object_class_list):
            attribute_token = '_'.join([base_token, 'attribute', str(token)])
            attribute_token_list.append(attribute_token)
            nuscenes_attribute_list.append({
                "token":
                attribute_token,
                "name":
                one_class,
                "description":
                "hy class name, token is class index {}.".format(token)
            })
        json.dump(
            nuscenes_attribute_list,
            open(os.path.join(v1_0_trainval_folder_path, 'attribute.json'),
                 'w'))

        # 2. 输出sensor.json
        nuscenes_sensor_list = []
        sensor_token_list = []
        for index, camera_name in enumerate(camera_name_list):
            nuscenes_sensor_token = '_'.join(
                [base_token, 'sensor', camera_name,
                 str(index)])
            sensor_token_list.append(nuscenes_sensor_token)
            nuscenes_sensor_list.append({
                "token": nuscenes_sensor_token,
                "channel": camera_name,
                "modality": "camera"
            })
        json.dump(
            nuscenes_sensor_list,
            open(os.path.join(v1_0_trainval_folder_path, 'sensor.json'), 'w'))

        # 3. calibrated_sensor.json
        camera_calibrated_data_list = [0, 1, 2, 3, 4, 5]
        nuscenes_calibrated_sensor_list = []
        calibrated_sensor_token_list = []
        for index, (camera_name, camera_calibrated_data,
                    sensor_token) in enumerate(
                        zip(camera_name_list, camera_calibrated_data_list,
                            sensor_token_list)):
            calibrated_sensor_token = '_'.join(
                [base_token, 'calibrated', camera_name])
            calibrated_sensor_token_list.append(calibrated_sensor_token)
            nuscenes_calibrated_sensor_list.append({
                "token":
                calibrated_sensor_token,
                "sensor_token":
                sensor_token,
                "translation": [3.412, 0.0, 0.5],
                "rotation":
                [0.9999984769132877, 0.0, 0.0, 0.0017453283658983088],
                "camera_intrinsic": []
            })
        json.dump(
            nuscenes_calibrated_sensor_list,
            open(
                os.path.join(v1_0_trainval_folder_path,
                             'calibrated_sensor.json'), 'w'))

        # 4. 输出category.json
        nuscenes_category_list = []
        category_token_list = []
        for index, one_class in enumerate(object_class_list):
            category_token = '_'.join([base_token, 'category', str(index)])
            category_token_list.append(category_token)
            nuscenes_category_list.append({
                "token":
                category_token,
                "name":
                one_class,
                "description":
                "hy class name, token is class index {}.".format(token)
            })
        json.dump(
            nuscenes_category_list,
            open(os.path.join(v1_0_trainval_folder_path, 'category.json'),
                 'w'))

        # 4. log.json
        log_token = '_'.join([base_token, 'log'])
        nuscenes_log_list = [{
            "token": log_token,
            "logfile": '_'.join(["hy_car", base_token]),
            "vehicle": "hy_car",
            "date_captured": "2022-01-1",
            "location": "hubei-wuhan"
        }]
        json.dump(
            nuscenes_log_list,
            open(os.path.join(v1_0_trainval_folder_path, 'log.json'), 'w'))

        # 5. map.json
        map_token = '_'.join([base_token, 'map'])
        nuscenes_map_list = [{
            "category": "semantic_prior",
            "token": map_token,
            "filename": "maps/0.png",
            "log_tokens": log_token
        }]
        map_image = Image.new(mode='RGB', size=(1024, 1024))
        map_image.save(os.path.join(maps_folder_path, "0.png"))
        json.dump(
            nuscenes_map_list,
            open(os.path.join(v1_0_trainval_folder_path, 'map.json'), 'w'))

        # 7. 输出visibility.json
        visibility_token_list = [
            '_'.join([base_token, 'visibility',
                      "v0-40"]), '_'.join([base_token, 'visibility',
                                           "v40-60"]),
            '_'.join([base_token, 'visibility', "v60-80"]),
            '_'.join([base_token, 'visibility', "v80-100"])
        ]
        nuscenes_visibility_list = [{
            "description": "visibility of whole object is between 0 and 40%",
            "token": visibility_token_list[0],
            "level": "v0-40"
        }, {
            "description": "visibility of whole object is between 40 and 60%",
            "token": visibility_token_list[1],
            "level": "v40-60"
        }, {
            "description": "visibility of whole object is between 60 and 80%",
            "token": visibility_token_list[2],
            "level": "v60-80"
        }, {
            "description": "visibility of whole object is between 80 and 100%",
            "token": visibility_token_list[3],
            "level": "v80-100"
        }]
        json.dump(
            nuscenes_visibility_list,
            open(os.path.join(v1_0_trainval_folder_path, 'visibility.json'),
                 'w'))

        # 8. 输出scene.json
        nbr_samples_count = len(total_annotation_path_list)
        nuscenes_scene_list = [{
            "token": scene_token,  # Unique record identifier
            "log_token": "0",  # 指向一个log，scene中的data都是从该log提取出来的
            "nbr_samples": nbr_samples_count,  # scene中的sample的数量
            "first_sample_token": "0",  # 指向场景中第一个sample
            "last_sample_token": str(nbr_samples_count - 1),  # 指向场景中最后一个sample
            "name": "hy_bev_dataset",
            "description": "hy bev dataset."
        }]
        json.dump(
            nuscenes_scene_list,
            open(os.path.join(v1_0_trainval_folder_path, 'scene.json'), 'w'))

        # 10. ego_pose.json
        pbar, update = multiprocessing_list_tqdm(
            total_annotation_path_list,
            desc='Count target dataset total object')
        object_count_dict_multiprocessing = {}
        pool = multiprocessing.Pool(dataset_instance.workers)
        for image_index, temp_annotation_path in enumerate(
                total_annotation_path_list):
            object_count_dict_multiprocessing.update({
                image_index:
                pool.apply_async(func=dataset.__dict__[
                    dataset_instance.target_dataset_style].total_object_count,
                                 args=(dataset_instance, image_index,
                                       temp_annotation_path),
                                 callback=update,
                                 error_callback=err_call_back)
            })
        pool.close()
        pool.join()
        pbar.close()

        object_count_dict = {}
        for key, value in object_count_dict_multiprocessing.items():
            object_count_dict.update({key: value.get()})

        # 9. instance.json
        # 11. sample_annotation.json
        # 12. sample_data.json
        # 13. sample.json
        # 14. calibrated_sensor.json
        # 15. generate nuscenes images 6400,2400 3*2
        pbar, update = multiprocessing_list_tqdm(
            total_annotation_path_list,
            desc='Extract target dataset annotation')
        sample_information_dict_list = []
        pool = multiprocessing.Pool(dataset_instance.workers)
        for image_index, temp_annotation_path in enumerate(
                total_annotation_path_list):
            sample_information_dict_list.append(
                pool.apply_async(
                    func=dataset.__dict__[
                        dataset_instance.target_dataset_style].extract_data,
                    args=(dataset_instance, image_index, temp_annotation_path,
                          camera_name_list, camera_image_folder_list,
                          nbr_samples_count, v1_0_trainval_folder_path,
                          scene_token, object_class_list, category_token_list,
                          attribute_token_list,
                          object_count_dict[image_index]),
                    callback=update,
                    error_callback=err_call_back))
        pool.close()
        pool.join()
        pbar.close()

        return

    @staticmethod
    def total_object_count(
        dataset_instance: Dataset_Base,
        image_index: int,
        temp_annotation_path: str,
    ):
        image = dataset_instance.TEMP_LOAD(dataset_instance,
                                           temp_annotation_path)
        image_object_count_dict = {}
        image_object_count_dict.update({image_index: len(image.object_list)})

        return image_object_count_dict

    @staticmethod
    def extract_data(dataset_instance: Dataset_Base, image_index: int,
                     temp_annotation_path: str, camera_name_list: list,
                     camera_image_folder_list: list, nbr_samples_count: int,
                     v1_0_trainval_folder_path: str, scene_token: str,
                     object_class_list: list, category_token_list: list,
                     attribute_token_list: list, object_count: int) -> dict:
        """获取bev标注结果信息

        Args:
            dataset_instance (Dataset_Base): 数据集信息字典
            image_index (int): _description_
            temp_annotation_path (str): 暂存标注路径
            camera_name_list (list): 相机名称列表
            camera_image_folder_list (list): 相机图片文件夹列表
            nbr_samples_count (int): _description_
            v1_0_trainval_folder_path (str): _description_
            scene_token (str): _description_
            object_class_list (list): _description_
            category_token_list (list): _description_
            attribute_token_list (list): _description_
            object_count (int): _description_

        Returns:
            dict: _description_
        """

        image = dataset_instance.TEMP_LOAD(dataset_instance,
                                           temp_annotation_path)
        base_timestamp = int(round(time.time() * 1000))
        # token
        base_token = '_'.join([
            dataset_instance.dataset_output_folder.split(os.sep)[-2],
            str(image_index)
        ])

        # generate nuscenes images 6400,2400 3*2
        total_concate_image = cv2.imread(image.image_path)
        dataset_instance.camera_image_wh[0] = total_concate_image.shape[1]
        dataset_instance.camera_image_wh[1] = total_concate_image.shape[0]
        CAM_BACK_image = total_concate_image[
            int(dataset_instance.camera_image_wh[1] / 2 *
                1):int(dataset_instance.camera_image_wh[1]),
            int(dataset_instance.camera_image_wh[0] / 3 *
                1):int(dataset_instance.camera_image_wh[0] / 3 * 2), :]
        CAM_BACK_LEFT_image = total_concate_image[
            int(dataset_instance.camera_image_wh[1] / 2 *
                1):int(dataset_instance.camera_image_wh[1]),
            0:int(dataset_instance.camera_image_wh[0] / 3 * 1), :]
        CAM_BACK_RIGHT_image = total_concate_image[
            int(dataset_instance.camera_image_wh[1] / 2 *
                1):int(dataset_instance.camera_image_wh[1]),
            int(dataset_instance.camera_image_wh[0] / 3 *
                2):int(dataset_instance.camera_image_wh[0]), :]
        CAM_FRONT_image = total_concate_image[
            0:int(dataset_instance.camera_image_wh[1] / 2 * 1),
            int(dataset_instance.camera_image_wh[0] / 3 *
                1):int(dataset_instance.camera_image_wh[0] / 3 * 2), :]
        CAM_FRONT_LEFT_image = total_concate_image[
            0:int(dataset_instance.camera_image_wh[1] / 2 * 1),
            0:int(dataset_instance.camera_image_wh[0] / 3 * 1), :]
        CAM_FRONT_RIGHT_image = total_concate_image[
            0:int(dataset_instance.camera_image_wh[1] / 2 * 1),
            int(dataset_instance.camera_image_wh[0] / 3 *
                1):int(dataset_instance.camera_image_wh[0]), :]
        camera_image_list = [
            CAM_BACK_image, CAM_BACK_LEFT_image, CAM_BACK_RIGHT_image,
            CAM_FRONT_image, CAM_FRONT_LEFT_image, CAM_FRONT_RIGHT_image
        ]
        image_output_path_list = []
        for camera_image, camera_name, camera_image_output_path in zip(
                camera_image_list, camera_name_list, camera_image_folder_list):
            image_name = image.image_name_new.split('.')
            camera_image_name = image_name[
                0] + '_' + camera_name + '.' + image_name[1]
            image_output_path = os.path.join(camera_image_output_path,
                                             camera_image_name)
            image_output_path_list.append(
                (os.sep).join(image_output_path.split(os.sep)[-3:]))
            cv2.imwrite(image_output_path, camera_image)

        # 9. instance.json
        # 11. sample_annotation.json
        image_instance_list = []
        image_sample_annotation_list = []
        x_to_w_scale = (dataset_instance.label_range[2] +
                        dataset_instance.label_range[3]) / (
                            dataset_instance.label_image_wh[0])
        y_to_h_scale = (dataset_instance.label_range[0] +
                        dataset_instance.label_range[1]) / (
                            dataset_instance.label_image_wh[1])
        for index, object in enumerate(image.object_list):
            image_instance_token = '_'.join(
                [base_token, 'image_instance',
                 str(index)])
            image_instance_category_token = category_token_list[
                object_class_list.index(object.box_clss)]
            instance = {
                "token": image_instance_token,
                "category_token": image_instance_category_token,
                "nbr_annotations": 1,
                "first_annotation_token": image_instance_token,
                "last_annotation_token": image_instance_token
            }
            image_instance_list.append(instance)

            image_sample_annotation_token = '_'.join(
                [base_token, 'sample_annotation',
                 str(index)])

            # box_rotated_rect_points
            box_rotated_rect_points = []
            for points in object.box_rotated_rect_points:
                x = points[0] * x_to_w_scale
                y = (points[1] -
                     dataset_instance.camera_image_wh[1]) * y_to_h_scale
                box_rotated_rect_points.append([x, y])

            # sample_annotation_translation
            center = np.array([
                (object.box_xywh[0] + object.box_xywh[2]) / 2 * x_to_w_scale,
                (object.box_xywh[1] + object.box_xywh[3]) / 2 * y_to_h_scale
            ])
            image_sample_annotation_translation = center.tolist()

            # sample_annotation_size
            a = abs(box_rotated_rect_points[3][0] -
                    box_rotated_rect_points[2][0])
            b = abs(box_rotated_rect_points[1][1] -
                    box_rotated_rect_points[2][1])
            if a >= b:
                ww = b
                hh = a
            else:
                ww = a
                hh = b
            image_sample_annotation_size = [ww, hh, 0]

            # sample_annotation_rotation
            image_sample_annotation_rotation = [0, 0, 0, 0]

            # sample_annotation_attribute_tokens
            image_sample_annotation_attribute_tokens = attribute_token_list[
                object_class_list.index(object.box_clss)]

            # prev token and next token
            # prev token
            # TODO
            if 0 == image_index:
                if 0 == index:
                    prev_token = ''
                else:
                    prev_token = '_'.join(
                        [base_token, 'sample_annotation',
                         str(index - 1)])
            else:
                if 0 == index:
                    prev_token = '_'.join([
                        dataset_instance.dataset_output_folder.split(
                            os.sep)[-2],
                        str(image_index - 1), 'sample_annotation'
                    ])
                else:
                    prev_token = '_'.join(
                        [base_token, camera_name_list[index - 1]])
            # next token
            if 5 == index:
                if image_index == nbr_samples_count:
                    next_token = ''
                else:
                    next_token = '_'.join([
                        dataset_instance.dataset_output_folder.split(
                            os.sep)[-2],
                        str(image_index + 1), camera_name_list[0]
                    ])
            else:
                next_token = '_'.join(
                    [base_token, camera_name_list[index + 1]])
            image_sample_annotation = {
                "token": image_sample_annotation_token,
                "sample_token": sample_token,
                "instance_token": image_instance_token,
                "visibility_token": "4",
                "attribute_tokens": image_sample_annotation_attribute_tokens,
                "translation": image_sample_annotation_translation,
                "size": image_sample_annotation_size,
                "rotation": image_sample_annotation_rotation,
                "prev": prev_token,
                "next": next_token,
                "num_lidar_pts": 0,
                "num_radar_pts": 0
            }
            image_sample_annotation_list.append(image_sample_annotation)
        json.dump(
            image_sample_list,
            open(os.path.join(v1_0_trainval_folder_path, 'sample.json'), 'w'))
        json.dump(
            image_sample_list,
            open(os.path.join(v1_0_trainval_folder_path, 'sample.json'), 'w'))
        # 13. sample.json
        sample_token_list = []
        for camera_name in camera_name_list:
            sample_token_list.append('_'.join([base_token, camera_name]))
        image_sample_list = []
        for index, (sample_token, camera_name) in enumerate(
                zip(sample_token_list, camera_name_list)):
            # prev token
            if 0 == image_index:
                if 0 == index:
                    prev_token = ''
                else:
                    prev_token = '_'.join(
                        [base_token, camera_name_list[index - 1]])
            else:
                if 0 == index:
                    prev_token = '_'.join([
                        dataset_instance.dataset_output_folder.split(
                            os.sep)[-2],
                        str(image_index - 1), camera_name_list[-1]
                    ])
                else:
                    prev_token = '_'.join(
                        [base_token, camera_name_list[index - 1]])
            # next token
            if 5 == index:
                if image_index == nbr_samples_count:
                    next_token = ''
                else:
                    next_token = '_'.join([
                        dataset_instance.dataset_output_folder.split(
                            os.sep)[-2],
                        str(image_index + 1), camera_name_list[0]
                    ])
            else:
                next_token = '_'.join(
                    [base_token, camera_name_list[index + 1]])

            image_sample_list.append({
                "token": sample_token,
                "timestamp": base_timestamp,
                "prev": prev_token,
                "next": next_token,
                "scene_token": scene_token
            })
        json.dump(
            image_sample_list,
            open(os.path.join(v1_0_trainval_folder_path, 'sample.json'), 'w'))

        # 12. sample_data.json
        image_sample_data_list = []
        ego_pose_list = []
        sample_data_token_list = []
        for index, (sample_token, image_output_path) in enumerate(
                zip(sample_token_list, image_output_path_list)):
            sample_data_token = '_'.join(
                [base_token, 'sample_data',
                 str(index)])
            sample_data_token_list.append(sample_data_token_list)
            image_sample_data_list.append({
                "token":
                sample_data_token,
                "sample_token":
                sample_token,
                "ego_pose_token":
                sample_data_token,
                "calibrated_sensor_token":
                str(index),
                "timestamp":
                base_timestamp,
                "fileformat":
                "jpg",
                "is_key_frame":
                "true",
                "height":
                image.height,
                "width":
                image.width,
                "filename":
                image_output_path,
                "prev":
                '_'.join([
                    dataset_instance.dataset_output_folder.split(os.sep)[-2],
                    str(image_index - 1)
                ] if 0 != image_index else ""),
                "next":
                '_'.join([
                    dataset_instance.dataset_output_folder.split(os.sep)[-2],
                    str(image_index + 1)
                ] if 0 != nbr_samples_count else "")
            })

            # ego_pose
            ego_pose_list.append({
                "token": sample_data_token,
                "timestamp": base_timestamp,
                "rotation": [0.0, 0.0, 0.0, 0.0],
                "translation": [0.0, 0.0, 0.0]
            })
        json.dump(
            image_sample_data_list,
            open(os.path.join(v1_0_trainval_folder_path, 'sample_data.json'),
                 'w'))
        json.dump(
            ego_pose_list,
            open(os.path.join(v1_0_trainval_folder_path, 'ego_pose.json'),
                 'w'))

        # 14. calibrated_sensor.json

        return {}