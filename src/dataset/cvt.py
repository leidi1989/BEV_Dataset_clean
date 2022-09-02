'''
Description: CVT
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-09-01 13:58:06
'''
import multiprocessing
import shutil
from pathlib import Path
import yaml
import time
from PIL import Image

import numpy as np
import torch
from base.dataset_base import Dataset_Base
from base.image_base import *
from utils.utils import *

import dataset


class CVT(Dataset_Base):

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
        """[读取CVT数据集图片类检测列表]

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
        """[生成CVT组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        # base setup
        # timestamp
        base_timestamp = int(round(time.time() * 1000))
        # token
        base_token = '_'.join([
            dataset_instance.dataset_output_folder.split(os.sep)[-2],
            str(base_timestamp)
        ])
        scene_token = '_'.join([base_token, 'scene'])
        scene_name = dataset_instance.dataset_input_folder.split(os.sep)[-1]
        # scene_name = 'scene-0001'
        # 生成组织结构文件夹
        # generate CVT folders
        CVT_output_path = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVT'))
        print('Clean dataset folder!')
        shutil.rmtree(CVT_output_path)
        print('Create new folder:')
        CVT_output_path = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVT'))

        # generate nuscenes folders
        nuscenes_output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVT', 'nuScenes'))  # 输出数据集文件夹
        nuscenes_output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVT', 'nuScenes'))  # 输出数据集文件夹
        maps_folder_path = check_output_path(
            os.path.join(nuscenes_output_root, 'maps'))
        samples_folder_path = check_output_path(
            os.path.join(nuscenes_output_root, 'samples'))
        v1_0_trainval_folder_path = check_output_path(
            os.path.join(nuscenes_output_root, 'v1.0-trainval'))

        camera_name_list = [
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT',
            'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT'
        ]
        camera_image_folder_list = []
        for folder_name in camera_name_list:
            camera_image_folder_list.append(
                check_output_path(
                    os.path.join(samples_folder_path, folder_name)))

        # generate cvt_labels_nuscenes
        cvt_labels_nuscenes_output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVT',
                         'cvt_labels_nuscenes'))  # 输出数据集文件夹
        cvt_labels_nuscenes_scene = check_output_path(
            os.path.join(cvt_labels_nuscenes_output_root,
                         scene_name))  # 输出数据集文件夹

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

        # get camera extrinsics intrinsics

        dataset_instance.camera_calibration_extrinsics = get_camera_calibration_extrinsics(
            dataset_instance)
        dataset_instance.camera_calibration_intrinsics = get_camera_calibration_intrinsics(
            dataset_instance)

        # 输出sample.json
        pbar, update = multiprocessing_list_tqdm(
            total_annotation_path_list,
            desc='Output target dataset annotation')
        pool = multiprocessing.Pool(dataset_instance.workers)
        image_information_dict_list = []
        for temp_annotation_path in total_annotation_path_list:
            image_information_dict_list.append(
                pool.apply_async(func=dataset.__dict__[
                    dataset_instance.target_dataset_style].extract_information,
                                 args=(dataset_instance, temp_annotation_path,
                                       camera_name_list,
                                       camera_image_folder_list, scene_token,
                                       scene_name, cvt_labels_nuscenes_scene),
                                 callback=update,
                                 error_callback=err_call_back))
        pool.close()
        pool.join()
        pbar.close()

        cvt_labels_nuscenes_json = []
        for image_information_dict in image_information_dict_list:
            cvt_labels_nuscenes_json.append(image_information_dict.get())
        cvt_labels_nuscenes_json_output_path = os.path.join(
            cvt_labels_nuscenes_output_root, '.'.join([scene_name, 'json']))
        json.dump(cvt_labels_nuscenes_json,
                  open(cvt_labels_nuscenes_json_output_path, 'w'))

        return

    @staticmethod
    def extract_information(dataset_instance: Dataset_Base,
                            temp_annotation_path: str, camera_name_list: list,
                            camera_image_folder_list: list, scene_token: str,
                            scene_name: str,
                            cvt_labels_nuscenes_scene: str) -> dict:
        """创建CVT格式数据集

        Args:
            dataset_instance (Dataset_Base): 数据集信息字典
            temp_annotation_path (str): 暂存标注文件路径
        """
        image = dataset_instance.TEMP_LOAD(dataset_instance,
                                           temp_annotation_path)
        object_class_list = dataset_instance.task_dict['Detection'][
            'Target_dataset_class']
        w = 200
        h = 200

        image_information_dict = {}
        # generate nuscenes images 6400,2400 3*2
        total_concate_image = cv2.imread(image.image_path)
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
        images_path_list = []
        for camera_image, camera_name, camera_image_output_path in zip(
                camera_image_list, camera_name_list, camera_image_folder_list):
            image_name = image.image_name_new.split('.')
            image_name_new = image_name[
                0] + '_' + camera_name + '.' + image_name[1]
            cv2.imwrite(os.path.join(camera_image_output_path, image_name_new),
                        camera_image)
            images_path_list.append(
                os.path.join(
                    os.sep.join(camera_image_output_path.split(os.sep)[-3:-1]),
                    image_name_new))

        # camera_calibration_intrinsics
        intrinsics_list = []
        for intrinsic in dataset_instance.camera_calibration_intrinsics.values(
        ):
            intrinsics_list.append(intrinsic.tolist())
        # camera_calibration_extrinsics
        extrinsics_list = []
        for extrinsic in dataset_instance.camera_calibration_extrinsics.values(
        ):
            extrinsics_list.append(extrinsic.tolist())

        # bev, aux, visibility image name
        bev_image_name = '_'.join([
            "bev",
            image.image_name_new.replace(
                ('.' + dataset_instance.target_dataset_image_form), '.png')
        ])
        aux_file_name = '_'.join([
            "aux",
            image.image_name_new.replace(
                ('.' + dataset_instance.target_dataset_image_form), '.npz')
        ])
        visibility_image_name = '_'.join([
            "visibility",
            image.image_name_new.replace(
                ('.' + dataset_instance.target_dataset_image_form), '.png')
        ])

        # view
        bev_dict = {
            'h': h,
            'w': w,
            'h_meters': 100,
            'w_meters': 100,
            'offset': 0.0
        }
        view = get_view_matrix(bev_dict['h'], bev_dict['w'],
                               bev_dict['h_meters'], bev_dict['w_meters'],
                               bev_dict['offset'])
        view_list = view.tolist()
        bev_shape = (bev_dict['h'], bev_dict['w'])
        # pose

        # pose_inverse

        # image_information_dict
        image_information_dict = {
            'images':
            images_path_list,
            'intrinsics':
            intrinsics_list,
            'extrinsics':
            extrinsics_list,
            "view":
            view_list,
            "bev":
            bev_image_name,
            "aux":
            aux_file_name,
            "visibility":
            visibility_image_name,
            "pose":
            [[-0.3455987274646759, 0.9383823871612549, 0.0, 411.3039245605469],
             [
                 -0.9383823871612549, -0.3455987274646759, 0.0,
                 1180.890380859375
             ], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "pose_inverse":
            [[
                -0.3455987274646759, -0.9383823871612549, 0.0,
                1250.2728271484375
            ],
             [0.9383823871612549, -0.3455987274646759, 0.0, 22.15386962890625],
             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            "cam_ids": [x for x in range(len(camera_name_list))],
            "cam_channels":
            camera_name_list,
            "token":
            scene_token,
            "scene":
            scene_name
        }

        # dataset x,y to w,h scale
        x_to_w_scale = (dataset_instance.label_range[2] +
                        dataset_instance.label_range[3]) / (
                            dataset_instance.label_image_wh[0])
        y_to_h_scale = (dataset_instance.label_range[0] +
                        dataset_instance.label_range[1]) / (
                            dataset_instance.label_image_wh[1])

        # generate aux and bev
        bev = np.zeros(bev_shape, dtype=np.uint8)
        segmentation = np.zeros((h, w), dtype=np.uint8)
        center_score = np.zeros((h, w), dtype=np.float32)
        center_offset = np.zeros((h, w, 2), dtype=np.float32)
        center_ohw = np.zeros((h, w, 4), dtype=np.float32)
        buf = np.zeros((h, w), dtype=np.uint8)
        center_hw = np.zeros((h, 2), dtype=np.float32)
        center_angle = np.zeros((h, 1), dtype=np.float32)
        center_ind = np.zeros((h, 1), dtype=np.int64)
        center_reg = np.zeros((h, 2), dtype=np.float32)
        center_reg_mask = np.zeros((h, 1), dtype=np.uint8)
        visibility = np.full((h, w), 255, dtype=np.uint8)
        coords = np.stack(np.meshgrid(np.arange(w), np.arange(h)),
                          -1).astype(np.float32)

        for k, object in enumerate(image.object_list):
            buf.fill(0)
            box_rotated_rect_points = []
            for points in object.box_rotated_rect_points:
                x = points[0] * x_to_w_scale
                y = (points[1] -
                     dataset_instance.camera_image_wh[1]) * y_to_h_scale
                box_rotated_rect_points.append([x, y])
            cv2.fillPoly(buf,
                         [np.array(box_rotated_rect_points).astype(np.int32)],
                         1, cv2.LINE_8)
            center = np.array([
                (object.box_xywh[0] + object.box_xywh[2]) / 2 * x_to_w_scale,
                (object.box_xywh[1] + object.box_xywh[3]) / 2 * y_to_h_scale
            ])
            front = np.array([
                (box_rotated_rect_points[2][0] + box_rotated_rect_points[3][0])
                / 2,
                (box_rotated_rect_points[2][1] + box_rotated_rect_points[3][1])
                / 2
            ])
            left = np.array([
                (box_rotated_rect_points[1][0] + box_rotated_rect_points[2][0])
                / 2,
                (box_rotated_rect_points[1][1] + box_rotated_rect_points[2][1])
                / 2
            ])

            mask = buf > 0
            if not np.count_nonzero(mask):
                continue
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
            if (center[0] > 199.9999) or (center[1] > 199.9999):
                continue
            ct_int = np.absolute(center.astype(np.int32))
            sigma = 1
            segmentation[mask] = 255
            center_offset[mask] = center[None] - coords[mask]
            center_score[mask] = np.exp(-(center_offset[mask]**2).sum(-1) /
                                        (sigma**2))
            # orientation, h/2, w/2
            center_ohw[mask,
                       0:2] = ((front - center) /
                               (np.linalg.norm(front - center) + 1e-6))[None]
            center_ohw[mask, 2:3] = np.linalg.norm(front - center)
            center_ohw[mask, 3:4] = np.linalg.norm(left - center)
            angle = front - center
            ang = math.atan2(angle[0], angle[1]) * 180 / math.pi
            ang = 180 - ang
            center_angle[k] = ang
            center_hw[k] = ww, hh

            center_ind[k] = ct_int[1] * 200 + ct_int[0]
            center_reg[k] = center - ct_int
            center_reg_mask[k] = 1
            visibility[mask] = object_class_list.index(object.box_clss)
            bev[mask] = object_class_list.index(object.box_clss)

        center_hw = torch.from_numpy(center_hw).repeat_interleave(
            h, dim=0).view(h, h, 2).numpy()
        center_reg = torch.from_numpy(center_reg).repeat_interleave(
            h, dim=0).view(h, h, 2).numpy()
        center_ind = torch.from_numpy(center_ind).repeat_interleave(
            h, dim=0).view(h, h, 1).numpy()
        center_reg_mask = torch.from_numpy(center_reg_mask).repeat_interleave(
            h, dim=0).view(h, h, 1).numpy()
        center_angle = torch.from_numpy(center_angle).repeat_interleave(
            h, dim=0).view(h, h, 1).numpy()

        segmentation = np.float32(segmentation[..., None])
        center_score = center_score[..., None]
        result = np.concatenate(
            (segmentation, center_score, center_offset, center_ohw, center_hw,
             center_reg, center_ind, center_reg_mask, center_angle), 2)

        # generate aux
        aux_output_path = os.path.join(cvt_labels_nuscenes_scene,
                                       aux_file_name)
        np.savez(aux_output_path, aux=result)

        # save bev
        bev_image = Image.fromarray(bev)
        bev_image_output_path = os.path.join(cvt_labels_nuscenes_scene,
                                             bev_image_name)
        bev_image.save(bev_image_output_path)

        # generate visibility
        visibility_image = Image.fromarray(visibility)
        visibility_image_output_path = os.path.join(cvt_labels_nuscenes_scene,
                                                    visibility_image_name)
        visibility_image.save(visibility_image_output_path)

        return image_information_dict


def get_camera_calibration_extrinsics(dataset_instance: Dataset_Base) -> dict:

    camera_calibration_extrinsics_dict = {}
    for camera_name, path in dataset_instance.camera_calibration_extrinsics_file_path_dict.items(
    ):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            extrinsics_np_array = np.array(
                data['CameraExtrinsicMat']['data']).reshape(
                    (data['CameraExtrinsicMat']['rows'],
                     data['CameraExtrinsicMat']['cols']))
            camera_calibration_extrinsics_dict.update(
                {camera_name: extrinsics_np_array})

    return camera_calibration_extrinsics_dict


def get_camera_calibration_intrinsics(dataset_instance: Dataset_Base) -> dict:

    camera_calibration_intrinsics_dict = {}
    for camera_name, path in dataset_instance.camera_calibration_intrinsics_file_path_dict.items(
    ):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
            intrinsics_np_array = np.array(data['CameraMat']['data']).reshape(
                (data['CameraMat']['rows'], data['CameraMat']['cols']))
            camera_calibration_intrinsics_dict.update(
                {camera_name: intrinsics_np_array})

    return camera_calibration_intrinsics_dict


def get_view_matrix(h=200, w=200, h_meters=100.0, w_meters=100.0, offset=0.0):
    sh = h / h_meters
    sw = w / w_meters

    return np.float32([[0., -sw, w / 2.], [-sh, 0., h * offset + h / 2.],
                       [0., 0., 1.]])
