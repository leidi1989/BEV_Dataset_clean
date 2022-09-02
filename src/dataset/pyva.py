'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-09-02 10:45:43
'''
import shutil
import multiprocessing
from PIL import Image

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base


class PYVA(Dataset_Base):

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
        """[读取PYVA数据集图片类检测列表]

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
        """[生成PYVA组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        class_names_dict = {}
        for x, clss in enumerate(dataset_instance.task_dict['Detection']
                                 ['Target_dataset_class']):
            class_names_dict.update({clss: x})

        # 获取全量数据编号字典
        file_name_dict = {}
        print('Collect file name dict.')
        with open(dataset_instance.temp_divide_file_list[0], 'r') as f:
            for x, n in enumerate(f.read().splitlines()):
                file_name = os.path.splitext(n.split(os.sep)[-1])[0]
                file_name_dict[file_name] = x
            f.close()

        dataset_instance.target_dataset_output_folder = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'PYVA'))  # 输出数据集文件夹
        PYVA_folder_list = ['input', 'dynamic_gt', 'static_gt']
        output_folder_path_list = []
        # 创建PYVA组织结构
        print('Clean dataset folder!')
        shutil.rmtree(dataset_instance.target_dataset_output_folder)
        print('Create new folder:')
        for n in PYVA_folder_list:
            output_folder_path_list.append(
                check_output_path(
                    os.path.join(dataset_instance.target_dataset_output_folder,
                                 n)))

        with open(dataset_instance.temp_divide_file_list[0], 'r') as f:
            data_list = f.read().splitlines()
            pbar, update = multiprocessing_list_tqdm(
                data_list,
                desc='Create annotation file to output folder',
                leave=False)
            pool = multiprocessing.Pool(dataset_instance.workers)
            for source_image_path in data_list:
                pool.apply_async(func=dataset.__dict__[
                    dataset_instance.target_dataset_style].
                                 create_annotation_file,
                                 args=(dataset_instance,
                                       output_folder_path_list,
                                       class_names_dict, source_image_path),
                                 callback=update,
                                 error_callback=err_call_back)
            pool.close()
            pool.join()
            pbar.close()

        return

    @staticmethod
    def create_annotation_file(
        dataset_instance: Dataset_Base,
        output_folder_path_list: list,
        class_names_dict: dict,
        source_image_path: str,
    ) -> None:
        """[创建PYVA格式数据集]

        Args:
            dataset (dict): [数据集信息字典]
            output_folder_path_list (list): [输出文件夹路径列表]
            class_names_dict (dict): [labelIds类别名对应id字典]
            x (str): [标签文件名称]
        """

        file_name = os.path.splitext(source_image_path.split(os.sep)[-1])[0]
        input_image_output_path = os.path.join(output_folder_path_list[0],
                                               file_name + '.jpg')
        dynamic_image_output_path = os.path.join(output_folder_path_list[1],
                                                 file_name + '.png')
        dynamic_image_tensor_output_path = os.path.join(
            output_folder_path_list[1], file_name + '.npz')
        static_image_output_path = os.path.join(output_folder_path_list[2],
                                                file_name + '.png')
        static_image_tensor_output_path = os.path.join(
            output_folder_path_list[2], file_name + '.npz')
        target_annotation_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            file_name + '.json')

        if dataset_instance.source_dataset_style in ['CVAT_IMAGE_BEV_NAS']:
            # CVAT_IMAGE_BEV_2
            # input
            shutil.copy(source_image_path, input_image_output_path)

            # dynamic and static
            image = dataset_instance.TEMP_LOAD(dataset_instance,
                                               target_annotation_path)
            image_mask = np.zeros((dataset_instance.label_image_wh[1],
                                   dataset_instance.label_image_wh[0], 1),
                                  np.uint8)
            for object in image.object_list:
                if 2 == len(class_names_dict):
                    color = [255]
                else:
                    class_id = class_names_dict[object.object_clss]
                    color = [class_id]
                points = np.int32([np.array(object.box_rotated_rect_points)])
                cv2.drawContours(image_mask, [points],
                                 -1,
                                 color=color,
                                 thickness=-1)
            dynamic_static_image = image_mask

            # adjust label range
            height_scale = (dataset_instance.label_range[0] +
                            dataset_instance.label_range[1]
                            ) / dataset_instance.label_image_wh[1]
            width_scale = (dataset_instance.label_range[2] +
                           dataset_instance.label_range[3]
                           ) / dataset_instance.label_image_wh[0]
            new_range_label_left = int(dataset_instance.adjust_label_range[2] /
                                       width_scale)
            new_range_label_right = int(
                dataset_instance.adjust_label_range[3] / width_scale)
            new_range_label_front = int(
                dataset_instance.adjust_label_range[0] / height_scale)
            new_range_label_back = int(dataset_instance.adjust_label_range[1] /
                                       height_scale)

            self_car_point = [
                int(dataset_instance.label_range[2] /
                    (dataset_instance.label_range[2] +
                     dataset_instance.label_range[3]) *
                    dataset_instance.label_image_wh[0]),
                int(dataset_instance.label_range[0] /
                    (dataset_instance.label_range[0] +
                     dataset_instance.label_range[1]) *
                    dataset_instance.label_image_wh[1])
            ]

            dynamic_static_image = dynamic_static_image[
                int(self_car_point[1] -
                    new_range_label_front):int(self_car_point[1] +
                                               new_range_label_back),
                int(self_car_point[0] -
                    new_range_label_left):int(self_car_point[0] +
                                              new_range_label_right)]
            dynamic_static_image = cv2.resize(
                dynamic_static_image,
                (dataset_instance.semantic_segmentation_label_image_wh[0],
                 dataset_instance.semantic_segmentation_label_image_wh[1]),
                interpolation=cv2.INTER_NEAREST)

            dynamic_static_image = Image.fromarray(
                np.uint8(dynamic_static_image))
            dynamic_static_image.save(dynamic_image_output_path)
            dynamic_static_image.save(static_image_output_path)
        else:
            # input
            source_image = cv2.imread(source_image_path)
            input_image = source_image[0:dataset_instance.camera_image_wh[1],
                                       0:dataset_instance.camera_image_wh[0]]
            cv2.imwrite(input_image_output_path, input_image)

            # dynamic and static
            image = dataset_instance.TEMP_LOAD(dataset_instance,
                                               target_annotation_path)
            image_mask = np.zeros((image.height, image.width, 1), np.uint8)
            for object in image.object_list:
                if 2 == len(class_names_dict):
                    color = [255]
                else:
                    class_id = class_names_dict[object.object_clss]
                    color = [class_id]
                points = np.int32([np.array(object.box_rotated_rect_points)])
                cv2.drawContours(image_mask, [points],
                                 -1,
                                 color=color,
                                 thickness=-1)
            dynamic_static_image = image_mask[
                dataset_instance.camera_image_wh[1]:,
                0:dataset_instance.camera_image_wh[0]]

            # adjust label range
            height_scale = (dataset_instance.label_range[0] +
                            dataset_instance.label_range[1]
                            ) / dataset_instance.label_image_wh[1]
            width_scale = (dataset_instance.label_range[2] +
                           dataset_instance.label_range[3]
                           ) / dataset_instance.label_image_wh[0]
            new_range_label_left = int(dataset_instance.adjust_label_range[2] /
                                       width_scale)
            new_range_label_right = int(
                dataset_instance.adjust_label_range[3] / width_scale)
            new_range_label_front = int(
                dataset_instance.adjust_label_range[0] / height_scale)
            new_range_label_back = int(dataset_instance.adjust_label_range[1] /
                                       height_scale)

            self_car_point = [
                int(dataset_instance.label_range[2] /
                    (dataset_instance.label_range[2] +
                     dataset_instance.label_range[3]) *
                    dataset_instance.label_image_wh[0]),
                int(dataset_instance.label_range[0] /
                    (dataset_instance.label_range[0] +
                     dataset_instance.label_range[1]) *
                    dataset_instance.label_image_wh[1])
            ]

            dynamic_static_image = dynamic_static_image[
                int(self_car_point[1] -
                    new_range_label_front):int(self_car_point[1] +
                                               new_range_label_back),
                int(self_car_point[0] -
                    new_range_label_left):int(self_car_point[0] +
                                              new_range_label_right)]
            dynamic_static_image = cv2.resize(
                dynamic_static_image,
                (dataset_instance.semantic_segmentation_label_image_wh[0],
                 dataset_instance.semantic_segmentation_label_image_wh[1]),
                interpolation=cv2.INTER_NEAREST)

            dynamic_static_image = Image.fromarray(
                np.uint8(dynamic_static_image))
            dynamic_static_image.save(dynamic_image_output_path)
            dynamic_static_image.save(static_image_output_path)

        return
