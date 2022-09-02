'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-17 17:43:06
'''
import shutil
import multiprocessing

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base


class CITYSCAPES(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'json'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base) -> None:
        """[输出target annotation]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart transform to target dataset:')
        total_annotation_path_list = []
        for dataset_temp_annotation_path_list in tqdm(dataset_instance.temp_divide_file_list[1:-1],
                                                      desc='Get total annotation path list'):
            with open(dataset_temp_annotation_path_list, 'r') as f:
                for n in f.readlines():
                    total_annotation_path_list.append(n.replace('\n', '')
                                                      .replace(dataset_instance.source_dataset_images_folder,
                                                               dataset_instance.temp_annotations_folder)
                                                      .replace(dataset_instance.target_dataset_image_form,
                                                               dataset_instance.temp_annotation_form))

        pbar, update = multiprocessing_list_tqdm(
            total_annotation_path_list, desc='Output target dataset annotation')
        pool = multiprocessing.Pool(dataset_instance.workers)
        for temp_annotation_path in total_annotation_path_list:
            pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].annotation_output,
                             args=(dataset_instance,
                                   temp_annotation_path,),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        return

    @staticmethod
    def annotation_output(dataset_instance: Dataset_Base, temp_annotation_path: str) -> None:
        """[读取暂存annotation]

        Args:
            dataset_instance (): [数据集信息字典]
            temp_annotation_path (str): [annotation路径]

        Returns:
            IMAGE: [输出IMAGE类变量]
        """

        image = dataset_instance.TEMP_LOAD(
            dataset_instance, temp_annotation_path)
        if image == None:
            return
        # 图片基础信息
        annotation_output_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            image.file_name + '.' + dataset_instance.target_dataset_annotation_form)
        annotation = {'imgHeight': image.height,
                      'imgWidth': image.width,
                      'objects': []
                      }
        segmentation = {}
        for object in image.object_list:
            segmentation = {'label': object.segmentation_clss,
                            'polygon': object.segmentation
                            }
            annotation['objects'].append(segmentation)
        json.dump(annotation, open(annotation_output_path, 'w'))

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取CITYSCAPES数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []
        dataset_instance.total_file_name_path = total_file(
            dataset_instance.temp_informations_folder)
        dataset_instance.target_check_file_name_list = os.listdir(
            dataset_instance.target_dataset_annotations_folder)  # 读取target_annotations_folder文件夹下的全部文件名
        dataset_instance.target_dataset_check_file_name_list = annotations_path_list(
            dataset_instance.total_file_name_path,
            dataset_instance.target_dataset_annotations_check_count)
        print('Start load target annotations:')
        for n in tqdm(dataset_instance.target_dataset_check_file_name_list,
                      desc='Load target dataset annotation'):
            target_annotation_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder,
                n + '.' + dataset_instance.target_dataset_annotation_form)
            with open(target_annotation_path, 'r') as f:
                data = json.loads(f.read())
                image_name = n + '.' + dataset_instance.target_dataset_image_form
                image_path = os.path.join(
                    dataset_instance.temp_images_folder, image_name)
                image_size = cv2.imread(image_path).shape
                height = image_size[0]
                width = image_size[1]
                channels = image_size[2]
                object_list = []
                for id, object in enumerate(data['objects']):
                    clss = str(object['label'])
                    clss = clss.replace(' ', '').lower()
                    segmentation = object['polygon']
                    object_list.append(
                        OBJECT(id, clss, clss, clss, clss, segmentation=segmentation))
                image = IMAGE(image_name, image_name, image_path, int(
                    height), int(width), int(channels), object_list)
                check_images_list.append(image)

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成CITYSCAPES组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        # 官方数值
        # colors = [
        #     [128, 64, 128], [244, 35, 232], [70, 70, 70], [
        #         102, 102, 156], [190, 153, 153],
        #     [153, 153, 153], [250, 170, 30], [220, 220, 0], [
        #         107, 142, 35],  [152, 251, 152],
        #     [0, 130, 180],  [220, 20, 60],  [
        #         255, 0, 0],  [0, 0, 142],     [0, 0, 70],
        #     [0, 60, 100],   [0, 80, 100],   [0, 0, 230],  [119, 11, 32], [0, 0, 0]]
        # label_colours = dict(zip(range(19), colors))
        # void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        # valid_classes = [7, 8, 11, 12, 13, 17, 19, 20,
        #                  21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        # class_map = dict(zip(valid_classes, range(19)))
        # class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
        #                'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
        #                'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
        #                'motorcycle', 'bicycle']
        # ignore_index = 255
        # num_classes_ = 19
        # class_weights_ = np.array([0.05570516, 0.32337477, 0.08998544, 1.03602707, 1.03413147, 1.68195437,
        #                            5.58540548, 3.56563995, 0.12704978, 1., 0.46783719, 1.34551528,
        #                            5.29974114, 0.28342531, 0.9396095, 0.81551811, 0.42679146, 3.6399074,
        #                            2.78376194], dtype=float)
        # labelIds
        # class_names_dict = {'unlabeled': 0,
        #                     'egovehicle': 1,
        #                     'rectificationborder': 2,
        #                     'outofroi': 3,
        #                     'static': 4,
        #                     'dynamic': 5,
        #                     'ground': 6,
        #                     'road': 7,
        #                     'sidewalk': 8,
        #                     'parking': 9,
        #                     'railtrack': 10,
        #                     'building': 11,
        #                     'wall': 12,
        #                     'fence': 13,
        #                     'guardrail': 14,
        #                     'bridge': 15,
        #                     'tunnel': 16,
        #                     'pole': 17,
        #                     'polegroup': 18,
        #                     'trafficlight': 19,
        #                     'trafficsign': 20,
        #                     'vegetation': 21,
        #                     'terrain': 22,
        #                     'sky': 23,
        #                     'person': 24,
        #                     'rider': 25,
        #                     'car': 26,
        #                     'truck': 27,
        #                     'bus': 28,
        #                     'caravan': 29,
        #                     'trailer': 30,
        #                     'train': 31,
        #                     'motorcycle': 32,
        #                     'bicycle': 33,
        #                     'licenseplate': -1,
        #                     }
        # road lane classes
        # class_names_dict = {'unlabeled': 0,
        #                     'road': 1,
        #                     'lane': 2,
        #                     }

        class_names_dict = {}
        for x, clss in enumerate(dataset_instance.task_dict['Semantic_segmentation']
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

        output_root = check_output_path(os.path.join(
            dataset_instance.dataset_output_folder, 'cityscapes', 'data'))   # 输出数据集文件夹
        cityscapes_folder_list = ['gtFine', 'leftImg8bit']
        data_divion_name = ['train', 'test', 'val']
        output_folder_path_list = []
        # 创建cityscapes组织结构
        print('Clean dataset folder!')
        shutil.rmtree(output_root)
        print('Create new folder:')
        for n in cityscapes_folder_list:
            output_folder_path = check_output_path(
                os.path.join(output_root, n))
            output_folder_path_list.append(output_folder_path)
            for m in data_divion_name:
                dataset_division_folder_path = os.path.join(
                    output_folder_path, m)
                check_output_path(dataset_division_folder_path)
                check_output_path(os.path.join(
                    dataset_division_folder_path,
                    dataset_instance.file_prefix.replace(
                        dataset_instance.file_prefix_delimiter, '')))

        for n in tqdm(dataset_instance.temp_divide_file_list[1:4],
                      desc='Create annotation file to output folder'):
            dataset_name = os.path.splitext(n.split(os.sep)[-1])[0]
            with open(n, 'r') as f:
                data_list = f.read().splitlines()
                pbar, update = multiprocessing_list_tqdm(data_list,
                                                         desc='Create annotation file to {} folder:'.format(dataset_name),
                                                         leave=False)
                pool = multiprocessing.Pool(dataset_instance.workers)
                for x in data_list:
                    pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].create_annotation_file,
                                     args=(dataset_instance, file_name_dict, output_folder_path_list,
                                           dataset_name, class_names_dict, x),
                                     callback=update,
                                     error_callback=err_call_back)
                pool.close()
                pool.join()
                pbar.close()

        return

    @staticmethod
    def create_annotation_file(dataset_instance: Dataset_Base, file_name_dict: dict, output_folder_path_list: list,
                               dataset_name: str, class_names_dict: dict, x: str, ) -> None:
        """[创建cityscapes格式数据集]

        Args:
            dataset (dict): [数据集信息字典]
            file_name_dict (dict): [全量数据编号字典]
            output_folder_path_list (list): [输出文件夹路径列表]
            dataset_name (str): [划分后数据集名称]
            class_names_dict (dict): [labelIds类别名对应id字典]
            x (str): [标签文件名称]
        """

        file = os.path.splitext(x.split(os.sep)[-1])[0]
        file_out = dataset_instance.file_prefix.replace(
            dataset_instance.file_prefix_delimiter, '') + '_000000_' + \
            str(format(file_name_dict[file], '06d'))
        # 调整image
        image_out = file_out + '_leftImg8bit' + \
            '.' + dataset_instance.target_dataset_image_form
        image_path = os.path.join(
            dataset_instance.temp_images_folder, file + '.' + dataset_instance.target_dataset_image_form)
        image_output_path = os.path.join(
            output_folder_path_list[1], dataset_name, dataset_instance.file_prefix.replace(
                dataset_instance.file_prefix_delimiter, ''), image_out)
        # 调整annotation
        annotation_out = file_out + '_gtFine_polygons' + \
            '.' + dataset_instance.target_dataset_annotation_form
        temp_annotation_path = os.path.join(
            dataset_instance.temp_annotations_folder, file + '.' + dataset_instance.temp_annotation_form)
        annotation_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder, file + '.' + dataset_instance.target_dataset_annotation_form)
        annotation_output_path = os.path.join(
            output_folder_path_list[0], dataset_name, dataset_instance.file_prefix.replace(
                dataset_instance.file_prefix_delimiter, ''), annotation_out)
        # 调整annotation为_gtFine_labelIds.png
        image = dataset_instance.TEMP_LOAD(dataset_instance, temp_annotation_path)
        labelIds_out = file_out + '_gtFine_labelIds.png'
        labelIds_output_path = os.path.join(
            output_folder_path_list[0], dataset_name, dataset_instance.file_prefix.replace(
                dataset_instance.file_prefix_delimiter, ''), labelIds_out)
        # 输出
        shutil.copy(image_path, image_output_path)
        shutil.copy(annotation_path, annotation_output_path)

        zeros = np.zeros((image.height, image.width), dtype=np.uint8)
        if len(image.object_list):
            for object in image.object_list:
                class_color = class_names_dict[object.segmentation_clss]
                points = np.array(object.segmentation)
                zeros_mask = cv2.fillPoly(
                    zeros, pts=[points], color=class_color)
            cv2.imwrite(labelIds_output_path, zeros_mask)
        else:
            cv2.imwrite(labelIds_output_path, zeros)

        return
