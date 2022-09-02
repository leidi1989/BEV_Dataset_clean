'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-25 18:44:26
'''
import shutil

from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base


class HY_VAL(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'json'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    def source_dataset_copy_annotation(self, root: str, n: str) -> None:
        """[复制源数据集标签文件至目标数据集中的source_annotations中]

        Args:
            dataset (dict): [数据集信息字典]
            root (str): [文件所在目录]
            n (str): [文件名]
        """

        fake_js = {}
        if os.path.splitext(n)[-1].replace('.', '') in \
                self.source_dataset_image_form_list:
            json_name = os.path.splitext(
                n)[0] + '.' + self.source_dataset_annotation_form
            json_output_path = os.path.join(
                self.source_dataset_annotations_folder, json_name)
            json.dump(fake_js, open(json_output_path, 'w'))

        return

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
        for n in total_source_dataset_annotations_list:
            source_dataset_annotation = os.path.join(
                self.source_dataset_annotations_folder, n)
            temp_annotation = os.path.join(
                self.temp_annotations_folder, self.file_prefix + n)
            shutil.copy(source_dataset_annotation, temp_annotation)

        # 更新输出统计
        success_count = len(total_source_dataset_annotations_list)

        # 输出读取统计结果
        print('\nSource dataset convert to temp dataset file count: ')
        print('Total annotations:         \t {} '.format(success_count))
        print('Convert fail:              \t {} '.format(fail_count))
        print('No object delete images: \t {} '.format(no_object))
        print('Convert success:           \t {} '.format(success_count))
        self.temp_annotation_name_list = temp_file_name_list
        print('Source dataset annotation transform to temp dataset end.')

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取HY_VAL数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        return []

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成HY_VAL组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')

        return
