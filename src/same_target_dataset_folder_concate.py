'''
Description: 
Version: 
Author: Leidi
Date: 2022-07-12 10:16:34
LastEditors: Leidi
LastEditTime: 2022-08-31 17:28:13
'''
import argparse
import copy
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.utils import *


def statistic_concate(opt):
    """拼接数据集中infomations文件夹

    Args:
        opt (_type_): _description_
    """
    print('Start to concate statistic:')
    # 按类别进行统计合并
    total_sample_statistics_folder = check_output_path(
        os.path.join(opt.datasets_path, 'total_infomations',
                     'sample_statistics'))
    # 按距离进行统计合并
    total_object_distance_statistics_folder = check_output_path(
        os.path.join(total_sample_statistics_folder,
                     'object_distance_statistics'))
    detection_object_distance_count_pdframe_list_dict = {
        'train': [],
        'test': [],
        'val': [],
        'redund': [],
        'total': []
    }
    total_detection_object_distance_count_pdframe_dict = {}
    total_detection_object_distance_count_proportion_pdframe_dict = {}

    # 按角度进行统计合并
    total_object_angle_statistics_folder = check_output_path(
        os.path.join(total_sample_statistics_folder,
                     'object_angle_statistics'))
    detection_object_angle_count_pdframe_list_dict = {
        'train': [],
        'test': [],
        'val': [],
        'redund': [],
        'total': []
    }
    total_detection_object_angle_count_pdframe_dict = {}
    total_detection_object_angle_count_proportion_pdframe_dict = {}

    for root, dirs, files in tqdm(os.walk(opt.datasets_path),
                                  desc='Get total cvs files data'):
        for folder in dirs:
            sample_statistics_folder_path = os.path.join(
                root, folder, 'temp_infomations', 'sample_statistics')
            if not os.path.exists(sample_statistics_folder_path):
                continue
            # 按距离进行统计合并
            object_distance_statistics_folder_path = os.path.join(
                sample_statistics_folder_path, 'object_distance_statistics')
            for csv in os.listdir(object_distance_statistics_folder_path):
                if csv.split('.')[-1] != 'csv':
                    continue
                csv_path = os.path.join(object_distance_statistics_folder_path,
                                        csv)
                data = pd.read_csv(csv_path)
                dataset_tpye = csv.split('.')[0].split('_')[-1]
                if 'proportion' in csv.split('_'):
                    continue
                else:
                    detection_object_distance_count_pdframe_list_dict[
                        dataset_tpye].append(data)
            # 按角度进行统计合并
            object_angle_statistics_folder_path = os.path.join(
                sample_statistics_folder_path, 'object_angle_statistics')
            for csv in os.listdir(object_angle_statistics_folder_path):
                if csv.split('.')[-1] != 'csv':
                    continue
                csv_path = os.path.join(object_angle_statistics_folder_path,
                                        csv)
                data = pd.read_csv(csv_path)
                dataset_tpye = csv.split('.')[0].split('_')[-1]
                if 'proportion' in csv.split('_'):
                    continue
                else:
                    detection_object_angle_count_pdframe_list_dict[
                        dataset_tpye].append(data)
    # 按距离进行统计合并
    for dataset_type, detection_object_distance_count_pdframe_list in tqdm(
            detection_object_distance_count_pdframe_list_dict.items(),
            desc=''):
        data_pdframe = None
        for detection_object_distance_count_pdframe in detection_object_distance_count_pdframe_list:
            if data_pdframe is None:
                data_pdframe = detection_object_distance_count_pdframe
            else:
                data_pdframe.iloc[:,
                                  1:] += detection_object_distance_count_pdframe.iloc[:,
                                                                                      1:]
        total_detection_object_distance_count_pdframe_dict.update(
            {dataset_type: data_pdframe})

    # 按角度进行统计合并
    for dataset_type, detection_object_angle_count_pdframe_list in tqdm(
            detection_object_angle_count_pdframe_list_dict.items(), desc=''):
        data_pdframe = None
        for detection_object_angle_count_pdframe in detection_object_angle_count_pdframe_list:
            if data_pdframe is None:
                data_pdframe = detection_object_angle_count_pdframe
            else:
                data_pdframe.iloc[:,
                                  1:] += detection_object_angle_count_pdframe.iloc[:,
                                                                                   1:]
        total_detection_object_angle_count_pdframe_dict.update(
            {dataset_type: data_pdframe})

    # 类别距离计数占比统计
    total_detection_object_distance_count_proportion_pdframe_dict = copy.deepcopy(
        total_detection_object_distance_count_pdframe_dict)
    for divide_dataset_name, object_distance_count in total_detection_object_distance_count_pdframe_dict.items(
    ):
        for index, each_class_distance_count in enumerate(
                object_distance_count.itertuples()):
            each_class_distance_count_array = np.array(
                each_class_distance_count[2:])
            total_each_class_distance_count = each_class_distance_count_array.sum(
            )
            if total_each_class_distance_count:
                each_class_distance_count_array_proportion = each_class_distance_count_array / total_each_class_distance_count
            else:
                each_class_distance_count_array_proportion = np.zeros_like(
                    each_class_distance_count_array)
            total_detection_object_distance_count_proportion_pdframe_dict[
                divide_dataset_name].iloc[
                    index, 1:] = each_class_distance_count_array_proportion
    # 记录目标距离计数分布
    for divide_file_name, object_distance_count_dataframe in total_detection_object_distance_count_pdframe_dict.items(
    ):
        object_distance_count_dataframe.to_csv(
            os.path.join(
                total_object_distance_statistics_folder,
                'Detection_object_distance_count_{}.csv'.format(
                    divide_file_name)))
        # 绘图
        # object_distance_count_dataframe.plot(kind='bar')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig((os.path.join(
        #     self.temp_sample_objec_distance_statistics_folder,
        #     'Detection_object_distance_count_{}.png'.format(
        #         divide_file_name))))
    for divide_file_name, object_distance_count_proportion_dataframe in total_detection_object_distance_count_proportion_pdframe_dict.items(
    ):
        object_distance_count_proportion_dataframe.to_csv(
            os.path.join(
                total_object_distance_statistics_folder,
                'Detection_object_distance_count_proportion_{}.csv'.format(
                    divide_file_name)))
        # 绘图
        # object_distance_count_proportion_dataframe.plot()
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig((os.path.join(
        #     self.temp_sample_objec_distance_statistics_folder,
        #     'Detection_object_distance_count_proportion_{}.png'.format(
        #         divide_file_name))))

    # 类别角度计数占比统计
    total_detection_object_angle_count_proportion_pdframe_dict = copy.deepcopy(
        total_detection_object_angle_count_pdframe_dict)
    for divide_dataset_name, object_angle_count in total_detection_object_angle_count_pdframe_dict.items(
    ):
        for index, each_class_angle_count in enumerate(
                object_angle_count.itertuples()):
            each_class_angle_count_array = np.array(each_class_angle_count[2:])
            total_each_class_angle_count = each_class_angle_count_array.sum()
            if total_each_class_angle_count:
                each_class_angle_count_array_proportion = each_class_angle_count_array / total_each_class_angle_count
            else:
                each_class_angle_count_array_proportion = np.zeros_like(
                    each_class_angle_count_array)
            total_detection_object_angle_count_proportion_pdframe_dict[
                divide_dataset_name].iloc[
                    index, 1:] = each_class_angle_count_array_proportion
    # 记录目标距离计数分布
    for divide_file_name, object_angle_count_dataframe in total_detection_object_angle_count_pdframe_dict.items(
    ):
        object_angle_count_dataframe.to_csv(
            os.path.join(
                total_object_angle_statistics_folder,
                'Detection_object_angle_count_{}.csv'.format(
                    divide_file_name)))
        # 绘图
        # object_angle_count_dataframe.plot(kind='bar')
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig((os.path.join(
        #     self.temp_sample_objec_angle_statistics_folder,
        #     'Detection_object_angle_count_{}.png'.format(
        #         divide_file_name))))
    for divide_file_name, object_angle_count_proportion_dataframe in total_detection_object_angle_count_proportion_pdframe_dict.items(
    ):
        object_angle_count_proportion_dataframe.to_csv(
            os.path.join(
                total_object_angle_statistics_folder,
                'Detection_object_angle_count_proportion_{}.csv'.format(
                    divide_file_name)))
        # 绘图
        # object_angle_count_proportion_dataframe.plot()
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig((os.path.join(
        #     self.temp_sample_objec_angle_statistics_folder,
        #     'Detection_object_angle_count_proportion_{}.png'.format(
        #         divide_file_name))))

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='same_target_dataset_folder_concate.py')
    parser.add_argument(
        '--datasets_path',
        '--dp',
        dest='datasets_path',
        default=
        r'/mnt/data_2/Dataset/Autopilot_bev_dataset/hy_hq1_bev_source_total_20220724/20220512_20220615_data_statistic_total_xml_20220711/clean_dataset_20220725',
        type=str,
        help='Datasets folder path.')
    parser.add_argument('--distance',
                        '--dc',
                        dest='distance',
                        default=True,
                        type=bool,
                        help='Datasets distance statistc.')
    parser.add_argument('--angle',
                        '--ag',
                        dest='angle',
                        default=True,
                        type=bool,
                        help='Datasets angle statistc.')

    opt = parser.parse_args()

    statistic_concate(opt)
