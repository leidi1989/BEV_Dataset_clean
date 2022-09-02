'''
Description: 
Version: 
Author: Leidi
Date: 2022-07-12 10:16:34
LastEditors: Leidi
LastEditTime: 2022-07-29 15:56:18
'''
import argparse
import multiprocessing
import os
import shutil

from tqdm import tqdm

from utils.utils import *


def temp_dataset_images_and_annotations_concate(opt):
    """拼接数据集中infomations文件夹

    Args:
        opt (_type_): _description_
    """
    print('Start to concate statistic:')
    # 图片合并
    total_temp_dataset_images_folder = check_output_path(
        os.path.join(opt.datasets_path, 'total_temp_dataset', 'images'))
    # 标注文件合并
    total_temp_dataset_annotations_folder = check_output_path(
        os.path.join(opt.datasets_path, 'total_temp_dataset', 'annotations'))

    # 获取暂存数据集图片及标注文件源路径
    total_temp_dataset_images_path_list = []
    total_temp_dataset_annotations_path_list = []
    for root, dirs, files in tqdm(
            os.walk(opt.datasets_path),
            desc='Get total temp dataset images and annotations'):
        for folder in dirs:
            if dirs == 'total_temp_dataset':
                continue
            one_dataset = os.path.join(root, folder)
            for root_, dirs_, files_ in tqdm(
                    os.walk(one_dataset),
                    desc='Get one temp dataset images and annotations',
                    leave=False):
                for dir_ in dirs_:
                    if dir_ == 'source_dataset_images':
                        one_dataset_images_folder = os.path.join(root_, dir_)
                        for n in tqdm(os.listdir(one_dataset_images_folder),
                                      desc='Get images path',
                                      leave=False):
                            image_path = os.path.join(root_, dir_, n)
                            total_temp_dataset_images_path_list.append(
                                image_path)
                    if dir_ == 'temp_annotations':
                        one_dataset_annotations_folder = os.path.join(
                            root_, dir_)
                        for n in tqdm(
                                os.listdir(one_dataset_annotations_folder),
                                desc='Get annotations path',
                                leave=False):
                            annotation_path = os.path.join(root_, dir_, n)
                            total_temp_dataset_annotations_path_list.append(
                                annotation_path)

        # 拷贝暂存数据集图片及标注文件
        total_temp_dataset_images_and_annotations_path_list = zip(
            total_temp_dataset_images_path_list,
            total_temp_dataset_annotations_path_list)
        total_temp_dataset_images_and_annotations_path_range = [
            m for m, n in enumerate(
                total_temp_dataset_images_and_annotations_path_list)
        ]
        pbar, update = multiprocessing_object_tqdm(
            len(total_temp_dataset_images_and_annotations_path_range),
            'Copy images and annotations')
        pool = multiprocessing.Pool(opt.workers)
        for image_path, annotation_path in zip(
                total_temp_dataset_images_path_list,
                total_temp_dataset_annotations_path_list):
            pool.apply_async(temp_dataset_copy_images_and_annotations,
                             args=(image_path, annotation_path,
                                   total_temp_dataset_images_folder,
                                   total_temp_dataset_annotations_folder),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        return


def temp_dataset_copy_images_and_annotations(
        image_path: str, annotation_path: str,
        total_temp_dataset_images_folder: str,
        total_temp_dataset_annotations_folder: str) -> None:
    """拷贝暂存数据集图片及标注文件

    Args:
        image_path (str): 暂存数据集图片源路径
        annotation_path (str): 暂存数据集标注文件源路径
        total_temp_dataset_images_folder (str): 全部暂存数据集图片文件夹
        total_temp_dataset_annotations_folder (str): 全部暂存数据集标注文件文件夹
    """

    # 拷贝图片
    image_name = image_path.split(os.sep)[-1]
    image_output_path = os.path.join(total_temp_dataset_images_folder,
                                     image_name)
    shutil.copy(image_path, image_output_path)

    # 拷贝标注文件
    annotation_name = annotation_path.split(os.sep)[-1]
    annotation_output_path = os.path.join(
        total_temp_dataset_annotations_folder, annotation_name)
    shutil.copy(annotation_path, annotation_output_path)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='temp_dataset_images_and_annotations_concate.py')
    parser.add_argument(
        '--datasets_path',
        '--dp',
        dest='datasets_path',
        default=
        r'/mnt/data_1/Dataset/dataset_temp/hq1_sync_images_annotations_20220511_20220526_20220729',
        type=str,
        help='Datasets folder path.')
    parser.add_argument(
        '--workers',
        '--w',
        dest='workers',
        default=16,
        type=int,
        help='maximum number of dataloader workers(multiprocessing.cpu_count())'
    )
    opt = parser.parse_args()

    temp_dataset_images_and_annotations_concate(opt)
