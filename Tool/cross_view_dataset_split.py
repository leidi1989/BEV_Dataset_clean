'''
Description:
Version:
Author: Leidi
Date: 2022-05-04 20:07:30
LastEditors: Leidi
LastEditTime: 2022-05-11 18:35:11
'''
import argparse
import os
import random
import math

from tqdm import tqdm


def main(opt) -> None:
    """切分cross view数据集,生成train,test,val

    Args:
        opt (_type_): 参数列表
    """
    image_path = os.path.join(opt.data_path, 'input')
    image_name_list = os.listdir(image_path)
    image_name_list_count = len(image_name_list)
    dataset_divide_file_dict = {'train': {'file': open(os.path.join(opt.output_path, 'train_files.txt'), 'w'),
                                          'image_count': 0,
                                          'divide_proportion': opt.proportion[0],
                                          'image_name_list': []},
                                'test': {'file': open(os.path.join(opt.output_path, 'test_files.txt'), 'w'),
                                         'image_count': 0,
                                         'divide_proportion': opt.proportion[1],
                                         'image_name_list': []},
                                'val': {'file': open(os.path.join(opt.output_path, 'val_files.txt'), 'w'),
                                        'image_count': 0,
                                        'divide_proportion': opt.proportion[2],
                                        'image_name_list': []},
                                'redund': {'file': open(os.path.join(opt.output_path, 'redund_files.txt'), 'w'),
                                           'image_count': 0,
                                           'divide_proportion': opt.proportion[3],
                                           'image_name_list': []}, }

    # 按文件名前缀进行数据分类
    file_name_prefix_dict = {}
    for n, image_name in tqdm(enumerate(image_name_list), desc='Get file name'):
        file_name_prefix = image_name.split('_')[0]
        if file_name_prefix not in file_name_prefix_dict:
            file_name_prefix_dict.update(
                {file_name_prefix: [image_name_list[n]]})
        else:
            file_name_prefix_dict[file_name_prefix].append(image_name_list[n])

    # 分配数据至不同set
    for file_name_list in tqdm(file_name_prefix_dict.values()):
        file_name_list_count = len(file_name_list)
        # 除训练集向下取整,获取不同数据集图片数量
        temp_dataset_split_count = {'train': 0,
                                    'test': 0,
                                    'val': 0,
                                    'redund': 0}
        split_image_count = 0
        for key, value in dataset_divide_file_dict.items():
            temp_dataset_split_count[key] = math.floor(
                file_name_list_count * value['divide_proportion'])
            split_image_count += temp_dataset_split_count[key]
        if split_image_count < file_name_list_count:
            temp_dataset_split_count['train'] += file_name_list_count - \
                split_image_count

        temp_name_list = file_name_list.copy()
        file_name_list_count = len(file_name_list)
        for n, [set_name, set_file_list] in enumerate(dataset_divide_file_dict.items()):
            count = 0
            for image_name in file_name_list:
                if count < temp_dataset_split_count[set_name]:
                    set_file_list['image_name_list'].append(
                        temp_name_list.pop())
                    count += 1

    for n in dataset_divide_file_dict.values():
        random.shuffle(n['image_name_list'])
        for image_name in n['image_name_list']:
            if image_name is not n['image_name_list'][-1]:
                n['file'].write(os.path.join(image_path, image_name) + '\n')
            else:
                n['file'].write(os.path.join(image_path, image_name) + '\r')
        n['file'].close()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='clean.py')
    parser.add_argument("--data_path", type=str,
                        default='/mnt/data_2/Dataset/Autopilot_bev_dataset/hy_bev_6v_hq2_wd_hsgc_40_40_15_15_20220420_cross_view',
                        help="path to folder of raw images")
    parser.add_argument("--dataset_divide_proportion", dest='proportion', type=list, default=[0.75, 0.1, 0.15, 0],
                        help="ratio of train, test, val and redund data")
    parser.add_argument("--output_path", type=str, default='/mnt/data_2/Dataset/Autopilot_bev_dataset/hy_bev_6v_hq2_wd_hsgc_40_40_15_15_20220420_cross_view',
                        help="path to folder of raw images")

    opt = parser.parse_args()

    main(opt)
