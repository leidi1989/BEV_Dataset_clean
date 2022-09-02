'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2022-02-25 18:38:56
'''
import argparse
import multiprocessing
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm


def main(config: dict) -> None:
    """按指定数量拼接图片

    Args:
        config (dict): 配置信息
    """

    image_input_folder = Path(config['input_folder'])
    image_output_folder = Path(config['output_folder'])
    if not image_output_folder.exists():
        image_output_folder.mkdir()
    concate_num = config['concate_num']-1 if config['concate_num'] != 0 else 0
    temp_four_image = []
    file_prefix = config['File_prefix']
    file_prefix_delimiter = config['File_prefix_delimiter']
    output_count = 0
    image_count = 0
    for image_path in tqdm(image_input_folder.iterdir(),
                           total=len(image_input_folder.iterdir()),
                           desc='concate image'):
        temp_four_image.append(
            [image_path, cv2.imread(str(image_path.absolute()))])
        image_count += 1
        if len(temp_four_image) == 4:
            concate_image_0 = None
            concate_image_1 = None
            image_output_path = image_output_folder / \
                (file_prefix + file_prefix_delimiter + str(output_count) + '.png')
            for m, image in enumerate(temp_four_image):
                if m in [x for x in range(int((concate_num+1)/2))]:
                    if concate_image_0 is None:
                        height, weight, _ = image[1].shape
                        concate_image_0 = cv2.resize(
                            image[1], dsize=(int(weight/2), int(height/2)))
                    else:
                        height, weight, _ = image[1].shape
                        image_resize_1 = cv2.resize(
                            image[1], dsize=(int(weight/2), int(height/2)))
                        concate_image_0 = cv2.hconcat(
                            (concate_image_0, image_resize_1))
                else:
                    if concate_image_1 is None:
                        height, weight, _ = image[1].shape
                        concate_image_1 = cv2.resize(
                            image[1], dsize=(int(weight/2), int(height/2)))
                    else:
                        height, weight, _ = image[1].shape
                        image_resize_1 = cv2.resize(
                            image[1], dsize=(int(weight/2), int(height/2)))
                        concate_image_1 = cv2.hconcat(
                            (concate_image_1, image_resize_1))
            concate_image_total = cv2.vconcat(
                (concate_image_0, concate_image_1))
            cv2.imwrite(str(image_output_path.absolute()), concate_image_total)
            output_count += 1
            temp_four_image = []

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='image_concat.py')
    parser.add_argument('--config', '--c', dest='config', default=r'Tool/config/image_concat.yaml',
                        type=str, help='dataset config file path')
    parser.add_argument('--workers', '--w', dest='workers', default=multiprocessing.cpu_count(),
                        type=int, help='maximum number of dataloader workers(multiprocessing.cpu_count())')

    opt = parser.parse_args()
    config = yaml.load(
        open(opt.config, 'r', encoding="utf-8"), Loader=yaml.FullLoader)

    main(config)
