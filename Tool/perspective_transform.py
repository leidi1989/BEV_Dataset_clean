'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-18 14:26:58
LastEditors: Leidi
LastEditTime: 2021-12-21 16:13:12
'''
import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm


def check_output_path(path: str, attach: str = '') -> str:
    """[检查输出路径是否存在]

    Args:
        path (str): [输出路径]
        attach (str, optional): [输出路径后缀]. Defaults to ''.

    Returns:
        str: [添加后缀的输出路径]
    """

    if os.path.exists(os.path.join(path, attach)):
        print(os.path.join(path, attach))
        return os.path.join(path, attach)
    else:
        print(os.path.join(path, attach))
        os.makedirs(os.path.join(path, attach))
        return os.path.join(path, attach)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='perspective_transform.py')
    parser.add_argument('--imageinfolder', '--if', dest='imageinfolder',
                        default=r'/home/leidi/Dataset/number_test',
                        type=str, help='dataset path')
    parser.add_argument('--imageoutfolder', '--of', dest='imageoutfolder',
                        default=r'/home/leidi/Dataset/number_test_out',
                        type=str, help='dataset path')
    parser.add_argument('--toshape', '--t', dest='toshape', default=(1080, 480),
                        type=tuple, help='target image shape')
    parser.add_argument('--offset', '--o', dest='offset', default=[250, 150],
                        type=list, help='source point offset')
    parser.add_argument('--baiescale', '--b', dest='baiescale', default=50,
                        type=int, help='baie scale value')
    opt = parser.parse_args()

image_in_folder = opt.imageinfolder
image_out_folder = check_output_path(opt.imageoutfolder)
dst_space = opt.toshape
offset = opt.offset
baie_scale = opt.baiescale

print('Start transform image:')
for x in tqdm(os.listdir(image_in_folder)):
    image_path = os.path.join(image_in_folder, x)
    image_out_path = os.path.join(image_out_folder, x)
    image = cv2.imread(image_path)
    rows, cols = image.shape[:2]

    for n in range(10):
        baies = []
        for m in range(8):
            b = random.random()
            baies.append(int(b * baie_scale))
        pts1 = np.float32([[0, 0], [0, 440], [140, 0], [140, 440]])
        pts2 = np.float32([[offset[0] + 0 + baies[0], offset[1] + 0 + baies[1]],
                           [offset[0] + 0 + baies[2], offset[1] + 440 + baies[3]],
                           [offset[0] + 140 + baies[4], offset[1] + 0 + baies[5]],
                           [offset[0] + 140 + baies[6], offset[1] + 440 + baies[7]]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        dst = cv2.warpPerspective(image, M, dst_space, 1)
        image_output_path = image_out_path.split(
            '.')[0] + str(n) + '.' + os.path.splitext(image_out_path)[-1]
        cv2.imwrite(image_output_path, dst)
