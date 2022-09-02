'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-18 09:34:21
LastEditors: Leidi
LastEditTime: 2021-12-22 11:08:56
'''
import os
import cv2
import argparse
from tqdm import tqdm


def image_blur(image_folder: str, blur_lever: int,  dataset: dict = None) -> None:
    """[图片模糊化]

    Args:
        image_folder (str): [description]
        blur_scale (int): [description]
        dataset (dict, optional): [description]. Defaults to None.
    """
    
    print('Start image blur:')
    for n in tqdm(os.listdir(image_folder)):
        if os.path.splitext(n)[-1] == 'jpg':
            image_path = os.path.join(image_folder, n)
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            # 首轮模糊
            image_blur = cv2.GaussianBlur(
                image, (blur_lever*2+1, blur_lever*2+1), blur_lever*3)
            image_blur = cv2.medianBlur(image_blur, blur_lever*2+1)
            # 分辨率模糊
            image_blur = cv2.resize(image_blur, (int(
                width*(1-0.1*blur_lever)), int(height*(1-0.1*blur_lever))), interpolation=cv2.INTER_LANCZOS4)
            image_blur = cv2.resize(
                image_blur, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
            # 马赛克
            bais = blur_lever
            for m in range(height-bais):
                for n in range(width-bais):
                    if m % bais == 0 and n % bais == 0:
                        for i in range(bais):
                            for j in range(bais):
                                b, g, r = image_blur[m, n]
                                image_blur[m+i, n+j] = (b, g, r)
            # 再模糊
            try:
                image_blur = cv2.GaussianBlur(
                    image_blur, (blur_lever, blur_lever), blur_lever)
            except:
                image_blur = cv2.GaussianBlur(
                    image_blur, (blur_lever+1, blur_lever+1), blur_lever)
            try:
                image_blur = cv2.medianBlur(image_blur, blur_lever)
            except:
                image_blur = cv2.medianBlur(image_blur, blur_lever+1)
            # 保存图片
            cv2.imwrite(image_path, image_blur)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='image_blur.py')
    parser.add_argument('--imagefolder', '--if', dest='imagefolder', default=r'/home/leidi/Dataset/number_test_out',
                        type=str, help='dataset path')
    parser.add_argument('--blurlever', '--b', dest='blurlever', default=5,
                        type=int, help='blue lever, 1-9')
    opt = parser.parse_args()

image_folder = opt.imagefolder
blur_lever = opt.blurlever

image_blur(image_folder, blur_lever)
