"""
Description: 
Version: 
Author: Leidi
Date: 2021-06-09 14:54:44
LastEditors: Leidi
LastEditTime: 2021-12-22 11:09:48
"""
# 导入需要的库
import os
import cv2
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np


def avi2img(avi_path: str, img_path: str, image_time: int) -> None:
    """[将avi按指定截取时间间隔截取为jpg]

    Args:
        avi_path (str): [视频路径]
        img_path (str): [图片保存路径]
        image_time (int): [截取图片时间间隔]
    """

    cap = cv2.VideoCapture(avi_path)  # 读入一个视频，打开cap
    fps = cap.get(cv2.CAP_PROP_FPS)  # 获取视频的帧频
    pickup_image_fps = fps * image_time  # 依据时间确定截取的帧间隔
    total_fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    i, x = 0, 0
    # 进入循环
    while(cap.isOpened):
        # 循环达到帧总数时退出
        if i == total_fps:
            break
        (flag, frame) = cap.read()  # 读取每一帧，一张图像flag 表明是否读取成果frame内容
        if flag == True and 0 == i % pickup_image_fps:  # flag表示是否成功读图
            fileName = img_path + profix + '_image' + \
                str(x).zfill(6) + '.jpg'  # 要保存图片的名称
            cv2.imwrite(fileName, frame, [
                        cv2.IMWRITE_JPEG_QUALITY, 100])  # 写入图片，cv2.IMWRITE_JPEG_QUALITY控制质量
            print(fileName)
            x += 1
        i += 1

    return


def img2mp4(image_folder_path: str, mp4fps: int) -> None:
    """[将图片拼接成mp4]

    Args:
        image_folder_path (str): [输入图片文件夹路径]
        mp4fps (int): [mp4帧率]
    """

    image_list = sorted(os.listdir(image_folder_path))
    first_image_path = os.path.join(image_folder_path, image_list[0])
    root = image_folder_path.split(os.sep)[1:]
    output_folder = '/'+(os.sep).join(root[:-1])
    mp4_name = root[-1]+'.mp4'
    # mp4_name = root[-1]+'.avi'
    mp4_file_path = os.path.join(output_folder, mp4_name)

    if os.path.splitext(first_image_path)[-1] == '.jpg':
        img = cv2.imread(first_image_path)  # 读取第一张图片
        imgInfo = img.shape
        size = (imgInfo[1], imgInfo[0])  # 获取图片宽高度信息
    if os.path.splitext(first_image_path)[-1] == '.png':
        img = Image.open(first_image_path)
        size = (img.width, img.height)
        img = np.array(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # 视频写入编码器
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    videoWrite = cv2.VideoWriter(mp4_file_path, fourcc, mp4fps, size)
    # 根据图片的大小，创建写入对象(文件名，支持的编码器，帧频，视频大小(图片大小))
    for name in tqdm(image_list,
                     desc='Concate image'):
        fileName = os.path.join(image_folder_path, name)  # 读取所有图片的路径
        img = cv2.imread(fileName)  # 写入图片
        videoWrite.write(img)  # 将图片写入所创建的视频对象
    videoWrite.release()  # 释放内存，非常重要！！！

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='tool_image_mp4_exchange.py')
    parser.add_argument('--avipath', default=r'',
                        type=str, help='avi path')
    parser.add_argument('--input_folder', default=r'/mnt/data_1/hy_program/cross_view_multi_gpu_classes/output/dynamic/image4',
                        type=str, help='image output path')
    parser.add_argument('--pref', default=r'',
                        type=str, help='rename prefix')
    parser.add_argument('--mode', default='',
                        type=str, help='image output')
    parser.add_argument('--time', default=1,
                        type=int, help='the time of create image, secend')
    parser.add_argument('--mp4fps', default=5,
                        type=int, help='the fps of concate images.')
    opt = parser.parse_args()

    avi_path = opt.avipath
    img_folder_path = opt.input_folder
    profix = opt.pref
    image_time = opt.time
    mp4fps = opt.mp4fps

    if 'image' == opt.mode:
        avi2img(avi_path, img_folder_path, image_time)
    else:
        img2mp4(img_folder_path, mp4fps)
