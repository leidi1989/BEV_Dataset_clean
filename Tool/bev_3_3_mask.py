'''
Description: 
Version: 
Author: Leidi
Date: 2022-04-10 13:03:51
LastEditors: Leidi
LastEditTime: 2022-04-10 13:29:01
'''
import cv2
import numpy as np
import tqdm


def mask_to_pics(mask):
    """
    生成遮掩自身车辆区域的掩膜
        参数:
            mask:绘制掩膜前图像
        返回:
            mask:绘制掩膜后图像
    """
    pts = np.array(
        [[[0, 0], [35, 18], [87, 556], [517, 720], [0, 720]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array([[[1280, 720], [1052.68, 722.66], [1134.67, 458.74], [
                   1183.35, 394.68], [1280, 371.62]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))
    ###
    pts = np.array([[[1280, 720], [1280, 514.00], [1548.11, 456.17], [
                   2179.34, 485.39], [2377.59, 619.28], [2560, 582.30], [2560, 720]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array(
        [[[2377.59, 619.28], [2560, 582.30], [2560, 720]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array([[[2560, 720], [2563.56, 451.64], [
                   2627.62, 547.72], [2682.78, 720]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array([[[2560, 720], [2563.56, 451.64], [
                   2627.62, 547.72], [2682.78, 720.32]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array([[[0, 2160], [0, 1966.23], [98.08, 2037.40], [
                   110.53, 2112.14], [363.20, 2160]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array([[[3840, 2160], [3388.11, 2160], [3624.77, 2069.43], [
                   3678.15, 1950.21], [3749.32, 1893.27], [3840.00, 1782.07]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array(
        [[[3840, 720], [3350, 720], [3750.75, 553.54], [3780, 0], [3840, 0]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array(
        [[[1280, 1440], [1280, 1568.21], [1383.21, 1440]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    pts = np.array(
        [[[2560, 1440], [2441.43, 1440], [2560, 1586.14]]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    mask = cv2.fillConvexPoly(mask, pts, (0, 0, 0))

    return mask


if __name__ == "__main__":
    import os
    concat_path = "/mnt/data_1/Dataset/Autopilot_bev_dataset/bev_muti_20220408"
    names = os.listdir(concat_path)
    count = 0
    for name in tqdm.tqdm(names):
        count += 1
        concat = cv2.imread(concat_path + "/" + name)
        concat = mask_to_pics(concat)
        cv2.imwrite(
            '/mnt/data_1/Dataset/Autopilot_bev_dataset/bev_muti_20220408_mask/' + name, concat)
