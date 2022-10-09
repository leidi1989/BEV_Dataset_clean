'''
Description: 
Version: 
Author: Leidi
Date: 2022-09-30 15:31:58
LastEditors: Leidi
LastEditTime: 2022-10-09 16:20:08
'''
#!/usr/bin/python3
# 自适应点云的坐标极值，生成对应大小的俯瞰图
import json
import open3d as o3d
import numpy as np
import cv2
import argparse
import os


def load_args():
    parser = argparse.ArgumentParser(description='pcd->pic')
    parser.add_argument('--meter_per_pixel',
                        type=float,
                        default=0.05,
                        help='每个像素对应距离,单位m')
    parser.add_argument('--pcd_dir', type=str, default='/mnt/data_1/Dataset/dataset_temp/changchun/first', help='pcd 存储文件夹路径')
    parser.add_argument('--pic_dir', type=str, default='/mnt/data_1/Dataset/dataset_temp/changchun/firse_image/', help='pic 存储文件夹路径')
    parser.add_argument('--location_path',
                        type=str,
                        default='/mnt/data_1/Dataset/dataset_temp/changchun/firse_image_pose/firse_image_pose.json',
                        help='定位点存储文件路径')
    args = parser.parse_args()
    return args


def load_pcds(args):
    meter_per_pixel = args.meter_per_pixel  #
    pixel_per_meter = 1 / meter_per_pixel
    pcd_dir = args.pcd_dir  # 点云存储的文件夹路径
    pic_dir = args.pic_dir  # 图片存储文件夹
    location_path = args.location_path  # 定位文件
    location_file = open(location_path, 'w+')
    location_file.close()
    scale = {}
    scale['meter_per_pixel'] = meter_per_pixel
    all_locations = []
    all_locations.append(scale)
    count = 0

    for folderName, subfolders, filenames in os.walk(pcd_dir):
        names = sorted(filenames)
        for filename in names:
            file_path = folderName + '/' + filename
            pic_name = str(count)
            pic_path = pic_dir + pic_name + '.png'
            count = count + 1
            # pic_path = pic_dir+filename.replace('pcd', 'png')
            print("load file :  " + file_path)
            print("save file :  " + pic_path)
            print('count : ' + pic_name)
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            max_x = np.max(points[:, 0])
            min_x = np.min(points[:, 0])
            max_y = np.max(points[:, 1])
            min_y = np.min(points[:, 1])
            location = {}
            location['name'] = pic_name
            location['min_x'] = min_x
            location['max_x'] = max_x
            location['min_y'] = min_y
            location['max_y'] = max_y
            all_locations.append(location)
            colors = np.asarray(pcd.colors) * 255
            points = np.column_stack((points, colors))
            pic_width = int((max_x - min_x) / meter_per_pixel)
            pic_height = int((max_y - min_y) / meter_per_pixel)
            print("pic_width: " + str(pic_width))
            print("pic_height: " + str(pic_height))
            # points_size=len(points)
            # print("points size :  "+str(points_size))
            points[:, 0] = np.floor((points[:, 0] - min_x) * pixel_per_meter)
            points[:, 1] = np.floor((max_y - points[:, 1]) * pixel_per_meter)
            mask_w = (points[:, 0] >= 0) & (points[:, 0] < pic_width)
            mask_h = (points[:, 1] >= 0) & (points[:, 1] < pic_height)
            mask = mask_w & mask_h
            points = points[mask]
            bev_pointcloud_image_np = np.zeros([pic_height, pic_width, 3],
                                               np.uint8)
            u = points[:, 0].astype(np.int)
            v = points[:, 1].astype(np.int)
            cloud_color = points[:, 3:6].astype(np.uint8)
            bev_pointcloud_image_np[v, u] = cloud_color
            bev_pointcloud_image_np = bev_pointcloud_image_np[:, :, ::-1]
            cv2.imwrite(pic_path, bev_pointcloud_image_np)  # 保存；
            print(pic_path + "    gen suceessfully ")
        location_file = open(location_path, 'w+')
        json.dump(all_locations, location_file)
        location_file.close()
        print('all pics gen ok!')
    return


def main(args):
    load_pcds(args)


if __name__ == "__main__":
    args = load_args()
    main(args)
