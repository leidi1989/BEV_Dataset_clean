#!/usr/bin/python3
import utm
from pyproj import Proj
import argparse
import os
import numpy as np
import open3d as o3d

origin_lat = 30.42440417936125
origin_lon = 114.0993424867674
origin_x = 0.0
origin_y = 0.0
zone_number = 0
zone_letter = ''

mercator = 0


def load_args():
    parser = argparse.ArgumentParser(description='pointcloud_project_trans')
    parser.add_argument('--input_cloud_dir',
                        type=str,
                        help='输入点云文件夹',
                        default='./input')
    parser.add_argument('--output_cloud_dir',
                        type=str,
                        help='输出点云文件夹',
                        default='./output')
    parser.add_argument('--origin_lat',
                        type=float,
                        help='原点纬度',
                        default=30.425457029885372151)
    parser.add_argument('--origin_lon',
                        type=float,
                        help='原点精度',
                        default=114.09623523096009023)
    parser.add_argument('--utm2mercator',
                        type=bool,
                        help='是否由utm向mecator转换, 为false的话就是反向转换',
                        default=True)
    args = parser.parse_args()
    return args


def SetOrigin_Utm(lat: float, lon: float):
    """to set the origin_lat origin_lon origin_x origin_y and zone number"""
    global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
    x, y, num, letter = utm.from_latlon(lat, lon)
    origin_lat = lat
    origin_lon = lon
    origin_x = x
    origin_y = y
    zone_number = num
    zone_letter = letter


def ll2xy_Utm(lat: float, lon: float):
    """latitude and longtitude to x and y"""
    global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
    easting, northing, _, _ = utm.from_latlon(lat, lon, zone_number,
                                              zone_letter)
    x = easting - origin_x
    y = northing - origin_y
    return x, y


def xy2ll_Utm(x: float, y: float):
    """x and y to latitude and longtitude"""
    global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
    easting = x + origin_x
    northing = y + origin_y
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    return lat, lon


def SetOrigin_Mercator(lat: float, lon: float):
    global mercator
    lat_string = str(lat)
    lon_string = str(lon)
    proj_string = '+proj=tmerc +lat_0=' + lat_string + ' +lon_0=' + lon_string + ' +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs'
    mercator = Proj(proj_string, preserve_units=False)


def ll2xy_Mercator(lat: float, lon: float):
    global mercator
    x, y = mercator(lon, lat)
    return x, y


def xy2ll_Mercator(x: float, y: float):
    global mercator
    lon, lat = mercator(x, y, inverse=True)
    return lat, lon


def cloud_utm_2_mercator(args):
    pcd_in_dir = args.input_cloud_dir
    pcd_out_dir = args.output_cloud_dir + '/'
    for folderName, subfolders, filenames in os.walk(pcd_in_dir):
        for filename in filenames:
            file_path = folderName + '/' + filename
            file_out_path = pcd_out_dir + filename
            print('load ' + file_path)
            pcd = o3d.io.read_point_cloud(file_path)
            pcd_points = np.asarray(pcd.points)
            lat, lon = xy2ll_Utm(pcd_points[:, 0], pcd_points[:, 1])
            pcd_points[:, 0], pcd_points[:, 1] = ll2xy_Mercator(lat, lon)
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            o3d.io.write_point_cloud(file_out_path, pcd)
            print('save ' + file_out_path)


def cloud_mercator_2_utm(args):
    pcd_in_dir = args.input_cloud_dir
    pcd_out_dir = args.output_cloud_dir + '/'
    for folderName, subfolders, filenames in os.walk(pcd_in_dir):
        for filename in filenames:
            file_path = folderName + '/' + filename
            file_out_path = pcd_out_dir + filename
            print('load ' + file_path)
            pcd = o3d.io.read_point_cloud(file_path)
            pcd_points = np.asarray(pcd.points)
            lat, lon = xy2ll_Mercator(pcd_points[:, 0], pcd_points[:, 1])
            pcd_points[:, 0], pcd_points[:, 1] = ll2xy_Utm(lat, lon)
            pcd.points = o3d.utility.Vector3dVector(pcd_points)
            o3d.io.write_point_cloud(file_out_path, pcd)
            print('save ' + file_out_path)


def xy2ll_Mercator_map(x: float, y: float, mercator: Proj):
    lon, lat = mercator(x, y, inverse=True)
    return lat, lon


def ll2xy_Utm_map(lat: float, lon: float, origin_utm: dict):
    """latitude and longtitude to x and y"""
    # global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
    easting, northing, _, _ = utm.from_latlon(lat, lon,
                                              origin_utm['zone_number'],
                                              origin_utm['zone_letter'])
    x = easting - origin_utm['origin_x']
    y = northing - origin_utm['origin_y']
    return x, y


def cloud_mercator_2_utm_map(pcd_input_path: str, pcd_output_path: str,
                             mercator: Proj, origin_utm: dict) -> None:
    """将mercator点云转换为utm点云

    Args:
        pcd_input_path (str): 点云输入路径
        pcd_output_path (str): 点云输出路径
    """

    pcd = o3d.io.read_point_cloud(pcd_input_path)
    pcd_points = np.asarray(pcd.points)
    lat, lon = xy2ll_Mercator_map(pcd_points[:, 0], pcd_points[:, 1], mercator)
    pcd_points[:, 0], pcd_points[:, 1] = ll2xy_Utm_map(lat, lon, origin_utm)
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    o3d.io.write_point_cloud(pcd_output_path, pcd)

    return


def main(args):
    # 设置两种转换的坐标原点
    SetOrigin_Utm(args.origin_lat, args.origin_lon)
    SetOrigin_Mercator(args.origin_lat, args.origin_lon)
    if args.utm2mercator:
        # utm坐标到mercator坐标
        cloud_utm_2_mercator(args)
        pass
    else:
        # mercator坐标到utm坐标
        cloud_mercator_2_utm(args)
        pass
    return


if __name__ == "__main__":
    args = load_args()
    main(args)