#!/usr/bin/python3
from cmath import pi
import xml.etree.ElementTree as ET
from xml.dom import minidom
import argparse
import re
import utm
import math

id_count = 0
origin_lat = 0.0
origin_lon = 0.0
origin_x = 0.0
origin_y = 0.0
zone_number = 0
zone_letter = ''
angle_diff_tolerance = 0.0
distance_tolerance = 0.0


def distance(p1, p2):
    return math.sqrt(math.pow(p1[0] - p2[0], 2) + math.pow(p1[1] - p2[1], 2))


def cal_angle(start, end) -> float:
    delta_x = end[0]-start[0]
    delta_y = end[1]-start[1]
    return math.atan2(delta_y, delta_x)


def cal_angle_diff(last, cur, next) -> float:
    """返回角度差"""
    a1 = cal_angle(last, cur)
    a2 = cal_angle(last, next)
    return math.fabs(a1-a2)*180.0/math.pi


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
    easting, northing, _, _ = utm.from_latlon(
        lat, lon, zone_number, zone_letter)
    x = easting - origin_x
    y = northing - origin_y
    return x, y


def xy2ll_Utm(x: float, y: float):
    """x and y to latitude and longtitude"""
    global origin_lat, origin_lon, origin_x, origin_y, zone_number, zone_letter
    easting = x+origin_x
    northing = y + origin_y
    lat, lon = utm.to_latlon(easting, northing, zone_number, zone_letter)
    return lat, lon


def get_id() -> str:
    global id_count
    id_count += 1
    return str(id_count)


def load_args():
    parser = argparse.ArgumentParser(description='osm map concate')
    parser.add_argument('--input_file', type=str,
                        help='输入osm文件', default='./dfgj.osm')
    parser.add_argument('--out_file', type=str,
                        help='输出合并的地图文件', default='./out.osm')
    parser.add_argument("--enable_format", type=bool,
                        help='是否格式化输出文档', default=False)
    parser.add_argument("--angle_diff_tolerance", type=float,
                        help='认为是直线的角度限差，单位角度', default=1.0)
    parser.add_argument("--distance_tolerance", type=float,
                        help='强制打断的距离限差，单位m', default=10)
    args = parser.parse_args()
    return args


def parse_osm(file_path, out_path):
    global angle_diff_tolerance, distance_tolerance
    temp_tree = ET.ElementTree(file=file_path)
    temp_root = temp_tree.getroot()

    xml_nodes = temp_root.findall('node')
    """取第一个点作为坐标原点"""
    node_origin = xml_nodes[0]
    origin_lat_cur = float(node_origin.get('lat'))
    origin_lon_cur = float(node_origin.get('lon'))
    SetOrigin_Utm(origin_lat_cur, origin_lon_cur)

    """读取node并保存为以ID为键的字典"""
    nodedic = {}
    node_utm_dic = {}
    for node in xml_nodes:
        if node.get('action') == 'delete':
            temp_root.remove(node)
            continue
        nodedic[node.get('id')] = node
        z = 0
        lat = float(node.get('lat'))
        lon = float(node.get('lon'))
        x, y = ll2xy_Utm(lat, lon)
        utm_node = [x, y]
        node_utm_dic[node.get('id')] = utm_node

    xml_ways = temp_root.findall('way')
    """读取way并保存为以ID为键的字典"""
    waydic = {}
    for way in xml_ways:
        if way.get('action') == 'delete':
            temp_root.remove(way)
            continue
        nodes_ref = way.findall('nd')
        start_index = 0
        end_index = len(nodes_ref)-1
        if(end_index < 2):
            continue
        last_node = None
        for i in range(0, end_index):
            cur_node_ref = nodes_ref[i]
            cur_node_id = cur_node_ref.get('ref')
            cur_node = node_utm_dic[cur_node_id]
            if i == 0:
                last_node = cur_node
            else:
                next_node_ref = nodes_ref[i+1]
                next_node_id = next_node_ref.get('ref')
                next_node = node_utm_dic[next_node_id]
                angle_diff = cal_angle_diff(last_node, cur_node, next_node)
                if(angle_diff > angle_diff_tolerance) or (distance(last_node, cur_node) >= distance_tolerance):
                    # 不删除本节点
                    last_node = cur_node
                else:
                    # 删除本节点
                    way.remove(cur_node_ref)
                    temp_root.remove(nodedic[cur_node_id])
    temp_tree.write(out_path, encoding='utf-8', xml_declaration=True)


def main(args):
    global angle_diff_tolerance,distance_tolerance
    angle_diff_tolerance=args.angle_diff_tolerance
    distance_tolerance=args.distance_tolerance
    parse_osm(args.input_file, args.out_file)


if __name__ == "__main__":
    args = load_args()
    main(args)
