import os
import xml.etree.ElementTree as ET

import pandas as pd

# file path
base_path = '/home/afeng/Backups/bev_6v_hongqi1_wudatohjh_0507_20220609'
file_name = "annotations.xml"

file_path = os.path.join(base_path, file_name)

# Camera image width and height
camera_image_width, camera_image_height = 6400, 2400
# Label image width and height
label_image_width, label_image_height = 6400, 6400
# label range: front back left right
label_range_front, label_range_back, label_range_left, label_range_right = 80, 80, 80, 80

total_range = 120
range_offset = 3  # 分段统计偏移(m)
# statistics_dict = {'label': [str(i) for i in range(0, total_range, 3)]}  # 距离统计
statistics_dict = {}  # 距离统计
"""
1. 计算box中心像素坐标
2. 计本车中心像素坐标
3. 计算宽高像素与实际距离换算关系
4. 计算真实距离
5. 以距离x为分段统计
"""


def distance_pixel_rate(real_distance, pixel_distance) -> float:
    """计算真实距离与像素距离的换算关系(m/pixel)

    Args:
        real_distance (int): 真实距离
        pixel_distance (int): 像素距离

    Returns:
        float: m/pixel
    """
    return real_distance / pixel_distance


def self_position_parse(camera_image_height, label_image_width,
                        label_image_height, front_range, back_range,
                        left_range, right_range) -> tuple:
    """计算自车中心像素坐标(x,y)，以真实前距、左距比例计算

    Args:
        camera_image_height (int): 相机图像高度
        label_image_width (int): label图像宽度
        label_image_height (int): label图像高度
        front_range (int): 车辆前方距离
        back_range (int): 车辆后方距离
        left_range (int): 车辆左侧距离
        right_range (int): 车辆右侧距离

    Returns:
        tuple: (x,y)
    """
    # 计算本车中心点像素坐标(图片由上camera+下label组成)
    self_ycenter = label_image_height * \
        (front_range/(front_range+back_range)) + camera_image_height
    self_xcenter = label_image_width * (left_range /
                                        (left_range + right_range))

    return (self_xcenter, self_ycenter)


def calculate_distance(a_x: float, a_y: float, b_x: float,
                       b_y: float) -> float:
    """
    计算两点之间的距离
    """
    return ((a_x - b_x)**2 + (a_y - b_y)**2)**0.5


def image_annotation_parse(file_path: str):
    # 自车中心像素坐标
    self_center = self_position_parse(camera_image_height, label_image_width,
                                      label_image_height, label_range_front,
                                      label_range_back, label_range_left,
                                      label_range_right)
    # 距离像素换算关系 m/pixel
    dp_rate = distance_pixel_rate(label_range_left + label_range_right,
                                  label_image_width)

    # 解析xml文件
    root = ET.parse(file_path).getroot()
    for xml_element in root:
        # print(isinstance(xml_element, ET.Element))
        # print(xml_element.tag)

        if xml_element.tag != 'image':  # 只处理image标签
            continue
        # print(xml_element.attrib['width'], xml_element.attrib['height'])

        for element in xml_element:
            if element.tag != 'box':  # 只处理box标签
                continue
            # print(element.attrib['label'])
            label = element.attrib['label']
            # 计算box中心点像素坐标
            box_xcenter = (float(element.attrib['xtl']) +
                           float(element.attrib['xbr'])) / 2
            box_ycenter = (float(element.attrib['ytl']) +
                           float(element.attrib['ybr'])) / 2
            # print(box_xcenter, box_ycenter)
            # 计算box中心点与自车中心点的距离
            pixel_distance = calculate_distance(box_xcenter, box_ycenter,
                                                self_center[0], self_center[1])
            real_distance = pixel_distance * dp_rate
            # print(real_distance)
            if label not in statistics_dict.keys():
                statistics_dict[label] = [0] * (
                    (total_range + 1) // range_offset)
            statistics_dict[label][int(real_distance // range_offset)] += 1

    # 生成CSV文件
    statistics = pd.DataFrame(
        statistics_dict.values(),
        index=statistics_dict.keys(),
        columns=[f'{i}~{i+range_offset}' for i in range(0, total_range, 3)])

    statistics.loc['Summary'] = statistics.apply(
        lambda x: x.sum())  # 各列求和，添加新的行.

    print(statistics)
    statistics.to_csv('statistics.csv')


if __name__ == '__main__':
    image_annotation_parse(file_path)
