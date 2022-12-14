# 是否是NAS数据
Save_in_nas: False
# NAS ip
Nas_ip: "192.168.0.88"
# NAS username
Nas_username: hy
# NAS code
Nas_password: "88888888"
# 是否发送至训练服务器
Scp_to_train_server: False
# 训练服务器ip
Train_server_ip: "192.168.0.88"

# 数据集输入路径
Dataset_input_folder: /mnt/data_1/Dataset/dataset_temp/relabel_raw_data_20220926/wuda_baishazhou_dftc_hq1_6v_000304050607_100_40_40_40_20220607
# 数据集输出路径
Dataset_output_folder: /mnt/data_1/Dataset/dataset_temp/relabel_raw_data_20220926/wuda_baishazhou_dftc_hq1_6v_000304050607_100_40_40_40_20220607_20220926
# 重命名前缀
File_prefix: wuda2baishazhou2dftc
# name delimiter(cityscapes need: "@")
File_prefix_delimiter: "@"

# 输入数据集类型:
# CVAT_IMAGE_BEV_1, CVAT_IMAGE_BEV_2, CVAT_IMAGE_BEV_2_1(sync_data xml_128), CVAT_IMAGE_BEV_3_MAP, CVAT_IMAGE_1_1_MAP, CVAT_IMAGE_BEV_NAS, TEMP_DATASET
# CVAT_COCO2017, YUNCE_SEGMENT_COCO, YUNCE_SEGMENT_COCO_ONE_IMAGE, HUAWEIYUN_SEGMENT, HY_VAL,
# SJT, MYXB, HY_HIGHWAY, GEELY_COCO_ONE_IMAGE
# YOLO, COCO2017, BDD100K, TT100K, CCTSDB, LISA, KITTI, APOLLOSCAPE_LANE_SEGMENT,
# TI_EDGEAILITE_AUTO_ANNOTATION
Source_dataset_style: CVAT_IMAGE_BEV_2_1
# 目标数据集类型：
# NUSCENES, PYVA, CVT, HY_BEV, HDMAPNET
# COCO2017, CITYSCAPES, CITYSCAPES_VAL, CVAT_IMAGE_1_1, YOLO
Target_dataset_style: PYVA
# 标注车型（hq1, hq2, lt1, lt2）
Annotation_car: hq1
# 目标数据集视角（perspective, BEV）:
Target_dataset_view: BEV
# 数据集划分比例
# (train, val, test, redund)
Target_dataset_divide_proportion: 1,0,0,0

# 多annotation.xml输入
Multi_annotation_xml: True
# 是否保存无训练目标图片
Keep_no_object: True
# 检查图片车辆是否添加mask
Draw_car_mask: True
# 是否为二分类（backgound and object）:
Two_class: True

# 是否包含地图
Get_local_map: False
# 是否只提取地图信息
Only_local_map: False
# 是否删除无地图样本
Delete_no_map: False
# lat_lon_origin城市
Lat_lon_origin_city: wuhan
# 地图原点
Lat_lon_origin_point:
  wuhan:
    lat: 30.425457029885372151
    lon: 114.09623523096009023
  shenzhen:
    lat: 22.6860589
    lon: 114.3779897
  changchun':
    lat: 43.8285543
    lon: 125.1564909
  suzhou:
    lat: 31.3227443
    lon: 120.7274488
# laneline曲率限制
Curvature_limit:

# 标注时标注图片与相机图片是否已经拼接在一起
Camera_label_image_concat: True
# 相机图片宽和高
Camera_image_wh: 3200,1200
# 标注图片宽和高
Label_image_wh: 3200,5600
# Label_image_wh: 3200,5600
# 标注范围：前后左右（米）
# Label_range: 100,80,40,40
Label_range: 100,40,40,40
# 调整标注范围：前后左右（米）
Adjust_label_range: 80,40,40,40
# 标注框旋转角度
Label_object_rotation_angle: 90
# 激光雷达相对于车辆后轴中心点偏移量
Lida_to_vehicle_offset_xy:
# 语义分割标注图片宽高
Semantic_segmentation_label_image_wh: 512,512

# 是否仅做统计
Only_statistic: False
# 是否统计标注目标距离
Statistic_label_object_distance: True
# 统计标注范围
Statistic_label_distance: 120
# 统计标注范围间隔
Statistic_label_distance_statistic_segmentation: 3
# 是否统计标注目标角度
Statistic_label_object_angle: True
# 统计标注范围
Statistic_label_angle: 360
# 统计标注范围间隔
Statistic_label_angle_statistic_segmentation: 15

# 检测目标数据集标注文件
Target_dataset_check_annotations_count: 10
# 检测目标数据集输出是否为掩码
Target_dataset_check_annotations_output_as_mask: True

# 是否将语义分割转换为标注框，或标注框转换为语义分割
Need_convert:
# 任务和类别配置（唯一选项）
# Task and class config, select only one.
# Detection, Semantic_segmentation, Instance_segmentation, Keypoints
Task_and_class_config:
  Detection:
    Source_dataset_class_file_path: data/class/bev/hy_bev_60_classes_20220828.name
    Modify_class_file_path: data/class/bev/hy_bev_2_classes_20220828.txt
    Target_each_class_object_pixel_limit_file_path:
  # Semantic_segmentation:
  #   Source_dataset_class_file_path: Clean_up/data/auto_annotation/TI_edgeailite_auto_annotation_yunce_parking_6_classes.names
  #   Modify_class_file_path:
  #   Target_each_class_object_pixel_limit_file_path:
  # Instance_segmentation:
  #   Source_dataset_class_file_path: Clean_up/data/class/instance/instance_classes_bdd100k_11_classes.names
  #   Modify_class_file_path: Clean_up/data/class/instance/instance_classes_bdd100k_11_classes_to_7_classes_line.txt
  #   Target_each_class_object_pixel_limit_file_path:
  # Keypoints:
  #   Source_dataset_class_file_path: Clean_up/data/segmentation/segment_classes_huaweiyun_segmentation_183_classes_20211116.names
  #   Modify_class_file_path: Clean_up/data/segmentation/segment_classes_huaweiyun_segmentation_183_classes_to_3_classes.txt
  #   Target_each_class_object_pixel_limit_file_path:
  # Laneline:
  #   Source_dataset_class_file_path: data/class/laneline/hy_laneline_2_classes_20220919.name
  #   Modify_class_file_path:
  #   Target_each_class_object_pixel_limit_file_path:

# 图像标定内外参数
Camera_calibration_file_path:
  extrinsics:
    CAM_BACK: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/back_extrinsic.yml
    CAM_BACK_LEFT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/left_back_extrinsic.yml
    CAM_BACK_RIGHT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/right_back_extrinsic.yml
    CAM_FRONT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/front_center_extrinsic.yml
    CAM_FRONT_LEFT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/left_front_extrinsic.yml
    CAM_FRONT_RIGHT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/right_front_extrinsic.yml
  intrinsics:
    CAM_BACK: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/back_intrinsics.yml
    CAM_BACK_LEFT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/left_back_intrinsics.yml
    CAM_BACK_RIGHT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/right_back_intrinsics.yml
    CAM_FRONT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/front_center_intrinsics.yml
    CAM_FRONT_LEFT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/left_front_intrinsics.yml
    CAM_FRONT_RIGHT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/right_front_intrinsics.yml

# 是否debug
debug: False
