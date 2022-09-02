<!--
 * @Description: 
 * @Version: 
 * @Author: Leidi
 * @Date: 2022-07-19 14:53:07
 * @LastEditors: Leidi
 * @LastEditTime: 2022-08-22 16:34:26
-->
> Dataset_input_folder: (/path)

输入数据集路径

> Dataset_output_folder: (/path)

输出数据集路径

> Annotation_car: hq1  

标注车型（hq1, hq2, lt1, lt2）

> Draw_car_mask: True

是否添加mask

> Source_dataset_style: (CVAT_IMAGE_BEV or ...)

源数据集输入类型：
[CVAT_IMAGE_BEV_1, CVAT_IMAGE_BEV_2, TEMP_DATASET,
 CVAT_COCO2017, YUNCE_SEGMENT_COCO, YUNCE_SEGMENT_COCO_ONE_IMAGE, HUAWEIYUN_SEGMENT, HY_VAL, SJT, MYXB, HY_HIGHWAY, GEELY_COCO_ONE_IMAGE, YOLO, OCO2017, BDD100K, TT100K, CCTSDB, LISA, KITTI, APOLLOSCAPE_LANE_SEGMENT, TI_EDGEAILITE_AUTO_ANNOTATION]

> Target_dataset_style: (CROSS_VIEW or ...)

输入目标数据集输入类型
[NUSCENES, CROSSVIEW, CROSSVIEWTRANSFORMERS, COCO2017, CITYSCAPES, CITYSCAPES_VAL, CVAT_IMAGE_1_1, YOLO, CROSS_VIEW]

> Target_dataset_view: BEV

目标数据集视角

> Only_statistic: (True or False)

是否只进行统计

> File_prefix: (any str)

图片及标注文件添加的前缀

> File_prefix_delimiter: "@"

图片及标注文件添加的前缀分隔符(cityscapes need: "@")

> Statistic_label_object_distance: (True or False)

是否统计标注目标距离（True or False）

> Statistic_label_distance: 120

统计标注范围，数值

> Statistic_label_distance_statistic_segmentation: 3

统计标注范围间隔，数值

> Two_class: (True or False)

是否为二分类(True or False)


> Camera_image_wh: 6400,2400

相机图片宽和高(w,h)

> Label_image_wh: 6400,6400

标注图片宽和高(w,h)



> Label_range: 80,80,80,80

标注范围：前后左右（米）(80,80,80,80)

> Adjust_label_range: 80,80,80,80

调整标注范围：前后左右（米）

> Label_object_rotation_angle: 90

标注框旋转角度

> Camera_calibration_file_path:  
  extrinsics:  
    CAM_BACK: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/back_extrinsic.yml  
    CAM_BACK_LEFT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/left_back_extrinsic.yml  
    CAM_BACK_RIGHT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/right_back_extrinsic.yml  
    CAM_FRONT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/front_center_extrinsic.yml  
    CAM_FRONT_LEFT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/left_front_extrinsic.  
    CAM_FRONT_RIGHT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/extrinsics/right_front_extrinsic.yml  
  intrinsics:  
    CAM_BACK: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/back_intrinsics.yml  
    CAM_BACK_LEFT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/left_back_intrinsics.yml  
    CAM_BACK_RIGHT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/right_back_intrinsics.yml  
    CAM_FRONT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/front_center_intrinsics.yml  
    CAM_FRONT_LEFT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/left_front_intrinsics.yml  
    CAM_FRONT_RIGHT: /mnt/data_1/Dataset/dataset_temp/hq1_calibration_result_20220607/intrinsics/right_front_intrinsics.yml  

图像标定内外参数yml文件路径

> Semantic_segmentation_label_image_wh: 512,512

语义分割标注图片宽高(w,h)

> Offset:  
  Camera_hz:  
  Delay_time:  

同步时间偏移量（帧）


> Target_dataset_divide_proportion: 0.8,0.1,0.1,0

训练集，验证集，测试集，冗余数据分配比例(train, val, test, redund)

> Target_dataset_check_annotations_count: （100）

绘制输出目标数据集标签检测图片数量

> Target_dataset_check_annotations_output_as_mask: True

是否使用透明掩码绘制输出目标数据集标签检测图片

> Need_convert: （segmentation_to_box or box_to_segmentation）

是否进行标签类型转化，由分割转换为box，或者由box转换为分割（segmentation_to_box, box_to_segmentation）

> Label_object_rotation_angle: 0

标注框旋转角度

> Task_and_class_config:  
   Detection:  
     Source_dataset_class_file_path:(/.name file path)  
     Modify_class_file_path:(/.txt file path)  
     Target_each_class_object_pixel_limit_file_path:(/limit .txt path)  
   Semantic_segmentation:  
    Source_dataset_class_file_path:(/.name file path)  
     Modify_class_file_path:(/.txt file path)  
     Target_each_class_object_pixel_limit_file_path:(/limit .txt path)  
   Instance_segmentation:  
     Source_dataset_class_file_path:(/.name file path)  
     Modify_class_file_path:(/.txt file path)  
     Target_each_class_object_pixel_limit_file_path:(/limit .txt path)  
   Keypoints:  
     Source_dataset_class_file_path:(/path)  
     Modify_class_file_path:(/path)  
     Target_each_class_object_pixel_limit_file_path:(/path)  

任务和任务类别配置，需选择一项填写（Detection, Semantic_segmentation, Instance_segmentation, Keypoints），并给出对应的类别文件路径，类别修改文件路径，类别距离挑选文件路径

> debug: （True or False）

debug模式
