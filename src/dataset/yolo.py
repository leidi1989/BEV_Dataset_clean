'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-23 10:29:34
'''
import shutil
import multiprocessing

import dataset
from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base
from utils.convertion_function import yolo, revers_yolo


class YOLO(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'txt'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    def load_image_annotation(self, source_annotation_name: str, process_output: dict) -> None:
        """[读取单个图片标注信息]

        Args:
            source_annotation_name (str): [图片标注信息文件名称]
            process_output (dict): [多进程共享字典]
        """

        source_annotation_path = os.path.join(
            self.source_dataset_annotations_folder,
            source_annotation_name)
        with open(source_annotation_path, 'r') as f:
            image_name = os.path.splitext(source_annotation_name)[
                0] + '.' + self.target_dataset_image_form
            image_name_new = self.file_prefix + image_name
            image_path = os.path.join(
                self.temp_images_folder, image_name_new)
            img = cv2.imread(image_path)
            if img is None:
                print('Can not load: {}'.format(image_name_new))
                return
            size = img.shape
            width = int(size[1])
            height = int(size[0])
            channels = int(size[2])
            object_list = []
            for n, one_bbox in enumerate(f.read().splitlines()):
                true_box = one_bbox.split(' ')[1:]
                clss = self.task_dict['Detection']['Target_dataset_class'][int(
                    one_bbox.split(' ')[0])]
                clss = clss.strip(' ').lower()
                if clss not in self.total_task_source_class_list:
                    continue
                true_box = revers_yolo(size, true_box)
                xmin = min(
                    max(min(float(true_box[0]), float(true_box[1])), 0.), float(width))
                ymin = min(
                    max(min(float(true_box[2]), float(true_box[3])), 0.), float(height))
                xmax = max(
                    min(max(float(true_box[1]), float(true_box[0])), float(width)), 0.)
                ymax = max(
                    min(max(float(true_box[3]), float(true_box[2])), float(height)), 0.)
                box_xywh = [xmin, ymin, xmax-xmin, ymax-ymin]
                object_list.append(OBJECT(n,
                                          clss,
                                          box_clss=clss,
                                          box_xywh=box_xywh,
                                          need_convert=self.need_convert))  # 将单个真实框加入单张图片真实框列表
            image = IMAGE(image_name, image_name, image_path, int(
                height), int(width), int(channels), object_list)

            temp_annotation_output_path = os.path.join(
                self.temp_annotations_folder,
                image.file_name_new + '.' + self.temp_annotation_form)
            image.object_class_modify(self)
            image.object_pixel_limit(self)
            if 0 == len(image.object_list) and not self.keep_no_object:
                print('{} no object, has been delete.'.format(
                    image.image_name_new))
                os.remove(image.image_path)
                process_output['no_object'] += 1
                process_output['fail_count'] += 1
                return
            if image.output_temp_annotation(temp_annotation_output_path):
                process_output['temp_file_name_list'].append(
                    image.file_name_new)
                process_output['success_count'] += 1
            else:
                print('errow output temp annotation: {}'.format(
                    image.file_name_new))
                process_output['fail_count'] += 1

        return

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base) -> None:
        """[输出target annotation]

        Args:
            dataset (Dataset_Base): [数据集实例]
        """

        print('\nStart transform to target dataset:')
        total_annotation_path_list = []
        for dataset_temp_annotation_path_list in tqdm(dataset_instance.temp_divide_file_list[1:-1],
                                                      desc='Get total annotation path list'):
            with open(dataset_temp_annotation_path_list, 'r') as f:
                for n in f.readlines():
                    total_annotation_path_list.append(n.replace('\n', '')
                                                      .replace(dataset_instance.source_dataset_images_folder,
                                                               dataset_instance.temp_annotations_folder)
                                                      .replace(dataset_instance.target_dataset_image_form,
                                                               dataset_instance.temp_annotation_form))

        pbar, update = multiprocessing_list_tqdm(
            total_annotation_path_list, desc='Output target dataset annotation')
        pool = multiprocessing.Pool(dataset_instance.workers)
        for temp_annotation_path in total_annotation_path_list:
            pool.apply_async(func=dataset.__dict__[dataset_instance.target_dataset_style].annotation_output,
                             args=(dataset_instance,
                                   temp_annotation_path,),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        return

    @staticmethod
    def annotation_output(dataset_instance: Dataset_Base, temp_annotation_path: str) -> None:
        """读取暂存annotation

        Args:
            dataset_instance (Dataset_Base): 数据集实例
            temp_annotation_path (str): 暂存annotation路径
        """

        image = dataset_instance.TEMP_LOAD(
            dataset_instance, temp_annotation_path)
        if image == None:
            return
        annotation_output_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            image.file_name + '.' + dataset_instance.target_dataset_annotation_form)
        one_image_bbox = []                                     # 声明每张图片bbox列表
        for object in image.object_list:                        # 遍历单张图片全部bbox
            box_class = str(object.box_clss).replace(
                ' ', '').lower()    # 获取bbox类别
            clss_id = dataset_instance.task_dict['Detection']['Target_dataset_class'].index(
                box_class)
            xyxy = (object.box_xywh[0],
                    object.box_xywh[0] + object.box_xywh[2],
                    object.box_xywh[1],
                    object.box_xywh[1] + object.box_xywh[3])
            xywh_yolo = yolo((image.width, image.height),
                             xyxy)       # 转换bbox至yolo类型
            one_image_bbox.append([clss_id, xywh_yolo])

        with open(annotation_output_path, 'w') as f:   # 创建图片对应txt格式的label文件
            for one_bbox in one_image_bbox:
                f.write(" ".join([str(one_bbox[0]), " ".join(
                    [str(a) for a in one_bbox[1]])]) + '\n')
            f.close()

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取YOLO数据集图片类检测列表]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []
        dataset_instance.total_file_name_path = total_file(
            dataset_instance.temp_informations_folder)
        dataset_instance.target_check_file_name_list = annotations_path_list(
            dataset_instance.total_file_name_path, dataset_instance.target_dataset_annotations_check_count)
        for n in dataset_instance.target_check_file_name_list:
            target_annotation_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder,
                n + '.' + dataset_instance.target_dataset_annotation_form)
            with open(target_annotation_path, 'r') as f:
                image_name = n + '.' + dataset_instance.target_dataset_image_form
                image_path = os.path.join(
                    dataset_instance.temp_images_folder, image_name)
                img = cv2.imread(image_path)
                size = img.shape
                width = int(size[1])
                height = int(size[0])
                channels = int(size[2])
                object_list = []
                for n, one_bbox in enumerate(f.read().splitlines()):
                    true_box = one_bbox.split(' ')[1:]
                    clss = dataset_instance.task_dict['Detection']['Target_dataset_class'][int(
                        one_bbox.split(' ')[0])]
                    clss = clss.strip(' ').lower()
                    true_box = revers_yolo(size, true_box)
                    xmin = min(
                        max(min(float(true_box[0]), float(true_box[1])), 0.), float(width))
                    ymin = min(
                        max(min(float(true_box[2]), float(true_box[3])), 0.), float(height))
                    xmax = max(
                        min(max(float(true_box[1]), float(true_box[0])), float(width)), 0.)
                    ymax = max(
                        min(max(float(true_box[3]), float(true_box[2])), float(height)), 0.)
                    box_xywh = [xmin, ymin, xmax-xmin, ymax-ymin]
                    object_list.append(OBJECT(n,
                                              clss,
                                              box_clss=clss,
                                              box_xywh=box_xywh,
                                              need_convert=dataset_instance.need_convert))  # 将单个真实框加入单张图片真实框列表
                image = IMAGE(image_name, image_name, image_path, int(
                    height), int(width), int(channels), object_list)
                check_images_list.append(image)

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成YOLO组织格式的数据集]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder, 'YOLO'))
        shutil.rmtree(output_root)
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder, 'YOLO'))
        annotations_output_folder = check_output_path(
            os.path.join(output_root, 'annotations'))

        print('Start copy images:')
        image_list = []
        image_output_folder = check_output_path(
            os.path.join(output_root, 'images'))
        with open(dataset_instance.temp_divide_file_list[0], 'r') as f:
            for n in f.readlines():
                image_list.append(n.replace('\n', ''))
        pbar, update = multiprocessing_list_tqdm(
            image_list, desc='Copy images', leave=False)
        pool = multiprocessing.Pool(dataset_instance.workers)
        for image_input_path in image_list:
            image_output_path = image_input_path.replace(
                dataset_instance.temp_images_folder, image_output_folder)
            pool.apply_async(func=shutil.copy,
                             args=(image_input_path, image_output_path,),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        print('Start copy annotations:')
        for root, dirs, files in os.walk(dataset_instance.target_dataset_annotations_folder):
            for n in tqdm(files, desc='Copy annotations'):
                annotations_input_path = os.path.join(root, n)
                annotations_output_path = annotations_input_path.replace(
                    dataset_instance.target_dataset_annotations_folder,
                    annotations_output_folder)
                shutil.copy(annotations_input_path, annotations_output_path)

        return
