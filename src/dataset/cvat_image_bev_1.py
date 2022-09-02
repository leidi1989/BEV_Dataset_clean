'''
Description: 数据集图片由6相机图像及激光点云bev图拼接而成
Version:
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-08-24 10:45:02
'''
import multiprocessing
import shutil
import xml.etree.ElementTree as ET

from base.dataset_base import Dataset_Base
from utils import image_form_transform
from base.image_base import *
from PIL import Image
from utils.utils import *

import dataset


class CVAT_IMAGE_BEV_1(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'xml'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count(
        )

    def draw_mask(self, ann_img):
        if self.annotation_car == 'hq1':
            # 给红旗1数据加mask
            pts = np.array(
                [[[2851.58, 1200], [2929.35, 1169.51], [3050.25, 1145.83],
                  [3088.43, 1049.67], [3116.36, 1013.61], [3123.43, 1007.25],
                  [3143.58, 994.52], [3197.67, 927.00], [3199.43, 1200],
                  [2853.34, 1200]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array([[[0, 1028.33], [21.59, 1054.00], [51.05, 1071.38],
                             [93.76, 1125.00], [107.02, 1155.64],
                             [293.85, 1200], [1.32, 1200], [0, 1090.86]]],
                           np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[2898.67, 600], [3161.83, 490.60], [3198.49, 341.79],
                  [3198.49, 600], [2900.87, 600]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[2133, 347.36], [2188.71, 365.69], [2242.95, 469.05],
                  [2282.54, 600.26], [2133, 600.26], [2133, 350.29]]],
                np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[1066, 565.81], [1114.79, 557.01], [1232.81, 548.95],
                  [1540.69, 556.28], [1539.96, 498.37], [1556.82, 467.58],
                  [1822.18, 463.18], [1870.56, 491.04], [1870.56, 548.95],
                  [2019.37, 543.08], [2133.00, 531.36], [2133, 600],
                  [1066, 600], [1066, 568.01]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[861.15, 600], [911.00, 436.79], [968.18, 345.89],
                  [1010.70, 338.56], [1028.29, 326.83], [1066.41, 331.23],
                  [1064.94, 600], [864.82, 600]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[0, 0], [22.40, 0], [49.52, 342.01], [81.78, 505.48],
                  [325.15, 600], [0, 600], [0, 271.16]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array([[[1893.17, 546.85], [2133.50, 526.10],
                             [2133.69, 549.58], [1893.35, 548.13]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))
            pts = np.array(
                [[[1129.20, 556.10], [1207.11, 549.80], [1249.32, 547.71],
                  [1413.71, 552.69], [1495.25, 554.26], [1517.54, 556.36],
                  [1129.80, 557.10]]], np.int32)
            pts = pts.reshape((-1, 1, 2))
            ann_img = cv2.fillConvexPoly(ann_img, pts, (0, 0, 0))

        return ann_img

    def source_dataset_copy_image(self, root: str, n: str) -> None:
        """[复制源数据集图片至暂存数据集并修改图片类别、添加文件名前缀]

        Args:
            root (str): [文件所在目录]
            n (str): [文件名]
        """

        image = os.path.join(root, n)
        temp_image = os.path.join(self.source_dataset_images_folder,
                                  self.file_prefix + n)
        image_suffix = os.path.splitext(n)[-1].replace('.', '')
        if image_suffix != self.target_dataset_image_form:
            image_transform_type = image_suffix + \
                '_' + self.target_dataset_image_form
            image_form_transform.__dict__[image_transform_type](image,
                                                                temp_image)
            if self.draw_car_mask:
                image = cv2.imread(temp_image)
                image = self.draw_mask(image)
                cv2.imwrite(temp_image, image)
            return
        else:
            shutil.copy(image, temp_image)
            if self.draw_car_mask:
                image = cv2.imread(temp_image)
                image = self.draw_mask(image)
                cv2.imwrite(temp_image, image)
            return

    def transform_to_temp_dataset(self) -> None:
        """[转换标注文件为暂存标注]
        """

        print('\nStart transform to temp dataset:')
        success_count = 0
        fail_count = 0
        no_object = 0
        temp_file_name_list = []
        total_source_dataset_annotations_list = os.listdir(
            self.source_dataset_annotations_folder)

        for source_annotation_name in tqdm(
                total_source_dataset_annotations_list,
                desc='Total annotations'):

            process_temp_file_name_list = multiprocessing.Manager().list()
            process_output = multiprocessing.Manager().dict({
                'success_count':
                0,
                'fail_count':
                0,
                'no_object':
                0,
                'temp_file_name_list':
                process_temp_file_name_list
            })
            pool = multiprocessing.Pool(self.workers)
            source_annotations_path = os.path.join(
                self.source_dataset_annotations_folder, source_annotation_name)
            tree = ET.parse(source_annotations_path)
            root = tree.getroot()
            pbar, update = multiprocessing_list_tqdm(root,
                                                     desc='Total tags',
                                                     leave=False)
            for annotation in root:
                if annotation.tag != 'image':
                    continue
                pool.apply_async(func=self.load_image_annotation,
                                 args=(
                                     annotation,
                                     process_output,
                                 ),
                                 callback=update,
                                 error_callback=err_call_back)
            pool.close()
            pool.join()
            pbar.close()

            # debug
            # for annotation in root:
            #     if annotation.tag != 'image':
            #         continue
            #     self.load_image_annotation(annotation,process_output)

            success_count += process_output['success_count']
            fail_count += process_output['fail_count']
            no_object = process_output['no_object']
            temp_file_name_list += process_output['temp_file_name_list']

        # 输出读取统计结果
        print('\nSource dataset convert to temp dataset file count: ')
        print('Total annotations:         \t {} '.format(
            len(total_source_dataset_annotations_list)))
        print('Convert fail:              \t {} '.format(fail_count))
        print('No object delete images: \t {} '.format(no_object))
        print('Convert success:           \t {} '.format(success_count))
        self.temp_annotation_name_list = temp_file_name_list
        print('Source dataset annotation transform to temp dataset end.')

        return

    def load_image_annotation(self, annotation, process_output: dict) -> None:
        """将源标注转换为暂存标注

        Args:
            source_annotation_name (str): 源标注文件名称
            process_output (dict): 进程间通信字典
        """
        image_name = str(annotation.attrib['name']).split(
            '.')[0] + '.' + self.temp_image_form
        image_name_new = self.file_prefix + image_name
        image_path = os.path.join(self.temp_images_folder, image_name_new)
        channels = 0
        if not self.only_statistic:
            if not os.path.exists(image_path):
                print('\n {} not exist.'.format(image_name_new))
                process_output['fail_count'] += 1
                return
            img = cv2.imread(image_path)
            if img is None:
                print('Can not load: {}'.format(image_name_new))
                process_output['fail_count'] += 1
                return
            channels = img.shape[-1]
        else:
            channels = 3
        width = int(annotation.attrib['width'])
        height = int(annotation.attrib['height'])
        object_list = []
        annotation_children_node = annotation.getchildren()
        # get object box head orin
        head_points_dict = {}
        for obj in annotation_children_node:
            if obj.tag != 'points':  # 只处理points标签
                continue
            obj_children_node = obj.getchildren()
            if obj_children_node[0].text is None:
                continue
            head_point_id = ''.join(
                list(filter(str.isnumeric, obj_children_node[0].text)))
            point_list = list(map(float, obj.attrib['points'].split(',')))
            head_points_dict.update({head_point_id: point_list})
        # get object information
        for n, obj in enumerate(annotation_children_node):
            if obj.tag != 'box':  # 只处理box标签
                continue
            obj_children_node = obj.getchildren()
            if obj_children_node[0].text is None:
                object_head_point_id = None
                clss = obj.attrib['label']
            else:
                object_head_point_id = ''.join(
                    list(filter(str.isnumeric, obj_children_node[0].text)))
                clss = ''.join(
                    list(filter(str.isalpha, obj_children_node[0].text)))
            clss = clss.replace(' ', '').lower()
            if clss not in self.total_task_source_class_list:
                continue
            box_xywh = []
            x = int(float(obj.attrib['xtl']))
            y = int(float(obj.attrib['ytl']))
            w = int(float(obj.attrib['xbr']) - float(obj.attrib['xtl']))
            h = int(float(obj.attrib['ybr']) - float(obj.attrib['ytl']))
            box_xywh = [x, y, w, h]
            box_xtlytlxbrybr = [
                float(obj.attrib['xtl']),
                float(obj.attrib['ytl']),
                float(obj.attrib['xbr']),
                float(obj.attrib['ybr'])
            ]
            if 'rotation' in obj.attrib:
                box_rotation = float(obj.attrib['rotation'])
            else:
                box_rotation = 0
            # get head orientation
            if object_head_point_id is not None and object_head_point_id in head_points_dict:
                box_head_point = head_points_dict[object_head_point_id]
            else:
                box_head_point = None
            object_list.append(
                OBJECT(n,
                       clss,
                       box_clss=clss,
                       box_xywh=box_xywh,
                       box_xtlytlxbrybr=box_xtlytlxbrybr,
                       box_rotation=box_rotation +
                       self.label_object_rotation_angle,
                       box_head_point=box_head_point,
                       need_convert=self.need_convert))
        image = IMAGE(image_name, image_name_new, image_path, height, width,
                      channels, object_list)
        # 读取目标标注信息，输出读取的source annotation至temp annotation
        if image is None:
            process_output['fail_count'] += 1
            return
        temp_annotation_output_path = os.path.join(
            self.temp_annotations_folder,
            image.file_name_new + '.' + self.temp_annotation_form)
        image.object_class_modify(self)
        image.object_pixel_limit(self)
        if 0 == len(image.object_list) and not self.keep_no_object:
            print('{} no object, has been delete.'.format(
                image.image_name_new))
            if not self.only_statistic:
                os.remove(image.image_path)
            process_output['no_object'] += 1
            process_output['fail_count'] += 1
            return
        if image.output_temp_annotation(temp_annotation_output_path):
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            print('errow output temp annotation: {}'.format(
                image.file_name_new))
            process_output['fail_count'] += 1

        return

    def get_image_mean_std(self, img_filename: str) -> list:
        """[获取图片均值和标准差]

        Args:
            dataset (dict): [数据集信息字典]
            img_filename (str): [图片名]

        Returns:
            list: [图片均值和标准差列表]
        """

        img_type = os.path.splitext(img_filename)[-1].split('.')[-1]
        if img_type == 'png':
            img = Image.open(
                os.path.join(self.source_dataset_images_folder, img_filename))
            img = cv2.cvtColor(np.asarray(img.getdata(), dtype='uint8'),
                               cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(
                os.path.join(self.source_dataset_images_folder, img_filename))
        img = img[0:self.camera_image_wh[1], 0:self.camera_image_wh[0], :]
        m, s = cv2.meanStdDev(img / 255.0)
        name = img_filename

        return m.reshape((3, )), s.reshape((3, )), name

    def get_dataset_image_mean_std(self) -> None:
        """[计算读取的数据集图片均值、标准差]

        Args:
            dataset (dict): [数据集信息字典]
        """

        img_filenames = os.listdir(self.source_dataset_images_folder)
        print('\nStart count images mean and std:')
        pbar, update = multiprocessing_list_tqdm(
            img_filenames, desc='Count images mean and std')
        pool = multiprocessing.Pool(self.workers)
        mean_std_list = []
        for img_filename in img_filenames:
            mean_std_list.append(
                pool.apply_async(func=self.get_image_mean_std,
                                 args=(img_filename, ),
                                 callback=update,
                                 error_callback=err_call_back))
        pool.close()
        pool.join()
        pbar.close()

        m_list, s_list = [], []
        for n in mean_std_list:
            m_list.append(n.get()[0])
            s_list.append(n.get()[1])
        m_array = np.array(m_list)
        s_array = np.array(s_list)
        m = m_array.mean(axis=0, keepdims=True)
        s = s_array.mean(axis=0, keepdims=True)

        mean_std_file_output_path = os.path.join(self.temp_informations_folder,
                                                 'mean_std.json')
        mean_std = [{'mean': m[0].tolist(), 'std': s[0].tolist()}]
        mean_std_json = json.dumps(mean_std)
        with open(mean_std_file_output_path, 'w') as f:
            f.write(mean_std_json)
            f.close()
        print(mean_std)
        print('Count images mean and std end.')

        return

    def per_image_standardization(self, img_filename: str, mean, std,
                                  check_dataset_image_folder) -> list:
        """对图片进行标准化

        Args:
            img_filename (str): _description_

        Returns:
            list: _description_
        """

        img_type = os.path.splitext(img_filename)[-1].split('.')[-1]
        if img_type == 'png':
            img = Image.open(
                os.path.join(self.source_dataset_images_folder, img_filename))
            img = cv2.cvtColor(np.asarray(img.getdata(), dtype='uint8'),
                               cv2.COLOR_RGB2BGR)
        else:
            img = cv2.imread(
                os.path.join(self.source_dataset_images_folder, img_filename))

        img = (img[0:self.camera_image_wh[1], 0:self.camera_image_wh[0], :] -
               mean * 255) / std
        img_output_path = os.path.join(check_dataset_image_folder,
                                       img_filename)
        cv2.imwrite(img_output_path, img)

        return

    def check_dataset_image_mean_std(self) -> None:
        """检查数据集图片均值和方差
        """
        print('Start check dataset image mean std')
        mean_std_file_output_path = os.path.join(self.temp_informations_folder,
                                                 'mean_std.json')
        if not os.path.exists(mean_std_file_output_path):
            print('No mean std file, return!')
            return
        else:
            with open(mean_std_file_output_path, 'r', encoding='utf8') as f:
                mean_std = json.load(f)
                mean = np.array(mean_std[0]['mean'])
                std = np.array(mean_std[0]['std'])
        check_dataset_image_folder = os.path.join(
            self.temp_informations_folder, 'check_mean_std')
        if os.path.exists(check_dataset_image_folder):
            shutil.rmtree(check_dataset_image_folder)
        check_output_path(check_dataset_image_folder)
        img_filenames = os.listdir(self.temp_images_folder)
        if self.target_dataset_annotations_check_count < len(img_filenames):
            random_check_images_list = random.sample(
                img_filenames, self.target_dataset_annotations_check_count)
        else:
            random_check_images_list = img_filenames
        pbar, update = multiprocessing_list_tqdm(
            random_check_images_list, desc='Check images mean and std')
        pool = multiprocessing.Pool(self.workers)
        for img_filename in random_check_images_list:
            pool.apply_async(func=self.per_image_standardization,
                             args=(img_filename, mean, std,
                                   check_dataset_image_folder),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        return

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base) -> None:
        """输出target annotation

        Args:
            dataset_instance (Dataset_Base): 数据集实例
        """

        print('\nStart transform to target dataset:')
        class_color_encode_dict = {}
        for task_class_dict in dataset_instance.task_dict.values():
            if task_class_dict is not None:
                for n in task_class_dict['Target_dataset_class']:
                    class_color_encode_dict.update({n: 0})
        for n, key in zip(
                random.sample([x for x in range(255)],
                              len(class_color_encode_dict)),
                class_color_encode_dict.keys()):
            class_color_encode_dict[key] = RGB_to_Hex(
                str(n) + ',' + str(n) + ',' + str(n))

        # 生成空基本信息xml文件
        annotations = dataset.__dict__[
            dataset_instance.target_dataset_style].annotation_creat_root(
                dataset_instance, class_color_encode_dict)
        # 获取全部图片标签信息列表
        for task, task_class_dict in tqdm(dataset_instance.task_dict.items(),
                                          desc='Load each task annotation'):
            if task_class_dict is None \
                    or task_class_dict['Target_dataset_class'] is None:
                continue
            total_image_xml = []
            pbar, update = multiprocessing_list_tqdm(
                dataset_instance.temp_annotations_path_list,
                desc='transform to target dataset',
                leave=False)
            pool = multiprocessing.Pool(dataset_instance.workers)
            for temp_annotation_path in dataset_instance.temp_annotations_path_list:
                total_image_xml.append(
                    pool.apply_async(func=dataset.__dict__[
                        dataset_instance.target_dataset_style].
                                     annotation_get_temp,
                                     args=(
                                         dataset_instance,
                                         temp_annotation_path,
                                         task,
                                         task_class_dict,
                                     ),
                                     callback=update,
                                     error_callback=err_call_back))
            pool.close()
            pool.join()
            pbar.close()

            # 将image标签信息添加至annotations中
            for n, image in enumerate(total_image_xml):
                annotation = image.get()
                annotation.attrib['id'] = str(n)
                annotations.append(annotation)

            tree = ET.ElementTree(annotations)

            annotation_output_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder,
                'annotatons.' +
                dataset_instance.target_dataset_annotation_form)
            tree.write(annotation_output_path,
                       encoding='utf-8',
                       xml_declaration=True)

        return

    @staticmethod
    def annotation_creat_root(dataset_instance: Dataset_Base,
                              class_color_encode_dict: dict) -> object:
        """创建xml根节点

        Args:
            dataset_instance (dict): 数据集信息字典
            class_color_encode_dict (dict): 色彩字典

        Returns:
            object: ET.Element格式标注信息
        """

        class_id = 0
        annotations = ET.Element('annotations')
        version = ET.SubElement(annotations, 'version')
        version.text = '1.1'
        meta = ET.SubElement(annotations, 'meta')
        task = ET.SubElement(meta, 'task')
        # ET.SubElement(task, 'id')
        # ET.SubElement(task, 'name')
        # ET.SubElement(task, 'size')
        # mode = ET.SubElement(task, 'mode')
        # mode.text = 'annotation'
        # overlap = ET.SubElement(task, 'overlap')
        # ET.SubElement(task, 'bugtracker')
        # ET.SubElement(task, 'created')
        # ET.SubElement(task, 'updated')
        # subset = ET.SubElement(task, 'subset')
        # subset.text = 'default'
        # start_frame = ET.SubElement(task, 'start_frame')
        # start_frame.text='0'
        # ET.SubElement(task, 'stop_frame')
        # ET.SubElement(task, 'frame_filter')
        # segments = ET.SubElement(task, 'segments')
        # segment = ET.SubElement(segments, 'segment')
        # ET.SubElement(segments, 'id')
        # start = ET.SubElement(segments, 'start')
        # start.text='0'
        # ET.SubElement(segments, 'stop')
        # ET.SubElement(segments, 'url')
        # owner = ET.SubElement(task, 'owner')
        # ET.SubElement(owner, 'username')
        # ET.SubElement(owner, 'email')
        # ET.SubElement(task, 'assignee')
        labels = ET.SubElement(task, 'labels')

        class_dict_list_output_path = os.path.join(
            dataset_instance.target_dataset_annotations_folder,
            'class_dict_list.txt')
        with open(class_dict_list_output_path, 'w') as f:
            for n, c in class_color_encode_dict.items():
                label = ET.SubElement(labels, 'label')
                name = ET.SubElement(label, 'name')
                name.text = n
                color = ET.SubElement(label, 'color')
                color.text = c
                attributes = ET.SubElement(label, 'attributes')
                attribute = ET.SubElement(attributes, 'attribute')
                name = ET.SubElement(attribute, 'name')
                name.text = '1'
                mutable = ET.SubElement(attribute, 'mutable')
                mutable.text = 'False'
                input_type = ET.SubElement(attribute, 'input_type')
                input_type.text = 'text'
                default_value = ET.SubElement(attribute, 'default_value')
                default_value.text = None
                values = ET.SubElement(attribute, 'values')
                values.text = None
                # 输出标签色彩txt
                s = '  {\n    "name": "'+n+'",\n    "color": "' + \
                    str(c)+'",\n    "attributes": []\n  },\n'
                f.write(s)
                class_id += 1

            # ET.SubElement(task, 'dumped')
        return annotations

    @staticmethod
    def annotation_get_temp(dataset_instance: Dataset_Base,
                            temp_annotation_path: str, task: str,
                            task_class_dict: dict) -> object:
        """[获取temp标签信息]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]
            temp_annotation_path (str): [暂存标签路径]
            task (str): 任务类型
            task_class_dict (dict): 任务对应类别字典

        Returns:
            object: [ET.Element格式标注信息]
        """

        image = dataset_instance.TEMP_LOAD(dataset_instance,
                                           temp_annotation_path)
        if image == None:
            return
        image_xml = ET.Element(
            'image', {
                'id': '',
                'name': image.image_name_new,
                'width': str(image.width),
                'height': str(image.height)
            })
        for object in image.object_list:
            if task == 'Detection':
                clss = object.box_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                if object.box_exist_flag:
                    box = ET.SubElement(
                        image_xml, 'box', {
                            'label': object.box_clss,
                            'occluded': '0',
                            'source': 'manual',
                            'xtl': str(object.box_xywh[0]),
                            'ytl': str(object.box_xywh[1]),
                            'xbr':
                            str(object.box_xywh[0] + object.box_xywh[2]),
                            'ybr':
                            str(object.box_xywh[1] + object.box_xywh[3]),
                            'z_order': "0",
                            'group_id': str(object.object_id)
                        })
                    attribute = ET.SubElement(box, 'attribute', {'name': '1'})
                    attribute.text = object.box_clss + str(object.object_id)
            elif task == 'Semantic_segmentation':
                segmentation = np.asarray(
                    object.segmentation).flatten().tolist()
                clss = object.segmentation_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                if object.segmentation_exist_flag:
                    point_list = []
                    for x in object.segmentation:
                        point_list.append(str(x[0]) + ',' + str(x[1]))
                    if 2 == len(point_list):
                        continue
                    polygon = ET.SubElement(
                        image_xml, 'polygon', {
                            'label': object.segmentation_clss,
                            'occluded': '0',
                            'source': 'manual',
                            'points': ';'.join(point_list),
                            'z_order': "0",
                            'group_id': str(object.object_id)
                        })
                    attribute = ET.SubElement(polygon, 'attribute',
                                              {'name': '1'})
                    attribute.text = object.segmentation_clss + \
                        str(object.object_id)
            elif task == 'Instance_segmentation':
                segmentation = np.asarray(
                    object.segmentation).flatten().tolist()
                clss = object.segmentation_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                box = ET.SubElement(
                    image_xml, 'box', {
                        'label': object.box_clss,
                        'occluded': '0',
                        'source': 'manual',
                        'xtl': str(object.box_xywh[0]),
                        'ytl': str(object.box_xywh[1]),
                        'xbr': str(object.box_xywh[0] + object.box_xywh[2]),
                        'ybr': str(object.box_xywh[1] + object.box_xywh[3]),
                        'z_order': "0",
                        'group_id': str(object.object_id)
                    })
                attribute = ET.SubElement(box, 'attribute', {'name': '1'})
                attribute.text = object.box_clss + str(object.object_id)
                point_list = []
                for x in object.segmentation:
                    point_list.append(str(x[0]) + ',' + str(x[1]))
                if 2 == len(point_list):
                    continue
                polygon = ET.SubElement(
                    image_xml, 'polygon', {
                        'label': object.segmentation_clss,
                        'occluded': '0',
                        'source': 'manual',
                        'points': ';'.join(point_list),
                        'z_order': "0",
                        'group_id': str(object.object_id)
                    })
                attribute = ET.SubElement(polygon, 'attribute', {'name': '1'})
                attribute.text = object.segmentation_clss + str(
                    object.object_id)
            elif task == 'Keypoint':
                clss = object.keypoints_clss
                if clss not in task_class_dict['Target_dataset_class']:
                    continue
                if object.keypoints_exist_flag:
                    for m, xy in object.keypoints:
                        points = ET.SubElement(
                            image_xml, 'points', {
                                'label': object.keypoints_clss,
                                'occluded': '0',
                                'source': 'manual',
                                'points': str(xy[0]) + ',' + str(xy[1]),
                                'z_order': "0",
                                'group_id': str(object.object_id)
                            })
                        attribute = ET.SubElement(points, 'attribute',
                                                  {'name': '1'})
                        attribute.text = str(object.object_id) + '-' + m

        return image_xml

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取CVAT_IMAGE_BEV_1数据集图片类检测列表]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []
        dataset_instance.total_file_name_path = total_file(
            dataset_instance.temp_informations_folder)
        dataset_instance.target_check_file_name_list = os.listdir(
            dataset_instance.target_dataset_annotations_folder
        )  # 读取target_annotations_folder文件夹下的全部文件名

        print('Start load target annotations:')
        for n in tqdm(dataset_instance.target_check_file_name_list,
                      desc='Load target dataset annotation'):
            if os.path.splitext(n)[-1] != '.xml':
                continue
            source_annotations_path = os.path.join(
                dataset_instance.target_dataset_annotations_folder, n)
        tree = ET.parse(source_annotations_path)
        root = tree.getroot()
        random.shuffle(root)
        root = root[0:dataset_instance.target_dataset_annotations_check_count]
        for annotation in tqdm(root,
                               desc='Load object annotation',
                               leave=False):
            if annotation.tag != 'image':
                continue
            image_name = str(annotation.attrib['name'])
            image_name_new = image_name
            image_path = os.path.join(dataset_instance.temp_images_folder,
                                      image_name_new)
            img = cv2.imread(image_path)
            if img is None:
                print('Can not load: {}'.format(image_name_new))
                return
            width = int(annotation.attrib['width'])
            height = int(annotation.attrib['height'])
            channels = img.shape[-1]
            object_list = []
            for n, obj in enumerate(annotation):
                if obj.tag == 'box':
                    clss = str(obj.attrib['label'])
                    clss = clss.replace(' ', '').lower()
                    box_xywh = [
                        int(obj.attrib['xtl']),
                        int(obj.attrib['ytl']),
                        int(obj.attrib['xbr']) - int(obj.attrib['xtl']),
                        int(obj.attrib['ybr']) - int(obj.attrib['ytl'])
                    ]
                    object_list.append(
                        OBJECT(n,
                               clss,
                               box_clss=clss,
                               box_xywh=box_xywh,
                               need_convert=dataset_instance.need_convert))
                elif obj.tag == 'polygon':
                    clss = str(obj.attrib['label'])
                    clss = clss.replace(' ', '').lower()
                    segment = []
                    for seg in obj.attrib['points'].split(';'):
                        x, y = seg.split(',')
                        x = float(x)
                        y = float(y)
                        segment.append(list(map(int, [x, y])))
                    object_list.append(
                        OBJECT(n,
                               clss,
                               segmentation_clss=clss,
                               segmentation=segment,
                               need_convert=dataset_instance.need_convert))
            image = IMAGE(image_name, image_name, image_path, int(height),
                          int(width), int(channels), object_list)
            check_images_list.append(image)

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成CVAT_IMAGE_BEV_1组织格式的数据集]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]
        """

        print('\nStart build target dataset folder:')
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVAT_IMAGE_BEV_1'))
        shutil.rmtree(output_root)
        output_root = check_output_path(
            os.path.join(dataset_instance.dataset_output_folder,
                         'CVAT_IMAGE_BEV_1'))
        annotations_output_folder = check_output_path(
            os.path.join(output_root, 'annotations'))

        print('Start copy images:')
        image_list = []
        image_output_folder = check_output_path(
            os.path.join(output_root, 'images'))
        with open(dataset_instance.temp_divide_file_list[0], 'r') as f:
            for n in f.readlines():
                image_list.append(n.replace('\n', ''))
        pbar, update = multiprocessing_list_tqdm(image_list,
                                                 desc='Copy images',
                                                 leave=False)
        pool = multiprocessing.Pool(dataset_instance.workers)
        for image_input_path in image_list:
            image_output_path = image_input_path.replace(
                dataset_instance.temp_images_folder, image_output_folder)
            pool.apply_async(func=shutil.copy,
                             args=(
                                 image_input_path,
                                 image_output_path,
                             ),
                             callback=update,
                             error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        print('Start copy annotations:')
        for root, dirs, files in os.walk(
                dataset_instance.target_dataset_annotations_folder):
            for n in tqdm(files, desc='Copy annotations'):
                annotations_input_path = os.path.join(root, n)
                annotations_output_path = annotations_input_path.replace(
                    dataset_instance.target_dataset_annotations_folder,
                    annotations_output_folder)
                shutil.copy(annotations_input_path, annotations_output_path)

        return
