'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-23 10:29:20
'''
import multiprocessing

from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base


class TT100K(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg']
        self.source_dataset_annotation_form = 'json'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    def transform_to_temp_dataset(self) -> None:
        """[转换标注文件为暂存标注]
        """

        print('\nStart transform to temp dataset:')
        success_count = 0
        fail_count = 0
        no_object = 0
        temp_file_name_list = []

        for source_annotation_name in tqdm(os.listdir(self.source_dataset_annotations_folder),
                                           desc='Total annotations'):
            source_annotation_path = os.path.join(
                self.source_dataset_annotations_folder, source_annotation_name)
            with open(source_annotation_path, 'r') as f:
                data = json.loads(f.read())

            class_dict = {}
            for n, clss in enumerate(data['types']):
                class_dict['%s' % n] = clss

            total_image_list_processing = []
            pbar, update = multiprocessing_list_tqdm(
                data['imgs'], 'Load annotation', leave=False)
            pool = multiprocessing.Pool(self.workers)
            for image_annotation in data['imgs'].values():
                total_image_list_processing.append(pool.apply_async(func=self.load_annotation,
                                                                    args=(
                                                                        image_annotation,),
                                                                    callback=update,
                                                                    error_callback=err_call_back))
            pool.close()
            pool.join()
            pbar.close()

            total_image_list = []
            for n in total_image_list_processing:
                image = n.get()
                if image is not None:
                    total_image_list.append(image)

            del total_image_list_processing

            # 输出读取的source annotation至temp annotation
            pbar, update = multiprocessing_list_tqdm(
                total_image_list, desc='Output temp annotation', leave=False)
            process_temp_file_name_list = multiprocessing.Manager().list()
            process_output = multiprocessing.Manager().dict({'success_count': 0,
                                                             'fail_count': 0,
                                                             'no_object': 0,
                                                             'temp_file_name_list': process_temp_file_name_list
                                                             })
            pool = multiprocessing.Pool(self.workers)
            for image in total_image_list:
                pool.apply_async(func=self.output_temp_annotation,
                                 args=(image, process_output,),
                                 callback=update,
                                 error_callback=err_call_back)
            pool.close()
            pool.join()
            pbar.close()

            # 更新输出统计
            success_count += process_output['success_count']
            fail_count += process_output['fail_count']
            no_object += process_output['no_object']
            temp_file_name_list += process_output['temp_file_name_list']

        # 输出读取统计结果
        print('\nSource dataset convert to temp dataset file count: ')
        print('Total annotations:         \t {} '.format(len(total_image_list)))
        print('Convert fail:              \t {} '.format(fail_count))
        print('No object delete images: \t {} '.format(no_object))
        print('Convert success:           \t {} '.format(success_count))
        self.temp_annotation_name_list = temp_file_name_list
        print('Source dataset annotation transform to temp dataset end.')

        return

    def load_annotation(self, image_annotation: dict) -> IMAGE:
        """[读取单个图片标注信息]

        Args:
            source_annotation_name (str): [图片标注信息文件名称]
            process_output (dict): [多进程共享字典]
        """

        image_origin, image_name = image_annotation['path'].split('/')
        if image_origin not in ['train', 'test']:
            return
        image_name = os.path.splitext(image_name)[
            0] + '.' + self.target_dataset_image_form
        image_path = os.path.join(
            self.temp_images_folder, image_name)
        image_name_new = self.file_prefix + image_name
        image_path = os.path.join(
            self.temp_images_folder, image_name_new)
        img = cv2.imread(image_path)
        if img is None:
            print('Can not load: {}'.format(image_name_new))
            return
        height, width, channels = img.shape
        object_list = []
        for n, m in enumerate(image_annotation['objects']):
            clss = str(m['category'])
            clss = clss.replace(' ', '').lower()
            if clss not in self.total_task_source_class_list:
                continue
            true_box = m['bbox']
            box = (int(true_box['xmin']),
                   int(true_box['xmax']),
                   int(true_box['ymin']),
                   int(true_box['ymax']))
            xmin = max(min(int(box[0]), int(box[1]), int(width)), 0.)
            ymin = max(min(int(box[2]), int(box[3]), int(height)), 0.)
            xmax = min(max(int(box[1]), int(box[0]), 0.), int(width))
            ymax = min(max(int(box[3]), int(box[2]), 0.), int(height))
            box_xywh = [xmin, ymin, xmax-xmin, ymax-ymin]
            object_list.append(OBJECT(n,
                                      clss,
                                      box_clss=clss,
                                      box_xywh=box_xywh,
                                      need_convert=self.need_convert))  # 将单个真实框加入单张图片真实框列表
        image = IMAGE(image_name, image_name_new, image_path, int(
            height), int(width), int(channels), object_list)

        return image

    def output_temp_annotation(self, image: IMAGE, process_output: dict) -> None:
        """[输出单个标签详细信息至temp annotation]

        Args:
            image (IMAGE): [IMAGE类实例]
            process_output (dict): [进程间计数通信字典]
        """

        if image == None:
            return

        temp_annotation_output_path = os.path.join(
            self.temp_annotations_folder,
            image.file_name_new + '.' + self.temp_annotation_form)
        image.object_class_modify_and_pixel_limit(self)
        if 0 == len(image.object_list) and not self.keep_no_object:
            print('{} no object, has been delete.'.format(
                image.image_name_new))
            os.remove(image.image_path)
            process_output['no_object'] += 1
            return
        if image.output_temp_annotation(temp_annotation_output_path):
            process_output['temp_file_name_list'].append(image.file_name_new)
            process_output['success_count'] += 1
        else:
            print('errow output temp annotation: {}'.format(image.file_name_new))
            process_output['fail_count'] += 1

        return

    @staticmethod
    def target_dataset(dataset_instance: Dataset_Base) -> None:
        """[输出target annotation]

        Args:
            dataset (Dataset_Base): [数据集类]
        """

        print('\nStart transform to target dataset:')

        return

    @staticmethod
    def annotation_output(dataset_instance: Dataset_Base, temp_annotation_path: str) -> None:
        """读取暂存annotation

        Args:
            dataset_instance (Dataset_Base): 数据集实例
            temp_annotation_path (str): 暂存annotation路径
        """

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取TT100K数据集图片类检测列表]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成TT100K组织格式的数据集]

        Args:
            dataset_instance (Dataset_Base): [数据集实例]
        """

        print('\nStart build target dataset folder:')

        return
