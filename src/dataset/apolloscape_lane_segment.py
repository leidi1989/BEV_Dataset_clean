'''
Description: 
Version: 
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-22 14:59:19
'''
from collections import namedtuple

from utils.utils import *
from base.image_base import *
from base.dataset_base import Dataset_Base


class APOLLOSCAPE_LANE_SEGMENT(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg']
        self.source_dataset_annotation_form = 'png'
        self.source_dataset_image_count = self.get_source_dataset_image_count()
        self.source_dataset_annotation_count = self.get_source_dataset_annotation_count()

    def load_image_annotation(self, source_annotation_name: str, process_output: dict) -> None:
        """将源标注转换为暂存标注

        Args:
            source_annotation_name (str): 源标注文件名称
            process_output (dict): 进程间通信字典
        """
        
        # 声明Label格式命名元组
        Label = namedtuple('Label', [
            'name',
            'id',
            'trainId',
            'category',
            'categoryId',
            'hasInstances',
            'ignoreInEval',
            'color', ])

        labels = [
            Label('void', 0, 0, 'void', 0,
                  False, False, (0, 0, 0)),
            Label('s_w_d', 200, 1, 'dividing', 1,
                  False, False, (70, 130, 180)),
            Label('s_y_d', 204, 2, 'dividing', 1,
                  False, False, (220, 20, 60)),
            Label('ds_w_dn', 213, 3, 'dividing', 1,
                  False, True, (128, 0, 128)),
            Label('ds_y_dn', 209, 4, 'dividing',
                  1, False, False, (255, 0, 0)),
            Label('sb_w_do', 206, 5, 'dividing',
                  1, False, True, (0, 0, 60)),
            Label('sb_y_do', 207, 6, 'dividing',
                  1, False, True, (0, 60, 100)),
            Label('b_w_g', 201, 7, 'guiding', 2,
                  False, False, (0, 0, 142)),
            Label('b_y_g', 203, 8, 'guiding', 2,
                  False, False, (119, 11, 32)),
            Label('db_w_g', 211, 9, 'guiding', 2,
                  False, True, (244, 35, 232)),
            Label('db_y_g', 208, 10, 'guiding', 2,
                  False, True, (0, 0, 160)),
            Label('db_w_s', 216, 11, 'stopping', 3,
                  False, True, (153, 153, 153)),
            Label('s_w_s', 217, 12, 'stopping', 3,
                  False, False, (220, 220, 0)),
            Label('ds_w_s', 215, 13, 'stopping', 3,
                  False, True, (250, 170, 30)),
            Label('s_w_c', 218, 14, 'chevron', 4,
                  False, True, (102, 102, 156)),
            Label('s_y_c', 219, 15, 'chevron', 4,
                  False, True, (128, 0, 0)),
            Label('s_w_p', 210, 16, 'parking', 5,
                  False, False, (128, 64, 128)),
            Label('s_n_p', 232, 17, 'parking', 5,
                  False, True, (238, 232, 170)),
            Label('c_wy_z', 214, 18, 'zebra', 6,
                  False, False, (190, 153, 153)),
            Label('a_w_u', 202, 19, 'thru/turn', 7,
                  False, True, (0, 0, 230)),
            Label('a_w_t', 220, 20, 'thru/turn', 7,
                  False, False, (128, 128, 0)),
            Label('a_w_tl', 221, 21, 'thru/turn', 7,
                  False, False, (128, 78, 160)),
            Label('a_w_tr', 222, 22, 'thru/turn', 7,
                  False, False, (150, 100, 100)),
            Label('a_w_tlr', 231, 23, 'thru/turn', 7,
                  False, True, (255, 165, 0)),
            Label('a_w_l', 224, 24, 'thru/turn', 7,
                  False, False, (180, 165, 180)),
            Label('a_w_r', 225, 25, 'thru/turn', 7,
                  False, False, (107, 142, 35)),
            Label('a_w_lr', 226, 26, 'thru/turn', 7,
                  False, False, (201, 255, 229)),
            Label('a_n_lu', 230, 27, 'thru/turn', 7,
                  False, True, (0, 191, 255)),
            Label('a_w_tu', 228, 28, 'thru/turn', 7,
                  False, True, (51, 255, 51)),
            Label('a_w_m', 229, 29, 'thru/turn', 7,
                  False, True, (250, 128, 114)),
            Label('a_y_t', 233, 30, 'thru/turn', 7,
                  False, True, (127, 255, 0)),
            Label('b_n_sr', 205, 31, 'reduction', 8,
                  False, False, (255, 128, 0)),
            Label('d_wy_za', 212, 32, 'attention',
                  9, False, True, (0, 255, 255)),
            Label('r_wy_np', 227, 33, 'no parking', 10,
                  False, False, (178, 132, 190)),
            Label('vom_wy_n', 223, 34, 'others', 11,
                  False, True, (128, 128, 64)),
            Label('om_n_n', 250, 35, 'others', 11,
                  False, False, (102, 0, 204)),
            Label('noise', 249, 255, 'ignored', 255,
                  False, True, (0, 153, 153)),
            Label('ignored', 255, 255, 'ignored', 255,
                  False, True, (255, 255, 255)), ]

        # 获取data字典中images内的图片信息，file_name、height、width
        source_annotation_path = os.path.join(
            self.source_dataset_annotations_folder, source_annotation_name)
        image_name = (os.path.splitext(source_annotation_name)[
            0] + '.' + self.temp_image_form).replace('_bin', '')
        image_name_new = self.file_prefix + image_name
        image_path = os.path.join(
            self.temp_images_folder, image_name_new)
        img = cv2.imread(image_path)
        height, width, channels = img.shape

        # 转换png标签为IMAGE类实例
        annotation_image = cv2.imread(source_annotation_path)
        mask = np.zeros_like(annotation_image)
        object_list = []
        for one in labels:
            if one.name == 'void':
                continue
            bgr = [one.color[2], one.color[1], one.color[0]]
            for n, color in enumerate(bgr):
                mask[:, :, n] = color
            # 将mask和标注图片进行求亦或，获取亦或01结果图
            image_xor = cv2.bitwise_xor(mask, annotation_image)
            # 使用亦或01结果图获取灰度图
            image_gray = cv2.cvtColor(image_xor.copy(), cv2.COLOR_BGR2GRAY)
            # 使用灰度图获取指定mask类别色彩的二值图
            _, thresh1 = cv2.threshold(
                image_gray, 1, 255, cv2.THRESH_BINARY_INV)
            if np.all(thresh1 == 0):
                continue
            # 使用二值图求取指定类别的包围框，即获取标注polygon包围框
            contours, _ = cv2.findContours(
                thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # 将多个标注polygon包围框转换为object
            for n, point in enumerate(contours):
                point = np.squeeze(point)
                point = np.squeeze(point)
                point = point.tolist()
                if 3 > len(point):
                    continue
                object_list.append(OBJECT(n,
                                          object_clss=one.name,
                                          segmentation_clss=one.name,
                                          segmentation=point,
                                          need_convert=self.need_convert))

        # 将获取的图片名称、图片路径、高、宽作为初始化per_image对象参数，
        # 并将初始化后的对象存入total_images_data_list
        image = IMAGE(image_name, image_name_new, image_path,
                      height, width, channels, object_list)
        # 读取目标标注信息，输出读取的source annotation至temp annotation
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

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[读取APOLLOSCAPE_LANE_SEGMENT数据集图片类检测列表]

        Args:
            dataset_instance (object): [数据集实例]

        Returns:
            list: [数据集图片类检测列表]
        """

        check_images_list = []

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[生成APOLLOSCAPE_LANE_SEGMENT组织格式的数据集]

        Args:
            dataset_instance (object): [数据集实例]
        """

        print('\nStart build target dataset folder:')

        return
