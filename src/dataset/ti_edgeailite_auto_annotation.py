'''
Description:
Version:
Author: Leidi
Date: 2022-01-07 17:43:48
LastEditors: Leidi
LastEditTime: 2022-02-22 15:14:11
'''
import shutil
import multiprocessing
from collections import namedtuple

from utils.utils import *
from base.image_base import *
from utils import image_form_transform
from base.dataset_base import Dataset_Base


class TI_EDGEAILITE_AUTO_ANNOTATION(Dataset_Base):

    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.source_dataset_image_form_list = ['jpg', 'png']
        self.source_dataset_annotation_form = 'png'
        self.source_dataset_image_count = len(os.listdir(
            os.path.join(self.dataset_input_folder, 'images')))
        self.source_dataset_annotation_count = len(os.listdir(
            os.path.join(self.dataset_input_folder, 'annotations')))

    def source_dataset_copy_image_and_annotation(self):

        print('\nStart source dataset copy image and annotation:')
        pbar, update = multiprocessing_object_tqdm(
            self.source_dataset_image_count, 'Copy images')
        pool = multiprocessing.Pool(self.workers)
        for n in os.listdir(os.path.join(self.dataset_input_folder, 'images')):
            if os.path.splitext(n)[-1].replace('.', '') in \
                    self.source_dataset_image_form_list:
                pool.apply_async(self.source_dataset_copy_image,
                                 args=(n,),
                                 callback=update,
                                 error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        pbar, update = multiprocessing_object_tqdm(
            self.source_dataset_annotation_count, 'Copy annotations')
        pool = multiprocessing.Pool(self.workers)
        for n in os.listdir(os.path.join(self.dataset_input_folder, 'annotations')):
            if n.endswith(self.source_dataset_annotation_form):
                pool.apply_async(self.source_dataset_copy_annotation,
                                 args=(n,),
                                 callback=update,
                                 error_callback=err_call_back)
        pool.close()
        pool.join()
        pbar.close()

        print('Copy images and annotations end.')

        return

    def source_dataset_copy_image(self, n: str) -> None:
        """[???????????????????????????????????????????????????????????????????????????????????????]

        Args:
            root (str): [??????????????????]
            n (str): [?????????]
        """

        image = os.path.join(os.path.join(
            self.dataset_input_folder, 'images'), n)
        temp_image = os.path.join(
            self.source_dataset_images_folder, self.file_prefix + n)
        image_suffix = os.path.splitext(n)[-1].replace('.', '')
        if image_suffix != self.target_dataset_image_form:
            image_transform_type = image_suffix + \
                '_' + self.target_dataset_image_form
            image_form_transform.__dict__[
                image_transform_type](image, temp_image)
            return
        else:
            shutil.copy(image, temp_image)
            return

    def source_dataset_copy_annotation(self, n: str) -> None:
        """[??????????????????????????????????????????????????????source_annotations???]

        Args:
            root (str): [??????????????????]
            n (str): [?????????]
        """

        annotation = os.path.join(os.path.join(
            self.dataset_input_folder, 'annotations'), n)
        temp_annotation = os.path.join(
            self.source_dataset_annotations_folder, n)
        shutil.copy(annotation, temp_annotation)

        return

    def load_image_annotation(self, source_annotation_name: str, process_output: dict) -> None:
        """?????????????????????????????????

        Args:
            source_annotation_name (str): ?????????????????????
            process_output (dict): ?????????????????????
        """

        # ??????Label??????????????????
        Label = namedtuple('Label', [
            'name',
            'id',
            'category',
            'color', ])

        labels = []
        for n, clss in enumerate(self.task_dict['Semantic_segmentation']['Source_dataset_class']):
            labels.append(Label(clss, n, clss, (n, n, n)))

        # ??????data?????????images?????????????????????file_name???height???width
        source_annotation_path = os.path.join(
            self.source_dataset_annotations_folder, source_annotation_name)
        image_name = (os.path.splitext(source_annotation_name)[
            0] + '.' + self.temp_image_form).replace('_bin', '')
        image_name_new = self.file_prefix + image_name
        image_path = os.path.join(
            self.temp_images_folder, image_name_new)
        source_image = cv2.imread(image_path)
        source_image_height, source_image_width, source_image_channels = source_image.shape
        # ??????png?????????IMAGE?????????
        annotation_image = cv2.imread(source_annotation_path)
        annotation_height, annotation_width, annotation_channels = annotation_image.shape
        if source_image_height != annotation_height\
                or source_image_width != annotation_width:
            annotation_image = cv2.resize(
                annotation_image, (source_image_width, source_image_height), interpolation=cv2.INTER_CUBIC)

        mask = np.zeros_like(annotation_image)
        object_list = []
        for one in labels:
            if one.name == 'unlabeled':
                continue
            bgr = [one.color[2], one.color[1], one.color[0]]
            for n, color in enumerate(bgr):
                mask[:, :, n] = color
            # ???mask?????????????????????????????????????????????01?????????
            image_xor = cv2.bitwise_xor(mask, annotation_image)
            # ????????????01????????????????????????
            image_gray = cv2.cvtColor(image_xor.copy(), cv2.COLOR_BGR2GRAY)
            # ???????????????????????????mask????????????????????????
            _, thresh1 = cv2.threshold(
                image_gray, 1, 255, cv2.THRESH_BINARY_INV)
            if np.all(thresh1 == 0):
                continue
            # ???????????????????????????????????????????????????????????????polygon?????????
            contours, _ = cv2.findContours(
                thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # ???????????????polygon??????????????????object
            for n, point in enumerate(contours):
                # ?????????????????????
                if cv2.contourArea(point) >= 800000:
                    continue
                point = np.squeeze(point)
                point = np.squeeze(point)
                point = point.tolist()
                if 3 > len(point):
                    continue
                # ??????
                object_list.append(OBJECT(n,
                                          object_clss=one.name,
                                          segmentation_clss=one.name,
                                          segmentation=point,
                                          need_convert=self.need_convert))

        # ??????????????????????????????????????????????????????????????????per_image???????????????
        # ?????????????????????????????????total_images_data_list
        image = IMAGE(image_name, image_name_new, image_path,
                      source_image_height, source_image_width, source_image_channels, object_list)
        # ??????????????????????????????????????????source annotation???temp annotation
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
        """[??????target annotation]

        Args:
            dataset (Dataset_Base): [???????????????]
        """

        print('\nStart transform to target dataset:')

        return

    @staticmethod
    def annotation_check(dataset_instance: Dataset_Base) -> list:
        """[??????TI_EDGEAILITE_AUTO_ANNOTATION??????????????????????????????]

        Args:
            dataset_instance (object): [???????????????]

        Returns:
            list: [??????????????????????????????]
        """

        check_images_list = []

        return check_images_list

    @staticmethod
    def target_dataset_folder(dataset_instance: Dataset_Base) -> None:
        """[??????TI_EDGEAILITE_AUTO_ANNOTATION????????????????????????]

        Args:
            dataset_instance (object): [???????????????]
        """

        print('\nStart build target dataset folder:')

        return
