'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-13 12:10:53
LastEditors: Leidi
LastEditTime: 2021-10-21 20:08:42
'''
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='classes_load.py')
    parser.add_argument('--filepath', '--fp', dest='filepath', default=r'Multi_task_dataset_clean_up/data/classes_detect_huawei_173_classes_20210918.names',
                        type=str, help='dataset path')
    parser.add_argument('--output', '--o', dest='output', default=r'classes_detect_huawei_173_classes_20210918_cvat.names',
                        type=str, help='input labels style: pascal_voc, coco2017, tt100k')

    opt = parser.parse_args()

output = opt.output
classes_list = []
with open(opt.filepath, 'r') as f:
    with open(output, 'w') as q:
        for n in f.readlines():
            s = n.replace('\n', '')
            classes_list.append("\"" + s + "\":" + "\"" + s + "\"")
        q.write(','.join(classes_list))
        q.close()
    f.close()
