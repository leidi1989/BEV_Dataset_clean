'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-13 12:10:53
LastEditors: Leidi
LastEditTime: 2021-10-21 20:08:36
'''
import os
import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='classes_load.py')
    parser.add_argument('--filepath', '--fp', dest='filepath', default=r'/home/leidi/Dataset/tt100k_input_20210814/annotations.json',
                        type=str, help='dataset path')
    parser.add_argument('--datasetname', '--dn', dest='datasetname', default=r'tt100k',
                        type=str, help='output path')
    parser.add_argument('--output', '--o', dest='output', default=r'',
                        type=str, help='input labels style: pascal_voc, coco2017, tt100k')
    
    opt = parser.parse_args()

output = os.path.join('tool', opt.datasetname + '_classes.names')
with open(opt.filepath, 'r') as f:
    data = json.loads(f.read())
    with open(output, 'w') as q:
        data['types'].sort()
        for classes in data['types']:    # 获取数据集image中最大id数
            q.write('%s\n' % classes)
        q.close()
    f.close()
