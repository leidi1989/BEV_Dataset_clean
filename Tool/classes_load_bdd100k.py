'''
Description: 
Version: 
Author: Leidi
Date: 2021-08-13 12:10:53
LastEditors: Leidi
LastEditTime: 2021-09-24 18:10:51
'''
import os
import json
import argparse
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='classes_load.py')
    parser.add_argument('--filepath', '--fp', dest='filepath', default=r'/home/leidi/Dataset/BDD100K/bdd100k_labels',
                        type=str, help='dataset path')
    opt = parser.parse_args()

    segment_classes_list = []
    detect_classes_list = []
    for root, dirs, files in tqdm(os.walk(opt.filepath)):
        for n in files:
            json_path = os.path.join(root, n)
            with open(json_path, 'r') as f:
                data = json.loads(f.read())
                a = data['frames'][0]['objects']
                for i in a:
                    if 'poly2d' in i:
                        segment_classes_list.append(i['category'])
                    if 'box2d' in i:
                        detect_classes_list.append(i['category'])
                f.close()

    segment_classes_list = sorted(set(segment_classes_list))
    detect_classes_list = sorted(set(detect_classes_list))
    print(segment_classes_list)
    print(detect_classes_list)
