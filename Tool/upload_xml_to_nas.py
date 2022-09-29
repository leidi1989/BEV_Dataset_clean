'''
Description: 
Version: 
Author: Leidi
Date: 2022-09-29 10:40:01
LastEditors: Leidi
LastEditTime: 2022-09-29 10:41:06
'''
import argparse
import json
import os

import requests
from upload_to_nas import FTPSync


def export_xml(task_ids, output_path):
    for task_id in task_ids:
        xml_output_path = os.path.join(output_path, task_id + ".zip")
        os.system(
            f"python3 util/cli/cli.py --auth admin:admin123 --server-host 192.168.1.28 dump {task_id} {xml_output_path}"
        )


def get_all_task_id(project_name):
    username = "admin"
    email = "xiongdaecust@163.com"
    password = "admin123"
    token = get_token(username, email, password)
    headers = {
        'Content-Type':
        'application/json',
        "Authorization":
        f"Token {token}",
        "User-Agent":
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.81 Safari/537.36 Edg/104.0.1293.54"
    }
    rep = requests.get(url="http://192.168.1.28:8080/api/projects",
                       headers=headers)
    projects = json.loads(rep.text)["results"]
    project_ids = {}
    for obj in projects:
        project_ids[obj["name"]] = obj["tasks"]
    return project_ids[project_name]


def get_token(username, email, password):
    request_url = "http://192.168.1.28:8080/api/auth/login"
    params = {"username": username, "email": email, "password": password}
    response = requests.post(request_url, data=params)
    if response.status_code == 200:
        print(
            json.dumps(response.json(),
                       sort_keys=True,
                       indent=4,
                       separators=(', ', ': '),
                       ensure_ascii=False))
        dictRS = dict(response.json())
        return dictRS['key']
    else:
        print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='data studio',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output_path',
                        type=str,
                        default="",
                        help='output_path')
    parser.add_argument('--project_name',
                        type=str,
                        default="",
                        help='project_name')
    parser.add_argument('--ftp_path', type=str, default="", help='ftp_path')
    opt = parser.parse_args()
    project_name = opt.project_name
    output_path = opt.output_path
    ftp_path = opt.ftp_path

    task_ids = get_all_task_id(project_name)
    export_xml(task_ids, output_path)
    ftp = FTPSync('192.168.0.88')
    ftp.login('hy', "88888888")
    ftp.put_dir(local_path=output_path, ftp_path=ftp_path)
