import ast
import json
import os
import random
import string


def generate_json(data_path, training_path, timestamps):
    # 0.生成scene的token、sample的token、instance的token和annotation的token
    scenes_token = {}
    sample_token = {}
    instances_token = {}
    annotation_token = {}
    scenes = sorted(os.listdir(data_path))
    for scene in scenes:  # 遍历原始文件夹目录，每个子文件夹的名字就是一个scene
        if os.path.isdir(data_path + scene) and scene != ".DS_Store":  # 为每个scene生成一个token
            scenes_token[scene] = "".join(random.sample(
                string.ascii_letters + string.digits, 32))
            samples = sorted(os.listdir(data_path + scene))
            sample_token[scene] = {}
            for sample in samples:  # 遍历每个scene的文件夹目录，每个文件就是一个sample
                # 为每个sample生成一个token
                if not os.path.isdir(data_path + scene + sample) and sample[-1] == 'g':
                    sample_token[scene][sample] = "".join(
                        random.sample(string.ascii_letters + string.digits, 32))
            # 生成instance的token和annotation的token
            instances = {}
            scene_annotation = {}
            with open(data_path + scene + "/vehicle.json", 'r') as f:  # 每个scene的文件夹中存储一个该场景下的全部标注信息
                vehicles = json.load(f)

            for vehicle_samples in vehicles.keys():  # vehicle.json的第一层key是sample的名字
                sample_annotation = {}
                # vehicle.json的第二层key是各个instance的名字，如car1
                for k in vehicles[vehicle_samples].keys():
                    if k not in instances.keys():  # 每个场景中出现的car1都是同一个instance，每个instance只生成一个token
                        instances[k] = "".join(random.sample(
                            string.ascii_letters + string.digits, 32))
                    sample_annotation[k] = "".join(random.sample(
                        string.ascii_letters + string.digits, 32))
                scene_annotation[vehicle_samples] = sample_annotation
            # sample_annotation需要存储全部标注框的信息，有三层key，scene-sample-instance
            annotation_token[scene] = scene_annotation

            # instance_json只有两层key，scene-instance
            instances_token[scene] = instances
    with open(training_path + 'scene_token.json', "w") as json_file:
        json.dump(scenes_token, json_file)
    with open(training_path + 'sample_token.json', "w") as json_file:
        json.dump(sample_token, json_file)
    with open(training_path + 'instances_token.json', "w") as json_file:
        json.dump(instances_token, json_file)
    with open(training_path + 'annotation_token.json', "w") as json_file:
        json.dump(annotation_token, json_file)

    # 1.生成scene.json
    scenes_json = []
    for key in scenes_token.keys():
        token = scenes_token[key]
        log_token = "7e25a2c8ea1f41c5b0da1e69ecfa71a2"
        nbr_samples = len(sample_token[key])
        # sample要按照时间排顺序，需要存储时间上的前后关系
        # 例int_sample_name = {5:000005_3.jpg, 0:"000000_3.jpg}
        if int(key) <= 89:
            int_sample_name = {int(k[:6]): k for k in sample_token[key].keys()}
            # 例sort_sample_name={0, 5}
            sort_sample_name = sorted(int_sample_name.keys())
        elif 89 < int(key) < 116:
            int_sample_name = {
                int(k.split("_")[-1][:-4]): k for k in sample_token[key].keys()}
            sort_sample_name = sorted(int_sample_name.keys())
        else:
            int_sample_name = {
                int(k.split("_")[-2]): k for k in sample_token[key].keys()}
            sort_sample_name = sorted(int_sample_name.keys())
        name = "scene-" + str(key).zfill(4)
        description = "vehicles"
        scenes_json.append({"token": token, "log_token": log_token, "nbr_samples": nbr_samples,
                            "first_sample_token": sample_token[key][int_sample_name[sort_sample_name[0]]],
                            "last_sample_token": sample_token[key][int_sample_name[sort_sample_name[-1]]], "name": name,
                            "description": description})
    with open(training_path + 'scene.json', "w") as json_file:
        json.dump(scenes_json, json_file)

    # 2.生成sample.json
    samples_json = []
    for scene in scenes_token.keys():
        # 例int_sample_name = {5:000005_3.jpg, 0:"000000_3.jpg}
        if int(scene) <= 89:
            int_sample_name = {
                int(k[:6]): k for k in sample_token[scene].keys()}
            # 例sort_sample_name={0, 5}
            sort_sample_name = sorted(int_sample_name.keys())
        elif 89 < int(scene) < 116:
            int_sample_name = {
                int(k.split("_")[-1][:-4]): k for k in sample_token[scene].keys()}
            sort_sample_name = sorted(int_sample_name.keys())
        else:
            int_sample_name = {
                int(k.split("_")[-2]): k for k in sample_token[scene].keys()}
            sort_sample_name = sorted(int_sample_name.keys())

        for i in range(0, len(sort_sample_name)):
            token = sample_token[scene][int_sample_name[sort_sample_name[i]]]
            timestamp = timestamps[scene][int_sample_name[sort_sample_name[i]]]['timestamp']
            sample_prev = ""
            sample_next = ""
            scene_token = scenes_token[scene]
            if i > 0:
                sample_prev = sample_token[scene][int_sample_name[sort_sample_name[i - 1]]]
            if i < len(sort_sample_name) - 1:
                sample_next = sample_token[scene][int_sample_name[sort_sample_name[i + 1]]]
            samples_json.append({"token": token, "timestamp": timestamp, "prev": sample_prev,
                                 "next": sample_next, "scene_token": scene_token})
    with open(training_path + 'sample.json', "w") as json_file:
        json.dump(samples_json, json_file)

    # 3.生成instance.json
    instances_json = []
    for scene_instance in instances_token.keys():
        for instance_name in instances_token[scene_instance].keys():
            token = instances_token[scene_instance][instance_name]
            category_token = "fd69059b62a3469fbaef25340c0eab7f"  # car的token
            if instance_name[:3] == "bus":
                category_token = "fedb11688db84088883945752e480c2c"
            # 遍历这个scene的所有sample中这个instance_name的数量
            instance_annotation = []
            if int(scene_instance) <= 89:
                int_sample_name = {
                    int(k[:6]): k for k in annotation_token[scene_instance].keys()}
                # 例sort_sample_name={0, 5}
                sort_sample_name = sorted(int_sample_name.keys())
            elif 89 < int(scene_instance) < 116:
                int_sample_name = {
                    int(k.split("_")[-1][:-4]): k for k in annotation_token[scene_instance].keys()}
                sort_sample_name = sorted(int_sample_name.keys())
            else:
                int_sample_name = {
                    int(k.split("_")[-2]): k for k in annotation_token[scene_instance].keys()}
                sort_sample_name = sorted(int_sample_name.keys())

            for i in range(0, len(sort_sample_name)):
                for anno_instance in annotation_token[scene_instance][int_sample_name[sort_sample_name[i]]].keys():
                    if anno_instance == instance_name:
                        instance_annotation.append(
                            annotation_token[scene_instance][int_sample_name[sort_sample_name[i]]][anno_instance])
                        # 每个sample中每次instance只会出现一次
                        break
            instances_json.append({"token": token, "category_token": category_token,
                                   "nbr_annotations": len(instance_annotation),
                                   "first_annotation_token": instance_annotation[0],
                                   "last_annotation_token": instance_annotation[-1]})
    with open(training_path + 'instance.json', "w") as json_file:
        json.dump(instances_json, json_file)

    # 4.生成sample_annotation.json
    sample_annotations = []
    for scene_instance in instances_token.keys():  # 遍历一个scene中的所有instance，然后按顺序记录该instance的annotation
        for instance_name in instances_token[scene_instance].keys():
            instance_annotation = []

            if int(scene_instance) <= 89:
                int_sample_name = {
                    int(k[:6]): k for k in annotation_token[scene_instance].keys()}
                # 例sort_sample_name={0, 5}
                sort_sample_name = sorted(int_sample_name.keys())
            elif 89 < int(scene_instance) < 116:
                int_sample_name = {
                    int(k.split("_")[-1][:-4]): k for k in annotation_token[scene_instance].keys()}
                sort_sample_name = sorted(int_sample_name.keys())
            else:
                int_sample_name = {
                    int(k.split("_")[-2]): k for k in annotation_token[scene_instance].keys()}
                sort_sample_name = sorted(int_sample_name.keys())

            sample_squence = []
            for i in range(0, len(sort_sample_name)):
                for anno_instance in annotation_token[scene_instance][int_sample_name[sort_sample_name[i]]].keys():
                    if anno_instance == instance_name:
                        sample_squence.append(
                            int_sample_name[sort_sample_name[i]])
                        instance_annotation.append(
                            annotation_token[scene_instance][int_sample_name[sort_sample_name[i]]][anno_instance])
                        break
            translation = [0, 0, 0]
            size = [0, 0, 0]
            rotation = [0, 0, 0, 0]
            visibility_token = 4
            attribute_tokens = "cb5118da1ab342aa947717dc53544259"
            anno_prev = ""
            anno_next = ""
            for i in range(0, len(instance_annotation)):
                token = instance_annotation[i]
                s_token = sample_token[scene_instance][sample_squence[i]]
                ins_token = instances_token[scene_instance][instance_name]
                if i > 0:
                    anno_prev = instance_annotation[i - 1]
                if i < len(instance_annotation) - 1:
                    anno_next = instance_annotation[i + 1]
                sample_annotations.append({"token": token, "sample_token": s_token, "instance_token": ins_token,
                                           "visibility_token": visibility_token, "attribute_tokens": attribute_tokens,
                                           "translation": translation, "size": size, "rotation": rotation,
                                           "prev": anno_prev,
                                           "next": anno_next, "num_lidar_pts": 0, "num_radar_pts": 0})
    with open(training_path + 'sample_annotation.json', "w") as json_file:
        json.dump(sample_annotations, json_file)

    # 5.生成sensor的token和sensor.json
    sensors_token = {}
    sensors_json = []
    sensors = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
               "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]
    for sensor in sensors:
        code = "".join(random.sample(string.ascii_letters + string.digits, 32))
        sensors_token[sensor] = code
        token = code
        channel = sensor
        modality = "camera"
        sensor_json = {"token": token,
                       "channel": channel, "modality": modality}
        sensors_json.append(sensor_json)
    with open(training_path + 'sensor_token.json', "w") as json_file:
        json.dump(sensors_token, json_file)
    with open(training_path + 'sensor.json', "w") as json_file:
        json.dump(sensors_json, json_file)

    # 6.生成calibrated_sensor的token和calibrated_sensor.json
    calibrated_sensor_token = {}
    calibrated_sensor_json = []
    for scene in scenes_token.keys():
        calibrated_sensor_token[scene] = {}
        for sensor in sensors_token.keys():
            code = "".join(random.sample(
                string.ascii_letters + string.digits, 32))
            calibrated_sensor_token[scene][sensor] = code
            token = code
            sensor_token = sensors_token[sensor]
            translation = []
            rotation = []
            camera_intrinsic = []
            calibrated_sensor = {'token': token, 'sensor_token': sensor_token,
                                 'translation': translation, 'rotation': rotation,
                                 'camera_intrinsic': camera_intrinsic}
            calibrated_sensor_json.append(calibrated_sensor)
    with open(training_path + 'calibrated_sensor_token.json', "w") as json_file:
        json.dump(calibrated_sensor_token, json_file)
    with open(training_path + 'calibrated_sensor.json', "w") as json_file:
        json.dump(calibrated_sensor_json, json_file)

    # 7.生成sample_data的token、sample_data.json和ego_pose.json
    sample_data_token = {}
    sample_data_json = []
    ego_pose_json = []
    camera = ['CAM_BACK', 'CAM_FRONT', 'CAM_BACK_LEFT',
              'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']
    for scene in sample_token.keys():
        sample_data_token[scene] = {}
        for cam in camera:
            sample_data_token[scene][cam] = {}
            for sample in sample_token[scene].keys():
                sample_data_token[scene][cam][sample] = "".join(
                    random.sample(string.ascii_letters + string.digits, 32))

            if int(scene) <= 89:
                int_sample_name = {
                    int(k[:6]): k for k in sample_token[scene].keys()}
                # 例sort_sample_name={0, 5}
                sort_sample_name = sorted(int_sample_name.keys())
            elif 89 < int(scene) < 116:
                int_sample_name = {
                    int(k.split("_")[-1][:-4]): k for k in sample_token[scene].keys()}
                sort_sample_name = sorted(int_sample_name.keys())
            else:
                int_sample_name = {
                    int(k.split("_")[-2]): k for k in sample_token[scene].keys()}
                sort_sample_name = sorted(int_sample_name.keys())

            for i in range(0, len(sort_sample_name)):
                token = sample_data_token[scene][cam][int_sample_name[sort_sample_name[i]]]
                s_token = sample_token[scene][int_sample_name[sort_sample_name[i]]]
                c_sensor_token = calibrated_sensor_token[scene][cam]
                timestamp = timestamps[scene][int_sample_name[sort_sample_name[i]]]['timestamp']
                fileformat = 'jpg'
                is_key_frame = True
                height = 720
                width = 1280
                filename = 'samples/' + cam + '/' + \
                    int_sample_name[sort_sample_name[i]]
                sample_data_prev = ""
                sample_data_next = ""
                if i > 0:
                    sample_data_prev = sample_data_token[scene][cam][int_sample_name[sort_sample_name[i - 1]]]
                if i < len(sort_sample_name) - 1:
                    sample_data_next = sample_data_token[scene][cam][int_sample_name[sort_sample_name[i + 1]]]
                sample_data_json.append({'token': token, 'sample_token': s_token, 'ego_token': token,
                                         'calibrated_sensor_token': c_sensor_token, 'timestamp': timestamp,
                                         'fileformat': fileformat, 'is_key_frame': is_key_frame, 'height': height,
                                         'width': width, 'filename': filename, 'prev': sample_data_prev,
                                         'next': sample_data_next})
                # ego_pose_json
                rotation = []
                translation = []
                ego_pose_json.append(
                    {'token': token, 'timestamp': timestamp, 'rotation': rotation, 'translation': translation})
    with open(training_path + 'sample_data.json', "w") as json_file:
        json.dump(sample_data_json, json_file)
    with open(training_path + 'ego_pose.json', "w") as json_file:
        json.dump(ego_pose_json, json_file)

    # 8.生成hy_sample_annotation.json
    # 通过hy_sample_annotation[scene_token][sample_token][instance_token]获取到四个点的坐标
    hy_sample_annotation = {}
    scenes = sorted(os.listdir(data_path))
    for scene in scenes:
        if os.path.isdir(data_path + scene):
            scene_annotation = {}
            with open(data_path + scene + "/vehicle.json", 'r') as f:
                vehicles = json.load(f)
            for vehicle_samples in vehicles.keys():
                sample_annotation = {}
                for ins in vehicles[vehicle_samples].keys():
                    if int(scene) <= 89:
                        sample_annotation[instances_token[scene]
                                          [ins]] = vehicles[vehicle_samples][ins]
                    else:
                        sample_annotation[instances_token[scene][ins]
                                          ] = vehicles[vehicle_samples][ins]["key_points"]
                scene_annotation[sample_token[scene]
                                 [vehicle_samples]] = sample_annotation
            hy_sample_annotation[scenes_token[scene]] = scene_annotation
    with open(training_path + "hy_sample_annotation.json", "w") as json_file:
        json.dump(hy_sample_annotation, json_file)


if __name__ == '__main__':
    # 定义原始文件夹根目录
    # data_path = '/mnt/sda/yjy/new_hongshan/'
    # training_path = '/mnt/sda/yjy/trainval/trainval/v1.0-trainval/'
    data_path = '/Users/kismet/Desktop/hy/new_training_data/0401/new_hongshan/'
    training_path = '/Users/kismet/Desktop/v1.0-trainval/'

    with open(data_path + 'timestamp.json', 'r') as f:
        data = f.read()
        timestamps = ast.literal_eval(data)

    # 将NUSCENES数据集中的部分json文件copy到训练目录
    # file = ["log.json", "map.json", "category.json", "attribute.json", "visibility.json"]
    # for f in file:
    #     shutil.copy("/home/hy/yu_repo/fiery/mini/mini/v1.0-mini/" + f, training_path + f)

    # 根据公司数据生成NUSCENES格式剩余的json文件
    generate_json(data_path, training_path, timestamps)
