import xml.etree.ElementTree as ET
import math
import numpy as np
import json

def EulerAngles2RotationMatrix(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.基于右手系
    param theta: 1-by-3 list [rx, ry, rz] angle in degree
        theta[0]: roll          绕定轴X转动    
        theta[1]: pitch      绕定轴Y转动
        theta[2]: yaw        绕定轴Z转动  
        
    return:
        YPR角，是ZYX欧拉角，依次 绕定轴XYZ转动[rx, ry, rz]
    
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]
    
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
 
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    return R_x,R_y,R_z


def xodrParser2InertialXYZ():
    tree  = ET.parse('/media/hy/889b0bd5-777a-4ee4-bef4-171d2359de14/dataset/HY_dataset/todo/opendrive2lanelet2/hefei_lukou_73-83(1.4).xodr')
    root=tree.getroot()
    road_planView_objects = {}
    for child in root:
        if child.tag == 'road':
            for road_elem in child:    
                # road_planView_objects = {}
                road_planView_objects['GroundTruth'] = {}
                road_planView_objects['GroundTruth']['SolidDoubleWhite'] = []
                road_planView_objects['GroundTruth']['SolidSingleWhite'] = []
                road_planView_objects['GroundTruth']['DashedSolidWhite']= []
                road_planView_objects['GroundTruth']['DashedSingleWhite']= []
                if road_elem.tag == 'planView':     # 道路参考线 Road Reference Line
                    for planView in road_elem:
                        '''
                            <planView>
                                <geometry s="0.0000000000000000e+0" x="-6.4511999511718750e+2" y="1.6916000366210938e+2" hdg="-6.8449260471157736e-2" length="8.9289090265638470e+2">
                                    <line/>
                                </geometry>
                            </planView>
                        '''
                        '''
                        planView.tag : 'geometry'
                        planView.attrib : dict
                            's': '0.0000000000000000e+0'
                            'x': '-6.4511999511718750e+2'
                            'y': '1.6916000366210938e+2'
                            'hdg': '-6.8449260471157736e-2'
                            'length': '8.9289090265638470e+2'
                        '''
                        road_planView_OriginS = float(planView.attrib['s'])
                        road_planView_OriginX = float(planView.attrib['x'])
                        road_planView_OriginY = float(planView.attrib['y'])
                        planView_EulerAngles = [0, 0, float(planView.attrib['hdg'])]   # ZYX右手系旋转;
                        # 虽然在xodr中uvz坐标系转换的roll在前，但OpenDrive1.6官方文档中Local局部坐标系是依次按照yaw，pitch，roll来旋转的
                        planView_R_x,planView_R_y,planView_R_z = EulerAngles2RotationMatrix(planView_EulerAngles,format='rad')
                        planView_R_sth2xyz =  np.dot(planView_R_x, np.dot(planView_R_y,planView_R_z))
                        
                        road_planView_objects['road_planView_OriginS'] = road_planView_OriginS
                        road_planView_objects['road_planView_OriginX'] = road_planView_OriginX
                        road_planView_objects['road_planView_OriginY'] = road_planView_OriginY
                        road_planView_objects['road_planView_hdg'] = float(planView.attrib['hdg'])
                        road_planView_objects['road_planView_length'] = float(planView.attrib['length'])
                        road_planView_objects['road_planView_R_sth2xyz'] = planView_R_sth2xyz.tolist()
                
                if road_elem.tag == 'objects':
                    '''
                        <objects>
                            <object id="1" name="SolidSingleYellow" s="3.2462018053736426e+2" t="-4.4452044245227057e+1" zOffset="3.0517578125000000e-5" hdg="2.8250365658955405e+0" roll="0.0000000000000000e+0" pitch="0.0000000000000000e+0" orientation="+" type="-1" width="1.7763568394002505e-15" length="4.1558924502528150e+1">
                                <outline>
                                    <cornerLocal u="2.0779463683927361e+1" v="-3.5634620445534892e-6" z="0.0000000000000000e+0"/>
                                    <cornerLocal u="1.5584598121111298e+1" v="-3.5564584948133415e-6" z="0.0000000000000000e+0"/>
                                    <cornerLocal u="1.0389732558295293e+1" v="-3.5494549308623391e-6" z="0.0000000000000000e+0"/>
                                    <cornerLocal u="5.1948669954792877e+0" v="-3.5424513527004819e-6" z="0.0000000000000000e+0"/>
                                    <cornerLocal u="1.4326632822303509e-6" v="-3.5354478029603342e-6" z="0.0000000000000000e+0"/>
                                    <cornerLocal u="-5.1948641301527800e+0" v="-3.5284442390093318e-6" z="0.0000000000000000e+0"/>
                                    <cornerLocal u="-1.0389729692968729e+1" v="-3.5214406750583294e-6" z="0.0000000000000000e+0"/>
                                    <cornerLocal u="-1.5584595255784791e+1" v="-3.5144370968964722e-6" z="0.0000000000000000e+0"/>
                                    <cornerLocal u="-2.0779460818600853e+1" v="-3.5074335471563245e-6" z="0.0000000000000000e+0"/>
                                </outline>
                            </object>
                        </objects>
                    '''
                    for elem_object in road_elem:
                        # 解析并提取有效的.xodr文件信息
                        # print('object', elem_object.attrib['id'], elem_object.attrib['name'], elem_object.attrib['s'], elem_object.attrib['t'], elem_object.attrib['zOffset'],
                        #             elem_object.attrib['hdg'], elem_object.attrib['roll'], elem_object.attrib['pitch'], elem_object.attrib['width'], elem_object.attrib['length'],
                        #             elem_object.attrib['orientation'], elem_object.attrib['type'])
                        if  'SolidSingleYellow' == elem_object.attrib['name']:   #道路标线类别区分
                            pass
                        
                        object_id, object_name = elem_object.attrib['id'], elem_object.attrib['name']
                        
                        object_OriginS = float(elem_object.attrib['s'])
                        object_OriginT = float(elem_object.attrib['t'])
                        object_OriginH = float(elem_object.attrib['zOffset'])
                        object_OriginSTH = [object_OriginS,object_OriginT,object_OriginH]
                        
                        object_hdg = float(elem_object.attrib['hdg'])  # format='rad' 
                        object_roll = float(elem_object.attrib['roll'])
                        object_pitch = float(elem_object.attrib['pitch'])
                        object_EulerAngles = [object_roll, object_pitch, object_hdg]   # ZYX右手系旋转;
                        # 虽然在xodr中uvz坐标系转换的roll在前，但OpenDrive1.6官方文档中Local局部坐标系是依次按照yaw，pitch，roll来旋转的
                        object_R_x,object_R_y,object_R_z = EulerAngles2RotationMatrix(object_EulerAngles,format='rad')
                        object_R_uvz2sth =  np.dot(object_R_x, np.dot(object_R_y,object_R_z))
                        # object_R_uvz2sth =  np.matrix(np.dot(object_R_x, np.dot(object_R_y,object_R_z))).I        #不应该用xyz旋转，不能用求逆后的旋转矩阵
                        object_sthMat ={}
                        
                        for outline in elem_object:
                            cornerLocal_list =[]
                            for cornerLocal in outline:
                                # cornerLocal_list.append([float(cornerLocal.attrib['u']), float(cornerLocal.attrib['v']), float(cornerLocal.attrib['z'])])
                                #直接使用原始数据的u,v最终解析出来的参考线坐标系下的map，存在每个线要素的局部镜像问题，应该旋转180度，所以直接在原始uv坐标操作
                                # 将直接解析出来的(u,v)旋转180°变成(-u, -v)即可
                                cornerLocal_list.append([-float(cornerLocal.attrib['u']), -float(cornerLocal.attrib['v']), float(cornerLocal.attrib['z'])])
                            cornerLocal_list = np.asarray(cornerLocal_list)
                            cornerLocal_uvz = cornerLocal_list.T   ##转置，将(n,3)的uvz点列转为可用于旋转矩阵点积，进行坐标系转换的(3,n)矩阵

                            sth_pointsMat = np.dot(object_R_uvz2sth,cornerLocal_uvz)  #(3,n)
                            sth_points =sth_pointsMat.T - np.asarray(object_OriginSTH,np.float32) #(n,3) = (n,3) - (3,)  STH坐标原点平移
                            # print(sth_points.shape, sth_points)
                            
                            # object_sthMat[elem_object.attrib['id']] = sth_points        #TypeError ：Object of type ndarray is not JSON serializable
                            # object_sthMat[elem_object.attrib['id']] = sth_points.tolist()
                            object_sthMat[object_id] = sth_points.T.tolist()
                        
                        road_planView_objects['GroundTruth'][object_name].append(object_sthMat)
                        
                out_json = open('./hefei_test_XYZ_2.json', 'w')
                json.dump(road_planView_objects, out_json, indent=4)
                out_json.close()
     
    return road_planView_objects

            
            