#【针对文件夹图片进行处理】python裁剪图片、更改图片尺寸(resize)并保存
import os
import cv2
from PIL import Image
 
def getFileList(dir,Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)
    
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir=os.path.join(dir,s)
            getFileList(newDir, Filelist, ext)
 
    return Filelist
 
org_img_folder='/home/zyq/文档/zyq_demo/zyq_demo_v1/images'
 
# 检索文件
imglist = getFileList(org_img_folder, [], 'jpg')
print('本次执行检索到 '+str(len(imglist))+' 张图像\n')
 
for imgpath in imglist:
    '''
    imgname= os.path.splitext(os.path.basename(imgpath))[0]
    img = cv2.imread(imgpath, cv2.IMREAD_COLOR)
    '''
    srcPath = imgpath
    res = srcPath.split('/')
    dstPath1 = '/home/zyq/文档/zyq_demo/zyq_demo_v1/input/' + res[-1]
    dstPath2 = '/home/zyq/文档/zyq_demo/zyq_demo_v1/dynamic_gt/' + res[-1]

    # 读取图片
    # 原图：6400 * 8800
    img_1 = Image.open(srcPath)

    # 设置裁剪的位置
    crop_box_1 = (0, 0, 6400, 2400)
    crop_box_2 = (0, 2400, 6400, 8800)

    # 裁剪图片并resize
    img_2 = img_1.crop(crop_box_1)
    img_2 = img_2.resize((1280*3, 720*2))
    img_2.save(dstPath1)

    # 裁减图片并resize
    img_3 = img_1.crop(crop_box_2)
    img_3 = img_3.resize((256, 256))
    img_3.save(dstPath2)
