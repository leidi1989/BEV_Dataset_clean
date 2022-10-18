'''
Description:
Version:
Author: Leidi
Date: 2021-10-27 14:05:30
LastEditors: Leidi
LastEditTime: 2022-10-18 14:48:11
'''
from PIL import Image
Image.MAX_IMAGE_PIXELS = 117613842080

image_path = r'/home/leidi/Desktop/0.png'

image = Image.open(image_path)
image = image.resize((1280, 1280))
image.show()
