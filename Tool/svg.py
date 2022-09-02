'''
Description: 
Version: 
Author: Leidi
Date: 2021-12-06 15:03:18
LastEditors: Leidi
LastEditTime: 2021-12-22 11:09:25
'''
import sys  
import os  
from PIL import Image  

def convertPixel(r, g, b, a=1):  
    color = '#%02X%02X%02X' % (r, g, b)  
    opacity = a  
    return (color, opacity)  

for r in sys.argv[1:]:  
    root, ext = os.path.splitext(r)  

    image = Image.open(r)  
    mode = image.mode  
    pixels = image.load()  
    width, height = image.size  

    print(image.mode)  

    if 'RGB' in mode:  
        output = '<svg width=%d height=%d viewBox=0 0 %d %d xmlns=http://www.w3.org/2000/svg>' % (width, height, width, height)  

        for r in range(height):  
            for c in range(width):  
                color, opacity = convertPixel(*pixels[c, r])  
                output += '<rect x=%d y=%d width=1 height=1 fill=%s fill-opacity=%s/>' % (c, r, color, opacity)  

        output += '</svg>'  

        with open(root + '.svg', 'w') as f:  
            f.write(output)