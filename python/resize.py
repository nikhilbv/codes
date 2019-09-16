#!/usr/bin/python
from PIL import Image
import os, sys

path = "/home/nikhil/Documents/images/"
dirs = os.listdir( path )

print(dirs)

def resize():
    for item in dirs:
        print("image:{}".format(path+item))
        if os.path.isfile(path+item):
            im = Image.open(path+item)
            # print("im:{}".format(im))
            f, e = os.path.splitext(path+item)
            # print("f:{}".format(f))
            # print("e:{}".format(e))
            imResize = im.resize((1280,720), Image.ANTIALIAS)
            imResize.save(f + ' resized.jpg', 'JPEG', quality=100)

resize()