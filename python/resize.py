#!/usr/bin/python
from PIL import Image
import os, sys
import shutil
import tqdm

# from_path = "/aimldl-dat/data-gaze/AIML_Database/lnd-011019_165046/images/"
# from_path = "/aimldl-dat/data-gaze/AIML_Database/lnd-101019_103444/images/"
from_path = "/aimldl-dat/data-gaze/AIML_Database/lnd-151019_104101/images/"
print("from_path : {}".format(from_path))
dirs = os.listdir( from_path )
to_path = os.path.join(from_path,'resized')
print("to path : {}".format(to_path))

if os.path.exists(to_path):
    shutil.rmtree(to_path)
os.makedirs(to_path)

# print(dirs)

def resize():
    # for item in dirs:
    for idx,item in tqdm.tqdm(enumerate(dirs), total = len(dirs)):
        # print("image:{}".format(from_path+item))
        if os.path.isfile(from_path+item):
            im = Image.open(from_path+item)
            # print("im:{}".format(im))
            f, e = os.path.splitext(from_path+item)
            # print("f:{}".format(f))
            # print("e:{}".format(e))
            imResize = im.resize((1280,720), Image.ANTIALIAS)
            imResize.save(to_path+'/'+item, 'JPEG', quality=100)

resize()