#!/usr/bin/python
from PIL import Image
import os, sys
import shutil
import tqdm

from_path = "/aimldl-dat/samples/lanenet/rotated/"
print("from_path : {}".format(from_path))
dirs = os.listdir( from_path )
to_path = os.path.join(from_path,'rotated-180')
print("to path : {}".format(to_path))

if os.path.exists(to_path):
  shutil.rmtree(to_path)
os.makedirs(to_path)

# print(dirs)

def rotateImages(rotationAmt):
  # for item in dirs:
  for idx,item in tqdm.tqdm(enumerate(dirs), total = len(dirs)):
    # print("image:{}".format(from_path+item))
    if os.path.isfile(from_path+item):
      im = Image.open(from_path+item)
    #   print("im:{}".format(im))
      im.rotate(rotationAmt).save(from_path+item)
      # close the image
      im.close()

# examples of use
# rotateImages(90)
rotateImages(180)
# rotateImages(270)