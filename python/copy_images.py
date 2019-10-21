__author__ = 'nikhilbv'
__version__ = '1.0'

"""
# Visualize predictions
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by nikhilbv
# --------------------------------------------------------
"""

"""
# Usage
# python copy_images.py  --json_file <json_file>
# python copy_images.py  --json_file pred-7-170919_115403.json
"""

import argparse
import cv2
import json
# import common 
from common import getBasePath,mkdir_p
import shutil
import os


def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--json_file', type=str, help='The predicted json')

  return parser.parse_args()

def copy_images(json_file):
  assert os.path.exists(json_file),'{:s} not exists'.format(json_file)
  print("json_file : {}".format(json_file))
  base_path = getBasePath(json_file)
  # base_path = os.path.join(os.path.dirname(json_file),'')
  print("base_path : {}".format(base_path))
  path = os.path.join(base_path,'test_images')
  print("path : {}".format(path))
  mkdir_p(path)
  with open(json_file,'r') as json_file:
    json_lines = json_file.readlines()
    images = []
    for line_index,val in enumerate(json_lines):
      # print(line_index)
      json_line = json_lines[line_index]
      sample = json.loads(json_line)
      lanes = sample['lanes']
      image = sample['raw_file']
      images.append(image)        
    print(len(images))
    for im in images:
      # print(im)
      shutil.copy(im,path)

if __name__ == '__main__':
    args = init_args()
    copy_images(args.json_file)
