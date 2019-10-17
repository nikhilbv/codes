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
# python split.py --json_file <json_file>
# python split.py --json_file /aimldl-dat/data-gaze/AIML_Database/lnd-011019_165046/images-p1-110919_AT1_via205_270919-tusimple.json
"""

import argparse
import os.path as ops
import json
from common import getBasePath as getbasepath
import random


def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--json_file', type=str, help='Tusimple format json file')

  return parser.parse_args()

def split_train_test(json_file):
  assert ops.exists(json_file), '{:s} not exist'.format(json_file)
  
  base_path = getbasepath(json_file)
  print("base_path : {}".format(json_file))  

  train_dir = ops.join(base_path,"train")
  test_dir = ops.join(base_path,"test")
  print("train_dir : {}".format(train_dir))
  print("test_dir : {}".format(test_dir))
  # os.makedirs(train_dir, exist_ok=True)
  # os.makedirs(test_dir, exist_ok=True)
  
  lines = []
  f = open(json_file,"r")
  for line in f:
    lines.append(line)
  random.shuffle(lines)
    # print(line)
  # print("lines : {}".format(lines))
  print("len of lines : {}".format(len(lines)))
  # split_size = int(0.9*len(lines))
  split_size = int(0.85*len(lines))
  print("split_size : {}".format(split_size))
  train = []
  test = []
  train = lines[:split_size]
  random.shuffle(train)
  print("train: {}".format(len(train)))
  test = lines[split_size:]
  random.shuffle(test)
  print("test : {}".format(len(test)))

  with open(train_dir+"-tusimple"+".json",'w') as outfile:
    for item1 in train:
      # print("item1 : {}".format(item1))
      outfile.write(item1)
      # json.dump(item1, outfile)
      # outfile.write('\n')

  with open(test_dir+"-tusimple"+".json",'w') as outfile:
    for item2 in test:
      # print("item2 : {}".format(item2))
      outfile.write(item2)
      # json.dump(item2, outfile)
      # outfile.write('\n')

if __name__ == '__main__':
    args = init_args()
    split_train_test(args.json_file)

