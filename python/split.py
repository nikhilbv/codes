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
import tqdm


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

  res_lanes = {
    '0_lanes':0,
    '1_lanes':0,
    '2_lanes':0,
    '3_lanes':0,
    '4_lanes':0,
    '5_lanes':0,
    '6_lanes':0
  }
  no_of_lanes = {
  "zero_lane" : [],
  "one_lane" : [],
  "two_lane" : [],
  "three_lane" : [],
  "four_lane" : [],
  "five_lane" : [],
  "six_lane" : []
  }
  train = []
  test = []
  
  lines = []
  f = open(json_file,"r")
  for line in f:
    # lines.append(line)
    # print(len(lines))
  # with open(json_file, 'r') as file:
    json_lines = f.readlines()

    # Iterate over each image
    for line_index,val in enumerate(json_lines):
      json_line = json_lines[line_index]
      sample = json.loads(json_line)
      # image_name = sample['raw_file']
      lanes = sample['lanes']
      res_lane = []

      for lane in lanes:
        lane_id_found=False
        for lane_id in lane:
          if lane_id == -2:
            continue
          else:
            lane_id_found=True
            break
        if lane_id_found:
          res_lane.append(lane)        
      if len(res_lane) == 0:
        # res_lanes['0_lanes']=res_lanes['0_lanes']+1
        no_of_lanes["zero_lane"].append(json_lines)
        res_lanes['0_lanes']=len(no_of_lanes["zero_lane"])
      elif len(res_lane) == 1:
        no_of_lanes["one_lane"].append(json_lines)
        res_lanes['1_lanes']=len(no_of_lanes["one_lane"])
      elif len(res_lane) == 2:
        # res_lanes['2_lanes']=res_lanes['2_lanes']+1
        no_of_lanes["two_lane"].append(json_lines)
        res_lanes['2_lanes']=len(no_of_lanes["two_lane"])
      elif len(res_lane) == 3:
        # res_lanes['3_lanes']=res_lanes['3_lanes']+1
        no_of_lanes["three_lane"].append(json_lines)
        res_lanes['3_lanes']=len(no_of_lanes["three_lane"])
      elif len(res_lane) == 4:
        # res_lanes['4_lanes']=res_lanes['4_lanes']+1
        no_of_lanes["four_lane"].append(json_lines)
        res_lanes['4_lanes']=len(no_of_lanes["four_lane"])
      elif len(res_lane) == 5:
        # res_lanes['5_lanes']=res_lanes['5_lanes']+1
        no_of_lanes["five_lane"].append(json_lines)
        res_lanes['5_lanes']=len(no_of_lanes["five_lane"])
      elif len(res_lane) == 6:
        # res_lanes['6_lanes']=res_lanes['6_lanes']+1
        no_of_lanes["six_lane"].append(json_lines)
        res_lanes['6_lanes']=len(no_of_lanes["six_lane"])

  print(res_lanes)
  # print(no_of_lanes["three_lane"])
  
  # with open("no_of_lanes.json",'w') as outfile:
  #   json.dump(no_of_lanes,outfile)
  # # print(no_of_lanes.values()
  print("final command") 
  for i,k in enumerate(no_of_lanes.keys()):
    key = no_of_lanes[k]
    # print(key)

  #   # print("i : {}".format(i))
  #   # print("val : {}".format(val))

    # if i > 2:
      # break
    tmp_train = []
    tmp_test = []
    split_size = int(0.85*len(key))
    # print("split_size :{}".format(split_size))
    tmp_train = key[:split_size]
    print("tmp_train : {}".format(len(tmp_train)))
    tmp_test = key[split_size:]
    print("tmp_test : {}".format(len(tmp_test)))
  # #   # print(tmp_test)
  # #   # if len(tmp_train):
    train = train + tmp_train
  #   # print("train : {}".format(train))
  # #   # if len(tmp_test):
    test.append(tmp_test) 
    # test = test + tmp_test
  # print("len of train : {}".format(train[0]))
    print("len of train : {}".format(len(train)))
    print("len of train : {}".format(type(train)))
    print("len of test : {}".format(len(test)))
    print("len of test : {}".format(type(test)))
    # print("train : {}".format(type(train)))
  # print("test : {}".format(type(test)))
  # print(len(train))
  # print(len(test))
  #   # print(len(val))
  # print(len(no_of_lanes))      
  # print(type(no_of_lanes))      
  # random.shuffle(lines)
  #   # print(line)
  # # print("lines : {}".format(lines))
  # print("len of lines : {}".format(len(lines)))
  # # split_size = int(0.9*len(lines))
  # print("split_size : {}".format(split_size))
  # train = []
  # test = []
  # random.shuffle(train)
  # print("train: {}".format(len(train)))
  # test = lines[split_size:]
  # random.shuffle(test)
  # print("test : {}".format(len(test)))
  # for items in train:
  #   print("items : {}".format(items))
  # print("train : {}".format(train))

  # with open(train_dir+"-tusimple"+".json",'w') as outfile:
    # for item1 in train:
      # print("item1 : {}".format(item1))
      # outfile.write(item1)
      # json.dump(train, outfile)
      # outfile.write('\n')

  # with open(test_dir+"-tusimple"+".json",'w') as outfile:
    # for item2 in test:
      # print("item2 : {}".format(item2))
      # outfile.write(item2)
      # json.dump(test, outfile)
      # outfile.write('\n')

if __name__ == '__main__':
    args = init_args()
    split_train_test(args.json_file)

