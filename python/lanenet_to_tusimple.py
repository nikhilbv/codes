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
# --------------------------------------------------------
# python via_to_tusimple.py  --json_file <absolute path to json_file>
# python via_to_tusimple.py  --json_file /aimldl-dat/data-gaze-230120_155433/AIML_Annotation/lnd_poc-130220_114605/images-p1-120220_AT1_via205_120220.json
# --------------------------------------------------------
"""

import argparse
import json
import os
import cv2
import datetime
import common


def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-j','--json_file', type=str, help='The via format json file')
  return parser.parse_args()

def via_to_tusimple(json_file):
  with open(json_file, 'r') as file:
    new_json = []
    timestamp = ("{:%d%m%y_%H%M%S}").format(datetime.datetime.now())
    json_lines = file.readlines()
    line_index = 0

    # Iterate over each image
    while line_index < len(json_lines):
      json_line = json_lines[line_index]
      tusimple = {
        "lanes" : []
      }
      lanes = []
      sample = json.loads(json_line)
      x_axis = sample['x_axis']
      y_axis = sample['y_axis']
      rawfile = sample['image_name']
      for i in range(len(x_axis)):
        lane = []
        lane.append(x_axis[i])
        lane.append(y_axis[i])
        lanes.append(lane)
      tusimple["lanes"] = lanes
      tusimple["raw_file"] = rawfile
      new_json.append(tusimple)   
      line_index += 1     

  json_basepath = common.getBasePath(json_file)
  json_name = json_file.split('/')[-1]
  new_json_name = json_name.split('.')[0]
  with open(json_basepath+'/'+new_json_name+'-'+timestamp+'.json','w') as outfile:
    for items in new_json:
      json.dump(items, outfile)
      outfile.write('\n')
  print("Done!!")
  print("Saved in path -> {}".format(json_basepath+'/'+new_json_name+'-'+timestamp+'.json'))


if __name__ == '__main__':
  args = init_args()
  via_to_tusimple(args.json_file)