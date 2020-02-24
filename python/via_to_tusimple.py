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
  parser.add_argument('--json_file', type=str, help='The via format json file')
  return parser.parse_args()

def via_to_tusimple(json_file):
  with open(json_file, 'r') as file:
    new_json = []
    timestamp = ("{:%d%m%y_%H%M%S}").format(datetime.datetime.now())
    via = json.load(file)
    for line_index,val in enumerate(via.values()):
      tusimple = {
        "lanes" : []
      }
      lanes = []
      rawfile = val['filename']
      r = val["regions"]
      for j in r:
        lane = []
        lane.append(j["shape_attributes"]["all_points_x"])
        lane.append(j["shape_attributes"]["all_points_y"])
        lanes.append(lane)
      tusimple["lanes"] = lanes
      tusimple["raw_file"] = rawfile
      new_json.append(tusimple)      

  json_basepath = common.getBasePath(json_file)
  json_name = json_file.split('/')[-1]
  new_json_name = json_name.split('.')[0]
  with open(json_basepath+'/'+new_json_name+'-'+timestamp+'.json','w') as outfile:
    for items in new_json:
      json.dump(items, outfile)
      outfile.write('\n')


if __name__ == '__main__':
  args = init_args()
  via_to_tusimple(args.json_file)