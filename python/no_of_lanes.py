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
# python show_no_of_lanes.py  --json_file <absolute path to json_file>
# python show_no_of_lanes.py  --json_file pred-7-170919_115403.json
# --------------------------------------------------------
"""

import argparse
import json
import os
import cv2

def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--json_file', type=str, help='The predicted json 1')
  return parser.parse_args()

def show_no_of_lanes(json_file):
  with open(json_file, 'r') as file:
    json_lines = file.readlines()
    res_lanes = {'0_lanes':0,'1_lanes':0,'2_lanes':0,'3_lanes':0,'4_lanes':0,'5_lanes':0,'5_lanes':0}
    for line_index,val in enumerate(json_lines):
      #print(line_index)
      json_line = json_lines[line_index]
      sample = json.loads(json_line)
      lanes = sample['lanes']
      res_lane = []
      #print(len(lanes))
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
      #print(len(res_lane))
      if len(res_lane) == 0:
        res_lanes['0_lanes']=res_lanes['0_lanes']+1
      elif len(res_lane) == 1:
        res_lanes['1_lanes']=res_lanes['1_lanes']+1
      elif len(res_lane) == 2:
        res_lanes['2_lanes']=res_lanes['2_lanes']+1
      elif len(res_lane) == 3:
        res_lanes['3_lanes']=res_lanes['3_lanes']+1
      elif len(res_lane) == 4:
        res_lanes['4_lanes']=res_lanes['4_lanes']+1
      elif len(res_lane) == 5:
        res_lanes['5_lanes']=res_lanes['5_lanes']+1
      elif len(res_lane) == 6:
        res_lanes['6_lanes']=res_lanes['6_lanes']+1
    print(res_lanes)

if __name__ == '__main__':
  args = init_args()
  show_no_of_lanes(args.json_file)