__author__ = 'nikhilbv'
__version__ = '1.0'

"""
# Visualize tusimple annotation/prediction
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by nikhilbv
# --------------------------------------------------------
"""

"""
# Usage
# --------------------------------------------------------
# python tusimple_visualizer.py  --json_file <absolute path to json_file>
# python tusimple_visualizer.py  --json_file /aimldl-dat/logs/lanenet/evaluate/241019_104542/pred_json/pred-241019_104542_tuSimple.json
# --------------------------------------------------------
"""

import cv2
import numpy as np
import json
import os
from argparse import ArgumentParser as ArgParse

# Color palette for lane visualization
def getcolor(code):
  if code == 0:
    return (0, 255, 0)
  if code == 1:
    return (0, 255, 255)
  if code == 2:
    return (255, 255, 0)
  if code == 3:
    return (255, 0, 0)
  if code == 4:
    return (0, 0, 255)
  if code == 5:
    return (45, 88, 200)
  if code == 6:
    return (213, 22, 224)

def process(json_file):
  
  # Image visualization
  # cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN)
  cv2.namedWindow("image", cv2.WINDOW_NORMAL)
  # cv2.setWindowProperty("image",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
  cv2.setWindowProperty("image",cv2.WINDOW_NORMAL,cv2.WINDOW_FULLSCREEN)
  count = 1

  # Open lane and class ground truth files
  with open(json_file, 'r') as file:
    json_lines = file.readlines()
    line_index = 0

    # Iterate over each image
    while line_index < len(json_lines):
      json_line = json_lines[line_index]
      i = 0
      sample = json.loads(json_line)
      lanes = sample['lanes']
      raw_file = sample['raw_file']

      # # Display image and draw lane
      # while i < len(lanes):
      im = cv2.imread(raw_file)

        # for lane in lanes:
        #   for j in range(0, (len(lane[0])-1)):
        #     point_h_begin = lane[0][j]
        #     point_w_begin = lane[1][j]

        #     cv2.circle(im, (point_w_begin,point_h_begin), 3, getcolor(i))

      for lane in lanes:
        for k,j in enumerate(lane[0]):
          point_h_begin = lane[0][k]
          # print("point_h_begin",point_h_begin)
          point_w_begin = lane[1][k]
          # print("point_w_begin",point_w_begin)

          cv2.circle(im, (int(point_h_begin),int(point_w_begin)), 3, getcolor(i))
          # cv2.circle(im, (point_w_begin,point_h_begin), 3, getcolor(i))

      cv2.imshow('image', im)

      code = cv2.waitKey(0)
      # i+=1

      # Code for navigation
      # If z is pressed exit from the program
      if code == ord('z'):
        closeall(output, file)
        return
      if code == ord('l'):
        line_index += 100
        count += 100
        break                
      if code == ord('k'):
        line_index += 50
        count += 50
        break                
      if code == ord('j'):
        line_index += 20
        count += 20
        break
      if code == ord('h'):
        line_index += 10
        count += 10
        break
      count += 1
      line_index += 1

if __name__=='__main__':
  ap = ArgParse()
  ap.add_argument('--json_file', type=str, default='Predction or annotation json file')
  args = ap.parse_args()
  process(args.json_file)
