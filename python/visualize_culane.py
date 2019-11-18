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
# python visualize_pred.py --image_path <image_path> --json_file <json_file>
# python visualize_pred.py --image_path /aimldl-cod/practice/nikhil/sample-images/7.jpg --json_file pred-7-170919_115403.json_file
"""

import argparse
import cv2
import json

def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_path', type=str, help='The image to visualize')
  parser.add_argument('--json_file', type=str, help='The predicted json')

  return parser.parse_args()

def convert(string):
  li = list(string.split(" "))
  return li

def visualize(image_path,json_file):
  im = cv2.imread(image_path)
  with open(json_file,'r') as file:
    json_lines = file.readlines()
    line_index = 0
    x = []
    y = []
    while line_index < len(json_lines):
      json_line = json_lines[line_index]
      tmpx = []
      tmpy = []
      el = convert(json_line)
      for i in range(len(el)):
        print("el : {}".format(el[i]))
        print("type : {}".format(type(el[i])))
        try:
          el[i] = int(float(el[i]))
        except ValueError:
          print("Cannot convert")
        if i % 2 != 0:
          tmpy.append(el[i])
        else:
          tmpx.append(el[i])
      x.append(tmpx)
      y.append(tmpy)

      line_index += 1

    for i in x:
      if '\n' in i:
        i.remove('\n')

    for j in y:
      if '\n' in j:
        j.remove('\n')    

    print("x : {}".format(x))
    print("y : {}".format(y))
      
    if len(x) == len(y):
      for i,ele in enumerate(x):
        for i1,ele1 in enumerate(x[i]):
          im = cv2.circle(im,(int(x[i][i1]),int(y[i][i1])),3,(0, 255, 0))
      cv2.imshow("image",im)
      cv2.waitKey()
    else:
      print("false")

if __name__ == '__main__':
  args = init_args()
  visualize(args.image_path, args.json_file)
