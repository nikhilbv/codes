__author__ = 'nikhilbv'
__version__ = '1.0'

"""
# Remove annotations having small lines
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by nikhilbv
# --------------------------------------------------------
"""

"""
# Usage
# python remove_small_ann.py --json_file <json_file>
# python remove_small_ann.py --json_file /aimldl-dat/data-gaze/AIML_Annotation/images-p1-130919_AT1_via205_130919_tuSimple.json
"""

import argparse
import os
import time
import datetime
import tqdm
import json

import glog as log
import common

def remove_ann(json_file):

  log.info("json_file : {}".format(json_file))
  new_json = []
  ## annotations below this points will be removed
  dist = 30
  
  timestamp = ("{:%d%m%y_%H%M%S}").format(datetime.datetime.now())

  with open(json_file, 'r') as file:
    json_lines = file.readlines()
    res_lane = []

    for line_index,val in tqdm.tqdm(enumerate(json_lines), total = len(json_lines)):
     
      json_line = json_lines[line_index]
      sample = json.loads(json_line)
      lanes = sample['lanes']

      # Number of lanes
      for lane in lanes:
        count = 0
        for lane_id in lane:
          if lane_id == -2:
            continue
          else:
            count = count + 1
          if count > dist:  
            new_json.append(sample)
            break
        break    

  json_basepath = common.getBasePath(json_file)
  json_name = json_file.split('/')[-1]
  new_json_name = json_name.split('.')[0]
  with open(json_basepath+'/'+new_json_name+'-filtered-'+timestamp+'.json','w') as outfile:
    for items in new_json:
      # log.info("items : {}".format(items))
      json.dump(items, outfile)
      outfile.write('\n')
      

def parse_args():
  """Command line parameter parser
  """
  import argparse
  from argparse import RawTextHelpFormatter

  parser = argparse.ArgumentParser(
    description='Get image from server'
    ,formatter_class=RawTextHelpFormatter)

  parser.add_argument('--json_file'
    ,dest='json_file'
    ,help='/path/to/annotation/<directory_OR_annotation_json_file>'
    ,required=True)

  args = parser.parse_args()
  
  return args


if __name__ == '__main__':
  args = parse_args()
  json_file = args.json_file
  remove_ann(json_file)
  