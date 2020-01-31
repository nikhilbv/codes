__author__ = 'nikhilbv'
__version__ = '1.0'

"""
# Get images from server
# --------------------------------------------------------
# Copyright (c) 2019 Vidteq India Pvt. Ltd.
# Licensed under [see LICENSE for details]
# Written by nikhilbv
# --------------------------------------------------------
"""

"""
# Usage
# glob.glob("*.png") in images directory
# json reindent in sublime text and save as json
# python placebo.py --json_file <json_file>
"""

import argparse
import os
import sys
import time
import datetime
import tqdm
import glob
import shutil
import json

import glog as log

this_dir = os.path.dirname(__file__)
if this_dir not in sys.path:
  sys.path.append(this_dir)

APP_ROOT_DIR = os.getenv('AI_APP')
if APP_ROOT_DIR not in sys.path:
  sys.path.append(APP_ROOT_DIR)


ANNON_ROOT_DIR = os.path.join(os.getenv('AI_APP'),'annon')
if ANNON_ROOT_DIR not in sys.path:
  sys.path.append(ANNON_ROOT_DIR)

from _annoncfg_ import appcfg
import annonutils 
import common

def get_image(cfg,json_file):

  log.info("json_file : {}".format(json_file))
  to_path = "/home/nikhil/Documents/placebo-images"
  log.info("to_path : {}".format(to_path))


  IMAGE_API = cfg['IMAGE_API']
  USE_IMAGE_API = IMAGE_API['ENABLE']
  SAVE_LOCAL_COPY = True

  
  tic = time.time()
  timestamp = ("{:%d%m%y_%H%M%S}").format(datetime.datetime.now())

  log.info("json_file: {}".format(json_file))
  log.info("to_path: {}".format(to_path))

  save_path = os.path.join(to_path,'lnd-'+timestamp)
  common.mkdir_p(save_path)

  images_save_path = os.path.join(save_path,'images')
  log.info("images_save_path: {}".format(images_save_path))
  common.mkdir_p(images_save_path)

  with open(json_file, 'r') as file:
    json_lines = json.load(file)
    for i,im in tqdm.tqdm(enumerate(json_lines), total = len(json_lines)):
      if USE_IMAGE_API:
        get_img_from_url_success = annonutils.get_image_from_url(IMAGE_API, im, images_save_path, save_local_copy=SAVE_LOCAL_COPY, resize_image=True)  

          
def main(cfg, args):
  json_file = args.json_file
  get_image(cfg,json_file)


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
  main(appcfg, args)

