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
# python copy_images.py  --json_file <absolute path to json_file>
# python copy_images.py  --json_file pred-7-170919_115403.json
# --------------------------------------------------------
"""

import argparse

def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--json_file1', type=str, help='The predicted json 1')
  parser.add_argument('--json_file2', type=str, help='The predicted json 2')

  return parser.parse_args()

def copy_images(json_file1,json_file2)
  return

if __name__ == '__main__':
    args = init_args()
    copy_images(args.json_file1, args.json_file2)
