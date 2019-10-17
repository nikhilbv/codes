import json
import os
import cv2

# root_dir = '/home/nikhil/Documents/annotations/'
# json_file = 'ann.json'
root_dir = '/aimldl-dat/data-public/tusimple/train_set/'
# root_dir = '/aimldl-dat/data-public/tusimple/'
# json_file = 'label_data_0313.json'
json_file = 'label_data_0531.json'
# json_file = 'label_data_0601.json'
# json_file = 'test_label.json'
# json_file = 'test_baseline.json'
# print(root_dir)
# print(json_file)

with open(root_dir + json_file, 'r') as file:
  json_lines = file.readlines()
  res_lanes = {'0_lanes':0,'1_lanes':0,'2_lanes':0,'3_lanes':0,'4_lanes':0,'5_lanes':0,'6_lanes':0}
  for line_index,val in enumerate(json_lines):
    # print(line_index)
    json_line = json_lines[line_index]
    sample = json.loads(json_line)
    # print("sample :{}".format(sample))
    lanes = sample['lanes']
    # print("lanes :{}".format(lanes))
    # image = sample['raw_file']

    res_lane = []
  #   #print(len(lanes))
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
      # print(json_line)
      res_lanes['1_lanes']=res_lanes['1_lanes']+1
    elif len(res_lane) == 2:
      # print(json_line)
      res_lanes['2_lanes']=res_lanes['2_lanes']+1
    elif len(res_lane) == 3:
      print(json_line)
      res_lanes['3_lanes']=res_lanes['3_lanes']+1
    elif len(res_lane) == 4:
      res_lanes['4_lanes']=res_lanes['4_lanes']+1
    elif len(res_lane) == 4:
      res_lanes['5_lanes']=res_lanes['5_lanes']+1
    elif len(res_lane) == 4:
      res_lanes['6_lanes']=res_lanes['6_lanes']+1
  print(res_lanes)
