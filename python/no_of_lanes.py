import json
import os
import cv2

root_dir = '/home/nikhil/Documents/annotations/'
json_file = 'ann.json'
# root_dir = '/aimldl-dat/data-public/tusimple-sub/train_set/'
#json_file = 'label_data_0313.json'
# json_file = 'label_data_0601.json'
print(root_dir)
print(json_file)

with open(root_dir + json_file, 'r') as file:
  json_lines = file.readlines()
  res_lanes = {'0_lanes':0,'1_lanes':0,'2_lanes':0,'3_lanes':0,'4_lanes':0}
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
  print(res_lanes)
