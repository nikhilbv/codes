import glob
import os
import json
from tqdm import tqdm

# src_dir = '/aimldl-dat/data-public/CULane/driver_23_30frame'
# src_dir = '/aimldl-dat/data-public/CULane/driver_161_90frame'
src_dir = '/aimldl-dat/data-public/CULane/driver_182_30frame'


def convert(string):
  li = list(string.split(" "))
  return li

out_json = src_dir.split('/')[-1]
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(src_dir):
  for file in f:
    if '.txt' in file:
      files.append(os.path.join(r, file))

for f in tqdm(files):
  # print(f)
  with open(f,'r') as file:
    json_lines = file.readlines()
    line_index = 0
    x_axis = []
    y_axis = []
    while line_index < len(json_lines):
      json_line = json_lines[line_index]
      tmpx = []
      tmpy = []
      image_name = f.replace('.lines.txt','.jpg')
      # print("image_name : {}".format(image_name))
      el = convert(json_line)
      for i in range(len(el)):
        # print("el : {}".format(el[i]))
        # print("type : {}".format(type(el[i])))
        try:
          el[i] = int(float(el[i]))
        except ValueError:
          print("Cannot convert")
        if i % 2 != 0:
          tmpy.append(el[i])
        else:
          tmpx.append(el[i])
      x_axis.append(tmpx)
      y_axis.append(tmpy)

      line_index += 1

    for i in x_axis:
      if '\n' in i:
        i.remove('\n')

    for j in y_axis:
      if '\n' in j:
        j.remove('\n')

    if len(x_axis): 
      oneOut = {"x_axis" : x_axis,"y_axis":y_axis,"image_name":image_name}
    
    with open(out_json+'.json','a') as outfile:
      json.dump(oneOut,outfile)
      outfile.write('\n')