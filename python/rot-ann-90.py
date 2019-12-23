import json
import numpy as np

json_file = '/home/venom/Documents/ann/test.json'
merged_list = []
rotated = []
with open(json_file,'r') as json_file:
  data = json.load(json_file)
  lanes = data['lanes']
  h_samples = np.arange(160,720,10).tolist()
  for lane in lanes:
    merged_list.append([(lane[i], h_samples[i]) for i in range(0, len(lane)) if lane[i]!=-2])
for one in merged_list:
  # rotated.append(np.rot90(one))
  with open('rotated.json','a') as file:
    json.dump(np.rot90(one).tolist(),file)
  # break

print(rotated)