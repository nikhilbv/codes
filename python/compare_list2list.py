import json

nopred = '/home/venom/Documents/no_pred.json'
orig = '/home/venom/Documents/images-p1-221119_AT1_via205_221119_tuSimple.json'
l1 = []
l2 = []

with open(nopred,'r') as file:
  data = file.readlines()
  for i in data:
    l1.append(i.rstrip('\n'))
print("len of images with no pred: {}".format(len(l1)))

with open(orig,'r') as orig_file:
  jsonlines = orig_file.readlines()
  for j in jsonlines:
    sample = json.loads(j)
    raw_file = sample['raw_file']
    l2.append(raw_file)
print("len of all images: {}".format(len(l2)))

same = set(l1) & set(l2)
# print(same)
print("len of images with no pred after filter: {}".format(len(same)))
print("Only {} images are filtered".format(len(l1) - len(same)))