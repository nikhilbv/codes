import argparse
import cv2
import json

def init_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--image_path', type=str, help='The image to visualize')
  parser.add_argument('--json_file', type=str, help='The predicted json')

  return parser.parse_args()

def visualize(image_path,json_file):
  im = cv2.imread(image_path)
  with open(json_file,'r') as json_file:
    data = json.load(json_file)
    x = data[0]
    y = data[1]

    # if len(x) == len(y):
    #   for i,ele in enumerate(x):
    #     for i1,ele1 in enumerate(x[i]):
    #       im = cv2.circle(im,(int(x[i][i1]),int(y[i][i1])),3,(0, 255, 0))
    #   cv2.imshow("image",im)
    #   cv2.waitKey()
    # else:
    #   print("false")

    if len(x) == len(y):
      for i,ele in enumerate(x):
        im = cv2.circle(im,(int(x[i]),int(y[i])),3,(0, 255, 0))
      cv2.imshow("image",im)
      cv2.waitKey()
    else:
      print("false")


if __name__ == '__main__':
  args = init_args()
  visualize(args.image_path, args.json_file)
