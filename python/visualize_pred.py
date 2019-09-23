import cv2
import json

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("image", cv2.WINDOW_NORMAL, cv2.WINDOW_FULLSCREEN)

im = cv2.imread("source_image-09-09-2019_10-08-29.png")
with open(pred.json,'r') as file:
  json_lines = file.readlines()
sample = json.loads(json_lines)
x = sample['x_axis']
y = sample['y_axis']
for i in range(len(x)):
  cv2.circle(im,(x[i]),y[i],3,(0, 255, 0))