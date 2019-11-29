import cv2
import matplotlib.image as mpimg 
import matplotlib.pyplot as plt 

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("image",cv2.WINDOW_NORMAL,cv2.WINDOW_FULLSCREEN)
src = '/aimldl-cod/practice/nikhil/nopred.json'
with open(src,'r') as file:
  json_lines = file.readlines()
  for i in json_lines:
    b = i.rstrip('\n')
    im = cv2.imread(b)
    cv2.imshow("image",im)
    cv2.waitKey(0)
