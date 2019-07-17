import cv2
import numpy
import glob
import os

dir = "." # current directory
ext = ".png" # whatever extension you want

pathname = os.path.join(dir, "*" + ext)
images = [cv2.imread(img) for img in glob.glob(pathname)]

height = sum(image.shape[0] for image in images)
width = max(image.shape[1] for image in images)
output = numpy.zeros((height,width,3))

y = 0
for image in images:
    h,w,d = image.shape
    output[y:y+h,0:w] = image
    y += h
    
cv2.imwrite("test.jpg", output)