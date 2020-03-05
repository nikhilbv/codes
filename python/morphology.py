# import cv2 as cv
# import numpy as np
# img = cv.imread('j.png',0)
# kernel = np.ones((5,5),np.uint8)
# erosion = cv.erode(img,kernel,iterations = 1)


# Python program to demonstrate erosion and 
# dilation of images. 
import cv2 
import numpy as np 
import glog as log

def _connect_components_analysis(image):
    """
    connect components analysis to remove the small components
    :param image:
    :return:
    """
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    return cv2.connectedComponentsWithStats(gray_image, connectivity=8, ltype=cv2.CV_32S)

# Reading the input image 
# img = cv2.imread('/aimldl-dat/logs/testing/morphology/instance/170419_131147_16716_zed_l_030.jpg', 0) 
# img = cv2.imread('/aimldl-dat/logs/testing/morphology/instance/170419_131147_16716_zed_l_030.jpg') 
img = cv2.imread('/aimldl-dat/logs/testing/morphology/instance/170419_130711_16716_zed_l_051.jpg') 


# Taking a matrix of size 5 as the kernel 
kernel = np.ones((5,5), np.uint8) 

# The first parameter is the original image, 
# kernel is the matrix with which image is 
# convolved and third parameter is the number 
# of iterations, which will determine how much 
# you want to erode/dilate a given image.

opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# connect_components_analysis_ret = _connect_components_analysis(image=img)
# log.info("connect_components_analysis_ret : {}".format(connect_components_analysis_ret[1]))

img_erosion = cv2.erode(opening, kernel, iterations=1) 
img_dilation = cv2.dilate(img, kernel, iterations=1) 
edges = cv2.Canny(img,100,200)


cv2.imshow('Input', img) 
cv2.imshow('Opening', opening) 
cv2.imshow('Erosion', img_erosion) 
cv2.imshow('Dilation', img_dilation) 
cv2.imshow('Edges', edges) 
cv2.imwrite('edges.jpg',edges)

cv2.waitKey(0) 
