import numpy as np
import imutils
import cv2

def is_contour_bad(c):
  print("c: {}".format(c))
  # approximate the contour
  # peri = cv2.arcLength(c, True)
  # approx = cv2.approxPolyDP(c, 0.02 * peri, True)
  area = cv2.contourArea(c)
  print("area : {}".format(area))
  # the contour is 'bad' if it is not a rectangle
  return not area > 500

# load the shapes image, convert it to grayscale, and edge edges in
# the image
# image = cv2.imread("/home/nikhil/Documents/Erosion_screenshot_04.03.2020.png")
# image = cv2.imread("/home/nikhil/Documents/Erosion_screenshot_04.03.2020.png")
image = cv2.imread('/aimldl-dat/logs/testing/morphology/instance/170419_130711_16716_zed_l_051.jpg') 
# image = cv2.imread('/aimldl-dat/logs/testing/morphology/instance/170419_121433_16716_zed_l_031.jpg') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
skeleton = imutils.skeletonize(gray, size=(3, 3))
cv2.imshow("Skeleton", skeleton)

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# edged = cv2.Canny(gray, 100, 200)
# cv2.imshow("Original", image)
# find contours in the image and initialize the mask that will be
# used to remove the bad contours
cnts = cv2.findContours(skeleton.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
mask = np.ones(image.shape[:2], dtype="uint8") * 255
# loop over the contours
for c in cnts:
  # if the contour is bad, draw it on the mask
  if is_contour_bad(c):
    cv2.drawContours(mask, [c], -1, 0, -1)
# remove the contours from the image and show the resulting images
image = cv2.bitwise_and(image, image, mask=mask)
# cv2.imshow("Mask", mask)
cv2.imshow("After", image)
cv2.waitKey(0)
cv2.destroyAllWindows() 