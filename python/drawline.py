import numpy as np
import cv2

# Create a black image
img = cv2.imread("/aimldl-cod/practice/nikhil/sample-images/7-resized.jpg",1)

# Draw a diagonal blue line with thickness of 5 px
y = np.arange(240,710,10)
print(y)
# cv2.line(img,(0,50),(1280,50),(255,0,0),2)
for i in y:
  print(i)  
  cv2.line(img,(0,i),(1280,i),(255,0,0),1)
# cv2.namedWindow("image", cv2.WND_PROP_FULLSCREEN) 
cv2.imshow('image',img)
cv2.imwrite('lines_on_image.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()