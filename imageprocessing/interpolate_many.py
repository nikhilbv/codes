import cv2
import numpy
import glob
import os
from skimage.transform import resize

dir = "." # current directory
ext1 = ".png" # whatever extension you want

pathname = os.path.join(dir, "*" + ext1)
images = [cv2.imread(img) for img in glob.glob(pathname)]
# print(images)

height = min(image.shape[0] for image in images)
width = min(image.shape[1] for image in images)
# print(height)
# print(width)

output = numpy.zeros((height,width,3))
i=1
for image in images:
	# output = resize(image (height, width), anti_aliasing=True)
	output = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_CUBIC)
	# fname = "test_"+str(i)+".jpg"
	cv2.imwrite("test"+str(i)+".jpg", output)
	i=i+1


ext2 = ".jpg" # whatever extension you want

pathname = os.path.join(dir, "*" + ext2)
resized_images = [cv2.imread(img) for img in glob.glob(pathname)]
# print(resized_images)
for image in resized_images:
	# print(image)
	result = sum(resized_images)

cv2.imwrite("test4.jpg", result)