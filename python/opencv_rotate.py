import cv2
import math

def rotate_image(mat, angle):
  height, width = mat.shape[:2]
  image_center = (width / 2, height / 2)

  rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

  radians = math.radians(angle)
  sin = math.sin(radians)
  cos = math.cos(radians)
  bound_w = int((height * abs(sin)) + (width * abs(cos)))
  bound_h = int((height * abs(cos)) + (width * abs(sin)))

  rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
  rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

  rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
  print("successful")
  return rotated_mat

image = cv2.imread('170419_120936_16716_zed_l_120.jpg', cv2.IMREAD_COLOR)

img_r = rotate_image(image,90) 

cv2.imwrite('rotated.jpg',img_r)
