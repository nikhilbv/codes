import cv2
import skimage

im_1 = cv2.imread('img1.png')
print(im_1.shape)

im_2 = cv2.imread('img2.png')
print(im_2.shape)


im_1a = skimage.io.imread('img1.png')
print(im_1a.shape)


h,w = im_1.shape[:2]
print(h)
print(w)

print(type(im_1))


s = im_2.shape
im_12 = im_1[:720,:1280]
print(im_12.shape)
skimage.io.imsave('blah1.png',im_12)
print(im_2.shape)
im_merge = im_2+im_12
skimage.io.imsave('merge2.png',im_merge)
