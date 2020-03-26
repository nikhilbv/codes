import cv2
import numpy as np

# img = cv2.imread('/home/nikhil/Documents/170419_131147_16716_zed_l_030.jpg', 0)
img = cv2.imread('/home/nikhil/Documents/Erosion_screenshot_04.03.2020.png', 0)
img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
# num_labels, labels_im = cv2.connectedComponents(img)
# print("num_labels : {}".format(num_labels))
# print("labels_im : {}".format(labels_im))

# def imshow_components(labels):
# #     # Map component labels to hue val
#     label_hue = np.uint8(179*labels/np.max(labels))
#     blank_ch = 255*np.ones_like(label_hue)
#     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

# #     # cvt to BGR for display
#     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

# #     # set bg label to black
#     labeled_img[label_hue==0] = 0

#     cv2.imshow('labeled.png', labeled_img)
#     cv2.waitKey()

# imshow_components(labels_im)


#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)
#here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
min_size = 150 

#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if sizes[i] <= min_size:
        img2[output == i + 1] = 255

cv2.imshow('new', img2) 

cv2.waitKey(0) 
