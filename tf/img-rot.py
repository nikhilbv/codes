import tensorflow as tf
import random
import numpy as np
from matplotlib import pyplot as plt
import cv2

image_path = '/aimldl-dat/samples/lanenet/7.jpg'

with tf.Session() as sess:

   # im = 
   image = cv2.imread(image_path, cv2.IMREAD_COLOR)
   # image = np.reshape(np.arange(0., 4.), [2, 2, 1])
   print(image.shape)

   # this works
   k = random.randint(0, 3)
   print('k = ' + str(k))

   # this gives an error
   # k = random.randint(0, 3)
   # k = tf.convert_to_tensor(k, dtype=tf.int32)
   # k = tf.Print(k, [k], 'k = ')

   # this gives an error
   # k = tf.random_uniform([], minval=0, maxval=4, dtype=tf.int32)
   # k = tf.Print(k, [k], 'k = ')

   image2 = tf.image.rot90(image, k)
   img2 = sess.run(image2)
   plt.figure
   plt.subplot(121)
   plt.imshow(np.squeeze(image), interpolation='nearest')
   plt.subplot(122)
   plt.imshow(np.squeeze(img2), interpolation='nearest')
   plt.show()