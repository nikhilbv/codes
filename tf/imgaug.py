import imageio
import imgaug as ia
import matplotlib.pyplot as plt
# %matplotlib inline

image = imageio.imread("https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png")

print("Original:")

imgplot = plt.imshow(image)
plt.show()