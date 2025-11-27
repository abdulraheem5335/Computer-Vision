from skimage import img_as_ubyte
from skimage.filters import threshold_otsu
import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image_with_matplotlib(color_img, title, pos):
    image_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 2, pos)
    plt.imshow(image_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib(hist, title, pos, color, t):
    ax = plt.subplot(2, 2, pos)

    plt.xlabel('bins')
    plt.ylabel('no. of pixels')
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color)


fig = plt.figure(figsize=(6, 5))
fig.patch.set_facecolor('silver')
plt.suptitle('Thresholding Color Images', fontsize=14, fontweight='bold')


image = cv2.imread('leaf.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

thresh = threshold_otsu(gray_image) # apply otsu and return threshold value
binary = gray_image > thresh
print("binary image 'dtype': '{}'".format(binary.dtype))
binary = img_as_ubyte(binary)

show_image_with_matplotlib(image, "image", 1)
show_image_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "gray img", 2)
show_hist_with_matplotlib(hist, "grayscale histogram", 3, 'm', thresh)
show_image_with_matplotlib(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), "Otsu's binarization (scikit-image)", 4)

plt.show()