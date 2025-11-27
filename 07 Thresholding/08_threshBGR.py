import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_image_with_matplotlib(color_img, title, pos):
    image_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(image_RGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(13, 5))
fig.patch.set_facecolor('silver')
plt.suptitle('Thresholding Color Images', fontsize=14, fontweight='bold')


image = cv2.imread("leaf.png")

ret1, thresh1 = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

(b, g, r) = cv2.split(image)
ret2, thresh2 = cv2.threshold(b, 120, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)
image_bgr = cv2.merge((thresh2, thresh3, thresh4))

show_image_with_matplotlib(image, 'Color Cat', 1)
show_image_with_matplotlib(thresh1, 'threshold (120) BGR Image', 2)
show_image_with_matplotlib(image_bgr, 'threshold (120) each channel and merge', 3)

plt.show()