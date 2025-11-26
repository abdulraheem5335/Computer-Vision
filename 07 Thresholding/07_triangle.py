import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image_with_matplotlib(color_img, title, pos):
    image_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(3, 2, pos)
    plt.imshow(image_RGB)
    plt.title(title)
    plt.axis('off')

def show_hist_with_matplotlib(hist, title, pos, color, t):
    ax = plt.subplot(3, 2, pos)

    plt.xlabel('bins')
    plt.ylabel('no. of pixels')
    plt.xlim([0, 256])
    plt.axvline(x=t, color='m', linestyle='--')
    plt.plot(hist, color)

fig = plt.figure(figsize=(6, 6))
plt.suptitle("Triangle Binarization Algorithm", fontsize=14, fontweight='bold')


image = cv2.imread('leaf-noise.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_gaussian_image = cv2.GaussianBlur(gray_image, (25, 25), 0)


hist_gray = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
hist_gray_gaussian = cv2.calcHist([gray_gaussian_image], [0], None, [256], [0, 256])

ret1, thresh1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
ret2, thresh2 = cv2.threshold(gray_gaussian_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

show_image_with_matplotlib(image, "Original Image", 1)
show_image_with_matplotlib(cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR), "Gray Image", 2)
show_image_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "Thresh Noised Image(Bef. Gaussian)", 4)
show_image_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "Thresh Noised Image(Aft. Gaussian)", 6)

show_hist_with_matplotlib(hist_gray, "Hist Don't Know", 3, 'm', ret1)
show_hist_with_matplotlib(hist_gray_gaussian, "Hist Don't Know", 5, 'm', ret2)
plt.show()