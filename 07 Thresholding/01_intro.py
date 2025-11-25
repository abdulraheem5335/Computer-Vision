import cv2
import numpy as np
import matplotlib.pyplot as plt


def build_sample_image():
    tones = np.arange(start=50, stop=300, step=50)

    result = np.zeros((50, 50, 3), dtype = 'uint8')

    for tone in tones:
        img = np.ones((50, 50, 3), dtype='uint8') * tone
        
        result = np.concatenate((result, img.astype(np.uint8)), axis=1)
    return result

def show_image_with_matplotlib(color_img, title, pos):
    image_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(7, 1, pos)
    plt.imshow(image_RGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(6, 6))
fig.patch.set_facecolor('silver')
plt.suptitle("Thresholding Introduction", fontsize=14, fontweight='bold')

image = build_sample_image()
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image_with_matplotlib(image, "tones [0, 250]", 1)


ret1, thresh1 = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(gray_image, 150, 255, cv2.THRESH_BINARY)
ret5, thresh5 = cv2.threshold(gray_image, 200, 255, cv2.THRESH_BINARY)
ret6, thresh6 = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY)

show_image_with_matplotlib(cv2.cvtColor(thresh1, cv2.COLOR_GRAY2BGR), "threshold = 0", 2)
show_image_with_matplotlib(cv2.cvtColor(thresh2, cv2.COLOR_GRAY2BGR), "threshold = 50", 3)
show_image_with_matplotlib(cv2.cvtColor(thresh3, cv2.COLOR_GRAY2BGR), "threshold = 100", 4)
show_image_with_matplotlib(cv2.cvtColor(thresh4, cv2.COLOR_GRAY2BGR), "threshold = 150", 5)
show_image_with_matplotlib(cv2.cvtColor(thresh5, cv2.COLOR_GRAY2BGR), "threshold = 200", 6)
show_image_with_matplotlib(cv2.cvtColor(thresh6, cv2.COLOR_GRAY2BGR), "threshold = 250", 7)

plt.show()

