import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_image_with_matplotlib(color_img, title, pos):
    image_RGB = color_img[:, :, ::-1]

    plt.subplot(2, 3, pos)
    plt.imshow(image_RGB)
    plt.title(title)
    plt.axis('off')

def color_quantization(image, k):
    data = np.float32(image).reshape(-1, 3)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)

    result = center[label.flatten()]
    result = result.reshape(image.shape)
    return result



fig = plt.figure(figsize=(12, 6))
plt.suptitle("Color quantization using K-means clustering algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread('nbs.jpeg')

color_3 = color_quantization(image, 3)
color_5 = color_quantization(image, 5)
color_10 = color_quantization(image, 10)
color_20 = color_quantization(image, 20)
color_40 = color_quantization(image, 40)


cv2.imwrite("color_3.png", color_3)
cv2.imwrite("color_5.png", color_5)
cv2.imwrite("color_10.png", color_10)
cv2.imwrite("color_20.png", color_20)
cv2.imwrite("color_40.png", color_40)

# show_image_with_matplotlib(image, "Original Image", 1)
# show_image_with_matplotlib(color_3, "color quantization (k = 3)", 2)
# show_image_with_matplotlib(color_5, "color quantization (k = 5)", 3)
# show_image_with_matplotlib(color_10, "color quantization (k = 10)", 4)
# show_image_with_matplotlib(color_20, "color quantization (k = 20)", 5)
# show_image_with_matplotlib(color_40, "color quantization (k = 40)", 6)


# plt.show()