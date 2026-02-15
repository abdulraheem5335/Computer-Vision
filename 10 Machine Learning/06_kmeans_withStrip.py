import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections

def show_image_with_matplotlib(color_img, title, pos):
    image_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(2, 3, pos)
    plt.imshow(image_RGB)
    plt.title(title)
    plt.axis('off')

def color_quantization(image, k):

    data = np.float32(image).reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(image.shape)

    counter = collections.Counter(label.flatten())

    total = image.shape[0] * image.shape[1]
    
    desired_height = 70
    desired_height_colors = 50
    desired_width = image.shape[1]
    
    color_distribution = np.ones((desired_height, desired_width, 3), dtype = np.uint8) * 255

    start = 0

    for key, value in counter.items():
        value_normalized = value / total * desired_width

        end = start + value_normalized
        cv2.rectangle(color_distribution, (int(start), 0), (int(end), desired_height_colors), center[key].tolist(), -1)
        start = end
    
    return np.vstack((color_distribution, result))

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Color Quantization with Kmeans Clustering Algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

image = cv2.imread('nbs.jpeg')
color_3 = color_quantization(image, 3)
color_5 = color_quantization(image, 5)
color_10 = color_quantization(image, 10)
color_20 = color_quantization(image, 20)
color_40 = color_quantization(image, 40)

# Plot the images:
show_image_with_matplotlib(image, "original image", 1)
show_image_with_matplotlib(color_3, "color quantization (k = 3)", 2)
show_image_with_matplotlib(color_5, "color quantization (k = 5)", 3)
show_image_with_matplotlib(color_10, "color quantization (k = 10)", 4)
show_image_with_matplotlib(color_20, "color quantization (k = 20)", 5)
show_image_with_matplotlib(color_40, "color quantization (k = 40)", 6)
plt.show()