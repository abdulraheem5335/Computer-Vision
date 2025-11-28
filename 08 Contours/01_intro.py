import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_one_contour():

    cnts = [np.array([[[600, 320]], [[563, 460]], [[460, 562]], [[320, 600]], [[180, 563]], [[78, 460]], [[40, 320]], [[77, 180]],
         [[179, 78]], [[319, 40]], [[459, 77]], [[562, 179]]], dtype=np.int32)]

    return cnts

def array_to_tuple(arr):
    return tuple(arr.reshape(1, -1)[0])

def draw_contour_points(img, cnts, color):

    for cnt in cnts:
        print(cnt.shape)
        print(cnt)
        squeeze = np.squeeze(cnt)
        print(squeeze.shape)
        print(squeeze)
        for p in squeeze:
            print(p)
            p = array_to_tuple(p)
            print(p)
            cv2.circle(img, p, 10, color, -1)

    return img

def draw_contour_outline(img, cnts, color, thickness=1):
    for cnt in cnts:
        cv2.drawContours(img, [cnt], 0, color, thickness)


def show_image_with_matplotlib(color_img, title, pos):
    image_RGB = color_img[:, :, ::-1]

    ax = plt.subplot(1, 3, pos)
    plt.imshow(image_RGB)
    plt.title(title)
    plt.axis('off')

fig = plt.figure(figsize=(12, 5))
plt.suptitle("Contours Introduction", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')


canvas = np.zeros((640, 640, 3), dtype='uint8')


contours = get_one_contour()
print("contour shape: '{}'".format(contours[0].shape))
print("'detected' contours: '{}' ".format(len(contours)))


image_contour_points = canvas.copy()
image_contour_outline = canvas.copy()
image_contour_pointsOutline = canvas.copy()


draw_contour_points(image_contour_points, contours, (0, 0, 255))
draw_contour_outline(image_contour_outline, contours, (0, 255, 255), 2)
draw_contour_outline(image_contour_pointsOutline, contours, (0, 255, 255), 3)
draw_contour_points(image_contour_pointsOutline, contours, (0, 0, 255))

show_image_with_matplotlib(image_contour_points, "contour points", 1)
show_image_with_matplotlib(image_contour_outline, "contour outline", 2)
show_image_with_matplotlib(image_contour_pointsOutline, "contour point & outline", 3)

plt.show()
