import cv2
import numpy as np
import matplotlib.pyplot as plt

data = np.float32(np.random.randint(0, 100, (16, 2)))
labels = np.float32(np.random.randint(0, 2, (16, 1)))
sample = np.float32(np.random.randint(0, 100, (1, 2)))


knn = cv2.ml.KNearest_create()

knn.train(data, cv2.ml.ROW_SAMPLE, labels)

k = 3
ret, results, neighbours, dist = knn.findNearest(sample, k)

fig = plt.figure(figsize=(8, 6))
fig.patch.set_facecolor('silver')

red_triangles = data[labels.ravel() == 0]
plt.scatter(red_triangles[:, 0], red_triangles[:, 1], 200, 'r', '^')

blue_squares = data[labels.ravel() == 1]
plt.scatter(blue_squares[:, 0], blue_squares[:, 1], 200, 'b', 's')

plt.scatter(sample[:, 0], sample[:, 1], 200, 'g', 'o')

print("result: {}".format(results))
print("neighbours: {}".format(neighbours))
print("distance: {}".format(dist))

if results[0][0] > 0:
    plt.suptitle("k-NN algorithm: sample green point is classified as blue (k = " + str(k) + ")", fontsize=14,
                 fontweight='bold')
else:
    plt.suptitle("k-NN algorithm: sample green point is classified as red (k = " + str(k) + ")", fontsize=14,
                 fontweight='bold')

plt.show()