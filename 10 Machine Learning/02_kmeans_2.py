import numpy as np
import matplotlib.pyplot as plt
import cv2

data = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)), np.random.randint(30, 70, (50, 2)), np.random.randint(60, 100, (50, 2)))
))


criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 20, 1.0)

ret, label, center = cv2.kmeans(data, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

A = data[label.ravel() == 0]
print(A.size)
B = data[label.ravel() == 1]
print(B.size)

fig = plt.figure(figsize=(12, 6))
plt.suptitle("K-means Clustering Algorithm", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')


ax = plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c='c')
plt.title('data')


ax = plt.subplot(1, 2, 2)
plt.scatter(A[:, 0], A[:, 1], c='b')
plt.scatter(B[:, 0], B[:, 1], c='g')
plt.scatter(center[:, 0], center[:, 1], s=100, c='m', marker='s')
plt.title("clustered data and centroids (K = 2)")

plt.show()