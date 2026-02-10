import numpy as np
import matplotlib.pyplot as plt


data = np.float32(np.vstack(
    (np.random.randint(0, 40, (50, 2)), np.random.randint(30, 70, (50, 2)), np.random.randint(60, 100, (50, 2)))
))

print(data)

fig = plt.figure(figsize=(6, 6))
plt.suptitle("K-means Clustering", fontsize=14, fontweight='bold')
fig.patch.set_facecolor('silver')

ax = plt.subplot(1, 1, 1)
plt.scatter(data[:, 0], data[:, 1], c='c')
plt.title("data to be clustered")

plt.show()