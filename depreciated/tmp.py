import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate example data for a 3D space
np.random.seed(0)
num_points = 100
data = np.random.multivariate_normal([0, 0, 0], [[3, 1, 0.5], [1, 2, 0.3], [0.5, 0.3, 1]], num_points)

# Perform PCA on the data
mean_centered_data = data - np.mean(data, axis=0)
cov_matrix = np.cov(mean_centered_data, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Order eigenvectors by eigenvalues (largest first)
order = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, order]

# Simulate rotational dynamics (simplified for visualization)
time = np.linspace(0, 10, num_points)
rotation_matrix = np.array([[np.cos(time), -np.sin(time), np.zeros_like(time)],
                            [np.sin(time), np.cos(time), np.zeros_like(time)],
                            [np.zeros_like(time), np.zeros_like(time), np.ones_like(time)]]).T
rotated_data = np.einsum('ijk,kl->ijl', rotation_matrix, mean_centered_data.T).T

# Perform jPCA (simplified for illustration)
# Assume jPCA captures rotations in the plane of the first two principal components
jPC1 = eigenvectors[:, 0] + eigenvectors[:, 1]
jPC2 = eigenvectors[:, 0] - eigenvectors[:, 1]

# Plot the original data and the principal components
fig = plt.figure(figsize=(14, 7))

# 3D plot of the original data
ax = fig.add_subplot(121, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], alpha=0.6, label='Data')
ax.quiver(0, 0, 0, eigenvectors[0, 0], eigenvectors[1, 0], eigenvectors[2, 0], color='r', label='PC1')
ax.quiver(0, 0, 0, eigenvectors[0, 1], eigenvectors[1, 1], eigenvectors[2, 1], color='g', label='PC2')
ax.quiver(0, 0, 0, eigenvectors[0, 2], eigenvectors[1, 2], eigenvectors[2, 2], color='b', label='PC3')
ax.set_title('Original Data with PCA Components')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

# 3D plot of the rotational dynamics (jPCA)
ax = fig.add_subplot(122, projection='3d')
ax.scatter(rotated_data[:, 0], rotated_data[:, 1], rotated_data[:, 2], alpha=0, label='Rotated Data')
ax.quiver(0, 0, 0, jPC1[0], jPC1[1], jPC1[2], color='m', label='jPC1')
ax.quiver(0, 0, 0, jPC2[0], jPC2[1], jPC2[2], color='c', label='jPC2')
ax.set_title('Data with jPCA Components')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()

plt.show()



