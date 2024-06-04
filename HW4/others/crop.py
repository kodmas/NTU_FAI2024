import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Data points
X = np.array([[1, 2], [3, 4], [5, 6], [7, 0], [10, 2]])

# Initial centroids
initial_centroids = np.array([[3,4], [5, 6]])

# Define the KMeans model
kmeans = KMeans(n_clusters=2, init=initial_centroids, n_init=1, max_iter=300, random_state=42)

# Fit the model
kmeans.fit(X)

# Get the resulting centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Calculate clustering error
def calculate_clustering_error(X, labels, centroids):
    error = 0
    for i in range(len(X)):
        centroid = centroids[labels[i]]
        error += np.sum((X[i] - centroid) ** 2)
    return error

error = calculate_clustering_error(X, labels, centroids)

# Plotting
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b', 'y', 'c', 'm']
for i in range(len(X)):
    plt.scatter(X[i][0], X[i][1], s=100, color=colors[labels[i]])
plt.scatter(centroids[:, 0], centroids[:, 1], s=300, marker='*', color='black')
plt.title('K-Means Clustering')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.show()

print("Final centroids:\n", centroids)
print("Clustering error:", error)
