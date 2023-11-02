# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# K-Means is a simple learning algorithm for clustering analysis. The goal of K-Means is to divide n observations into k clusters
# where each observation belongs to the cluster with the nearest mean serving as a prototype of the cluster.

# Step 1: Randomly initialize the K cluster centers
def initialize_centers(X, K):
    random_idx = np.random.choice(len(X), K)
    centers = X[random_idx, :]
    return centers

# Step 2: Assign each xi to nearest cluster by calculating its distance to each centroid.
def assign_cluster(X, centers):
    distances = np.sqrt(((X - centers[:, np.newaxis])**2).sum(axis=2))
    cluster_labels = np.argmin(distances, axis=0)
    return cluster_labels

# Step 3: Find new cluster center by taking the average of the assigned points.
def compute_means(X, cluster_labels, K):
    centers = np.array([X[cluster_labels==k].mean(axis=0) for k in range(K)])
    return centers

# Step 4: Repeat Step 2 and 3 until none of the cluster assignments change.
def k_means(X, K, max_iters=100):
    centers = initialize_centers(X, K)
    for i in range(max_iters):
        cluster_labels = assign_cluster(X, centers)
        new_centers = compute_means(X, cluster_labels, K)
        if np.all(centers == new_centers):
            break
        centers = new_centers
    return centers, cluster_labels

# Testing the function with some data
np.random.seed(0)
X = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
              (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
              (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))

plt.scatter(X[:, 0], X[:, 1])
plt.show()

centers, labels = k_means(X, 3)
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], c='r', s=100)
plt.show()


