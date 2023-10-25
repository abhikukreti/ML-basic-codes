import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

n_samples = 300
n_features = 2
n_clusters = 3

X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title("Synthetic Data")
plt.show()

kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(X)

cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

colors = ['b', 'g', 'r']

for i in range(n_clusters):
    plt.scatter(X[cluster_labels == i, 0], X[cluster_labels == i, 1], s=30, c=colors[i], label=f'Cluster {i + 1}')

plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='k', marker='*', label='Cluster Centers')
plt.legend()
plt.title("K-Means Clustering")
plt.show()
