import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def kmeans(df, k, num_iterations):
    centroids = df.sample(k).values
    for i in range(num_iterations):
        distances = np.sqrt(
            ((df.values - centroids[:, np.newaxis])**2).sum(axis=2))
        labels = np.argmin(distances, axis=0)
        for j in range(k):
            centroids[j] = np.mean(df[labels == j], axis=0)
    inertia = ((df.values - centroids[:, np.newaxis])**2).sum()
    return labels, centroids, inertia
