import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline


def mainLibraries():
    data = pd.read_csv('clean.csv')

    scaler = StandardScaler()
    dataScale = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    kmeans = KMeans(n_clusters=2)
    pipeline = Pipeline([('pca', pca), ('kmeans', kmeans)])
    pipeline.fit(dataScale)
    labels = pipeline.predict(dataScale)
    components = pipeline.named_steps['pca'].transform(dataScale)
    plt.scatter(components[:, 0], components[:, 1], c=labels)
    plt.xlabel('PCA X')
    plt.ylabel('PCA Y')
    plt.show()

    inertia = kmeans.inertia_

    print(inertia)


mainLibraries()
