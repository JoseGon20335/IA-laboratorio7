import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from utils import kmeans


def main():
    data = pd.read_csv('cleanData.csv')
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    pca = PCA(n_components=2)
    pca.fit(data)

    pcaX = pca.transform(data)
    df_pca = pd.DataFrame(data=pcaX, columns=['PC1', 'PC2'])

    label, cent, ine = kmeans(df_pca, k=2, num_iterations=10)

    colores = ['red', 'green', 'blue']
    fig, ax = plt.subplots()
    ax.scatter(df_pca['PC1'], df_pca['PC2'], c=[colores[label]
               for label in label])
    ax.scatter(cent[:, 0], cent[:, 1], marker='*', s=200, c='black')
    plt.show()

    ax.scatter(df_pca['PC1'], df_pca['PC2'], c=[colores[label]
                                                for label in label])
    ax.scatter(cent[:, 0], cent[:, 1], marker='*', s=200, c='black')
    plt.show()
    print(ine)


main()
