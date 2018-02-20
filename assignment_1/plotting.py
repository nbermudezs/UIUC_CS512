import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD


def plot_clusters(X, y, n_clusters):
    svd = TruncatedSVD(n_components=40)
    reduced = svd.fit_transform(X)
    tsne = TSNE(perplexity=50)
    transformed = tsne.fit_transform(reduced)

    colors = np.random.rand(n_clusters)
    colors = [colors[i] for i in y]
    plt.scatter(transformed[:,0], transformed[:,1], c=colors)
    plt.show()