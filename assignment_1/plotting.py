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


def plot_avg_phrases_curve(data):
    for key, points in data.items():
        data[key] = sorted(points, key=lambda a: a[1])

    for ds, points in data.items():
        single_values = np.unique([point[0] for point in points])

        fig, ax = plt.subplots()
        plt.title('Number phrases for ' + ds)
        plt.xlabel('HIGHLIGHT_MULTI')
        plt.ylabel('# of phrases')
        for single in single_values:
            scatter = np.array([[point[1], point[2]] for point in points if
                                point[0] == single])
            ax.scatter(scatter[:, 0], scatter[:, 1], label=str(single))
            ax.plot(scatter[:, 0], scatter[:, 1])
        ax.legend(title='HIGHLIGHT_SINGLE', ncol=2, loc='upper right',
                  prop={'size': 8})
        plt.show()

        fig, ax = plt.subplots()
        plt.title('Average phrases/sentence for ' + ds)
        plt.xlabel('HIGHLIGHT_MULTI')
        plt.ylabel('Avg phrases/sentence')
        for single in single_values:
            scatter = np.array([[point[1], point[3]] for point in points if point[0] == single])
            ax.scatter(scatter[:,0], scatter[:,1], label=str(single))
            ax.plot(scatter[:, 0], scatter[:, 1])
        ax.legend(title='HIGHLIGHT_SINGLE', ncol=2, loc='upper right',
                  prop={'size': 8})
        plt.show()


def plot_total_phrases_curve(data):
    pass
