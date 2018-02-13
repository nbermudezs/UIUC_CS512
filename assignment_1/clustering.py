from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from spherecluster import SphericalKMeans


class PhraseClustering:
    def run(self, X, n_clusters, distance):
        cluster = SphericalKMeans(n_clusters)
        return cluster.fit_predict(X)
