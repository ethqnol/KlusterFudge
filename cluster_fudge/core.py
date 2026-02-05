import numpy.typing as npt
import numpy as np
from cluster_fudge.utils import DistanceMetrics, distance


class ClusterFudge:
    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 100,
        dist_metric: DistanceMetrics = DistanceMetrics.HAMMING,
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.dist_metric = dist_metric
        self.centroids = None
        self.labels = None

    def fit(self, X: npt.NDArray[np.float64]) -> None:
        self.centroids = np.random.rand(self.n_clusters, X.shape[1])
        self.labels = np.zeros(X.shape[0], dtype=int)
        for i in range(self.max_iter):  # for the number of iterations, fit, then adjust
            dist = distance(X, self.centroids, self.dist_metric)  # compute distance
            # Assign each point to its closest centroid
            for point in range(len(dist)):
                self.labels[point] = (
                    np.argmin(dist[point])
                    if self.dist_metric != DistanceMetrics.JACCARD
                    else np.argmax(dist[point])
                )

            # For each centroid, get the set of points assigned to it
            for centroid in range(self.n_clusters):
                mask = self.labels == centroid
                points_in_centroid = X[mask]
                centroid_object = []
                for i, var in enumerate(
                    points_in_centroid.items()
                ):  # can iterate over the index and the value (index is assigned to i, value is assigned to var)
                    centroid_object[i] = var.mode()
                self.centroids[centroid] = (
                    centroid_object  # for each var, the item at centroids for
                )
