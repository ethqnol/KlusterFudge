import numpy.typing as npt
import numpy as np
from cluster_fudge.dist import DistanceMetrics, distance
from cluster_fudge.init import init_centroids, InitMethod
from cluster_fudge.utils import update_centroids


class ClusterFudge:
    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 100,
        init_method: str = "cao",
        dist_metric: str = "hamming",
    ) -> None:
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.dist_metric = dist_metric
        self.centroids = None
        self.labels = None
        self.encodings = []
        self.decoded_centroids = None
        self.is_df = False

        if init_method == "random":
            self.init_method = InitMethod.RAND
        elif init_method == "huang":
            self.init_method = InitMethod.HUANG
        elif init_method == "cao":
            self.init_method = InitMethod.CAO
        else:
            raise ValueError(f"Unknown init method: {init_method}")

        if dist_metric == "hamming":
            self.dist_metric = DistanceMetrics.HAMMING
        elif dist_metric == "jaccard":
            self.dist_metric = DistanceMetrics.JACCARD
        elif dist_metric == "ng":
            self.dist_metric = DistanceMetrics.NG
        else:
            raise ValueError(f"Unknown distance metric: {dist_metric}")

    def _encode(self, X: npt.ndarray) -> npt.NDArray[np.int64]:
        self.encodings = []
        X_encoded = np.zeros(X.shape, dtype=int)

        # every column has its own integer encoding
        # e.g. for mapping ["a", "b", "c"] -> [0, 1, 2], we have ["a", "b", "c", "a", "b"] -> [0, 1, 2, 0, 1]
        for i in range(X.shape[1]):
            unique_vals, encoded = np.unique(X[:, i], return_inverse=True)
            self.encodings.append(unique_vals)
            X_encoded[:, i] = encoded

        return X_encoded

    def _decode(self, centroids_encoded: npt.NDArray[np.int64]) -> npt.NDArray[np.str_]:
        """
        Decode the centroids from integer encoding to original values.

        Args:
            centroids_encoded: (npt.NDArray[np.int64]) Array of encoded centroids (n_clusters, n_features)

        Returns:
            (npt.NDArray[np.str_]) Decoded centroids array (n_clusters, n_features)
        """

        decoded = []
        for i in range(len(self.encodings)):
            col_map = self.encodings[i]
            decoded_col = col_map[centroids_encoded[:, i].astype(int)]
            decoded.append(decoded_col)

        # decoded is a list of arrays representing columns; stack horizontally so each row is a centroid
        return np.array(decoded).T

    def fit(self, X: npt.NDArray[np.float64]) -> None:
        """
        Fit the model to the input data.

        Args:
            X: (npt.NDArray[np.float64]) Data array (n_samples, n_features)

        Returns:
            None
        """

        # check if X is a pandas dataframe, if so, convert to numpy array
        if hasattr(X, "values"):
            X = X.values
            self.is_df = True
        else:
            X = np.asarray(X)

        # encode X into integer array for efficiency
        X = self._encode(X)
        self.centroids = init_centroids(X, self.n_clusters, self.init_method)

        self.labels = np.zeros(X.shape[0], dtype=int)
        for i in range(self.max_iter):  # for the number of iterations, fit, then adjust
            # Use Hamming for the first iteration if metric is NG to generate initial labels
            current_metric = self.dist_metric
            if i == 0 and self.dist_metric == DistanceMetrics.NG:
                current_metric = DistanceMetrics.HAMMING

            dist = distance(
                X, self.centroids, current_metric, labels=self.labels
            )  # compute distance

            # Assign each point to its closest centroid (vectorize using np.argmin on axis 1)
            self.labels = np.argmin(dist, axis=1)

            # update centroids
            self.centroids = update_centroids(X, self.labels, self.n_clusters)

        self.decoded_centroids = self._decode(self.centroids)

    def predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """
        Predict the cluster labels for the input data.

        Args:
            X: (npt.NDArray[np.float64]) Data array (n_samples, n_features)

        Returns:
            (npt.NDArray[np.int64]) Labels array (n_samples,)
        """
        if hasattr(X, "values"):
            X = X.values
        else:
            X = np.asarray(X)

        X = self._encode(X)
        dist = distance(X, self.centroids, self.dist_metric)
        return np.argmin(dist, axis=1)

    def fit_predict(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.int64]:
        """
        Fit the model to the input data and return the cluster labels.

        Args:
            X: (npt.NDArray[np.float64]) Data array (n_samples, n_features)

        Returns:
            (npt.NDArray[np.int64]) Labels array (n_samples,)
        """
        self.fit(X)
        return self.labels
