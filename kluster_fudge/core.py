from __future__ import annotations
import numpy.typing as npt
import numpy as np
from kluster_fudge.dist import DistanceMetrics, distance
from kluster_fudge.init import init_centroids, InitMethod
from kluster_fudge.utils import update_centroids


class KModes:
    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 100,
        init_method: str = "cao",
        dist_metric: str = "hamming",
        random_state: int = 42,
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
        self.random_state = random_state
        self.cost_ = 0.0

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

    def _encode(self, X: npt.NDArray) -> npt.NDArray[np.int64]:
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

    def _compute_cost(
        self,
        X: npt.NDArray[np.int64],
        centroids: npt.NDArray[np.int64],
        labels: npt.NDArray[np.int64],
    ) -> float:
        """
        Compute the total cost (sum of distances) of the clustering.

        Args:
            X: (npt.NDArray[np.int64]) Data array (n_samples, n_features)
            centroids: (npt.NDArray[np.int64]) Centroids array (n_clusters, n_features)
            labels: (npt.NDArray[np.int64]) Labels array (n_samples,)

        Returns:
            (float) Total cost
        """
        cost = 0.0

        dist_mat = distance(X, centroids, self.dist_metric, labels=labels)

        rows = np.arange(X.shape[0])
        cost = np.sum(dist_mat[rows, labels])

        return cost

    def fit(self, X: npt.ArrayLike) -> None:
        """
        Fit the model to the input data.

        Args:
            X: (npt.ArrayLike) Data array (n_samples, n_features)

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

        best_cost = float("inf")
        best_centroids = None
        best_labels = None

        if self.n_init < 1:
            raise ValueError(f"n_init must be at least 1, got {self.n_init}")

        for init_idx in range(self.n_init):
            # Use a different random state for each initialization
            current_random_state = (
                self.random_state + init_idx if self.random_state is not None else None
            )

            centroids = init_centroids(
                X, self.n_clusters, self.init_method, random_state=current_random_state
            )

            labels = np.zeros(X.shape[0], dtype=int)

            # Iteration loop
            for i in range(self.max_iter):
                # Use Hamming for the first iteration if metric is NG to generate initial labels
                current_metric = self.dist_metric
                if i == 0 and self.dist_metric == DistanceMetrics.NG:
                    current_metric = DistanceMetrics.HAMMING

                dist = distance(
                    X, centroids, current_metric, labels=labels
                )  # compute distance

                labels = np.argmin(dist, axis=1)

                # update centroids
                new_centroids = update_centroids(X, labels, self.n_clusters)

                if np.array_equal(centroids, new_centroids):
                    break

                centroids = new_centroids

            # Compute cost for run and update final label assignments
            final_dist = distance(X, centroids, self.dist_metric, labels=labels)
            labels = np.argmin(final_dist, axis=1)
            cost = self._compute_cost(X, centroids, labels)

            # Update best run if this one is better
            if cost < best_cost:
                best_cost = cost
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels
        self.cost_ = best_cost
        self.decoded_centroids = self._decode(self.centroids)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """
        Predict the cluster labels for the input data.

        Args:
            X: (npt.ArrayLike) Data array (n_samples, n_features)

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

    def fit_predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """
        Fit the model to the input data and return the cluster labels.

        Args:
            X: (npt.ArrayLike) Data array (n_samples, n_features)

        Returns:
            (npt.NDArray[np.int64]) Labels array (n_samples,)
        """
        self.fit(X)
        return self.labels
