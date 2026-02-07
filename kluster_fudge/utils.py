import numpy as np
import numpy.typing as npt
from numba import prange, njit


@njit(parallel=True, fastmath=True)
def update_centroids(
    X: npt.NDArray[np.int64], labels: npt.NDArray[np.int64], n_clusters: int
) -> npt.NDArray[np.int64]:
    """
    Compute new centroids by mode for each cluster for each feature.

    Args:
        X: (npt.NDArray[np.int64]) Encoded data array (n_samples, n_features)
        labels: (npt.NDArray[np.int64]) Cluster labels for each sample (n_samples, 1)
        n_clusters: (int) Number of clusters

    Returns:
        new_centroids: (npt.NDArray[np.int64]) Array of new centroids (n_clusters, n_features)
    """

    n_samples, n_features = X.shape
    new_centroids = np.zeros((n_clusters, n_features), dtype=np.int64)

    # parallelize over cols; note that this is thread safe bc each thread writes to diff col of new_centroids
    for j in prange(n_features):
        col = X[:, j]

        # find max val to allocate counts array size
        max_val = -1
        for i in range(n_samples):
            if col[i] > max_val:
                max_val = col[i]

        # allocate counts array
        counts = np.zeros((n_clusters, max_val + 1), dtype=np.int32)

        # count frequencies by point
        for i in range(n_samples):
            cluster_id = labels[i]
            val = col[i]
            counts[cluster_id, val] += 1

        # find mode for each row
        for k in range(n_clusters):
            best_val = 0
            best_count = -1
            for v in range(max_val + 1):
                if counts[k, v] > best_count:
                    best_count = counts[k, v]
                    best_val = v

            new_centroids[k, j] = best_val

    return new_centroids
