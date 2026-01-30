from enum import Enum
import numpy.typing as npt
import numpy as np
from numba import njit, prange
from cluster_fudge.utils import distance, DistanceMetrics

class InitMethod(Enum):
    RAND = 0
    HUANG = 1
    CAO = 2

def init_centroids(X: npt.NDArray[np.float64], n_clusters: int, method: InitMethod, distance_metric: DistanceMetrics = DistanceMetrics.HAMMING) -> np.ndarray:
    if method == InitMethod.RAND:
        n_samples = X.shape[0]
        if n_clusters > n_samples:
                raise ValueError(f"n_clusters ({n_clusters}) cannot exceed n_samples ({n_samples})")

        random_indices = np.random.choice(n_samples, n_clusters, replace=False)
        return X[random_indices]
    elif method == InitMethod.HUANG:
        return _init_centroids_huang(X, n_clusters)
    elif method == InitMethod.CAO:
        return _init_centroids_cao(X, n_clusters, distance_metric)
    else:
        raise ValueError(f"Unknown init method: {method}")



def _freq_sort_categories(X: npt.ArrayLike) -> np.ndarray:
    unique_vals, counts = np.unique(X, return_counts=True)
    sort_indices = np.argsort(counts)
    return unique_vals[sort_indices[::-1]]


def _init_centroids_huang(X: npt.ArrayLike, n_clusters: int) -> np.ndarray:
    X = np.asarray(X)
    _, n_features = X.shape

    if X.ndim != 2:
        raise ValueError("Input array must be 2-dimensional")

    sorted_cols = [_freq_sort_categories(X[:, j]) for j in range(n_features)]

    synthetic_centroids = np.empty((n_clusters, n_features), dtype=X.dtype)

    for i in range(n_clusters):
        for j in range(n_features):
            unique_vals = sorted_cols[j]
            synthetic_centroids[i, j] = unique_vals[i % len(unique_vals)]

    final_centroids = np.empty((n_clusters, n_features), dtype=X.dtype)
    taken_indices = set()

    for i in range(n_clusters):
        best_idx = -1
        distances = np.sum(X != synthetic_centroids[i], axis=1)
        candidate_indices = np.argsort(distances)

        for idx in candidate_indices:
            if idx not in taken_indices:
                best_idx = idx
                break

        if best_idx == -1:
                best_idx = candidate_indices[0]

        taken_indices.add(best_idx)
        final_centroids[i] = X[best_idx]

    return final_centroids


@njit(parallel=True, fastmath=True)
def _compute_X_density(X: npt.ArrayLike) -> np.ndarray:
    X = np.asarray(X)
    U, A = X.shape

    densities = np.zeros(U, dtype=np.float64)

    for a in range(A):
        col = X[:, a]
        max_val = np.max(col)
        counts = np.zeros(max_val + 1, dtype=np.int64)
        for u in range(U):
            counts[col[u]] += 1

        for u in prange(U):
            val = col[u]
            densities[u] += counts[val]

    return densities / (U * A)

def _init_centroids_cao(X: npt.NDArray[np.float64], n_clusters: int, distance_metric: DistanceMetrics) -> np.ndarray:
    densities = _compute_X_density(X)
    centroid_set = [X[np.argmax(densities)]]
    for i, x in enumerate(densities):
        if i in centroid_set:
            continue
        next_centroid =
    return np.array(centroid_set)
