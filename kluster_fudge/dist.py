import numpy as np
import numpy.typing as npt
from numba import prange, njit
from enum import Enum


class DistanceMetrics(Enum):
    HAMMING = "hamming"
    JACCARD = "jaccard"
    NG = "ng"


# we have list of centroid which we want to compare our input to
# a point is a list of xyz, in higher dimensions it has n items
# we have a list of centroids we want to compare against a list of targets


# X is data, Centroids is centroids
@njit(parallel=True, fastmath=True)
def hamming(
    X: npt.NDArray[np.int64], centroids: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """
    Compute Hamming distance between X and centroids.

    Args:
        X: (npt.NDArray[np.int64]) Data array (n_samples, n_features)
        centroids: (npt.NDArray[np.int64]) Centroids array (n_clusters, n_features)

    Returns:
        (npt.NDArray[np.float64]) Distance matrix (n_samples, n_clusters)
    """

    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of features")

    rows = X.shape[0]  # NUMBER of rows
    # number of columns of distance matrix = number of centroids
    n_clusters = centroids.shape[0]

    # dist matrix
    distance: npt.NDArray[np.float64] = np.zeros((rows, n_clusters), dtype=np.float64)
    for i in prange(rows):  # iterate from 0 through the number of rows
        for j in range(n_clusters):
            dist = 0
            for a, b in zip(X[i], centroids[j]):
                if a != b:
                    dist += 1
            distance[i][j] = dist
    return distance


@njit(parallel=True, fastmath=True)
def jaccard(
    X: npt.NDArray[np.int64], centroids: npt.NDArray[np.int64]
) -> npt.NDArray[np.float64]:
    """
    Compute Jaccard distance between X and centroids.

    Args:
        X: (npt.NDArray[np.int64]) Data array (n_samples, n_features)
        centroids: (npt.NDArray[np.int64]) Centroids array (n_clusters, n_features)

    Returns:
        (npt.NDArray[np.float64]) Distance matrix (n_samples, n_clusters)
    """

    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of features")

    rows, cols = X.shape
    # number of columns of distance matrix = number of centroids
    n_clusters = centroids.shape[0]

    # dist matrix
    distance: npt.NDArray[np.float64] = np.zeros((rows, n_clusters), dtype=np.float64)
    for i in prange(rows):  # iterate from 0 through the number of rows
        for j in range(n_clusters):
            dist = 0
            for a, b in zip(X[i], centroids[j]):
                if a == b:
                    dist += 1
            distance[i][j] = (
                1 - (dist / (cols * 2 - dist))
            )  # double the num of columns (because a union between both centroids and data points), then subtract the numner of similar elements
    return distance


@njit(parallel=True, fastmath=True)
def ng(
    X: npt.NDArray[np.int64],
    centroids: npt.NDArray[np.int64],
    labels: npt.NDArray[np.int64],
) -> npt.NDArray[np.float64]:
    """
    Compute NG Distance between X and centroids.

    Args:
        X: (npt.NDArray[np.int64]) Data array (n_samples, n_features)
        centroids: (npt.NDArray[np.int64]) Centroids array (n_clusters, n_features)
        labels: (npt.NDArray[np.int64]) Labels array (n_samples,)

    Returns:
        (npt.NDArray[np.float64]) Distance matrix (n_samples, n_clusters)

    """
    if X.shape[1] != centroids.shape[1]:
        raise ValueError("X and centroids must have the same number of features")

    rows, cols = X.shape
    n_clusters = centroids.shape[0]

    # find max value to determine array size for counts
    max_val = 0
    for r in range(rows):
        for c in range(cols):
            if X[r, c] > max_val:
                max_val = X[r, c]

    # allocate counts & cluster sizes
    counts = np.zeros((n_clusters, cols, max_val + 1), dtype=np.int32)
    cluster_sizes = np.zeros(n_clusters, dtype=np.int32)

    # populate counts & cluster sizes
    for r in range(rows):
        cluster_id = labels[r]
        if cluster_id < 0 or cluster_id >= n_clusters:
            continue  # skip invalid labels (shouldn't happen)
        cluster_sizes[cluster_id] += 1
        for c in range(cols):
            val = X[r, c]
            counts[cluster_id, c, val] += 1

    distance = np.zeros((rows, n_clusters), dtype=np.float64)

    for r in prange(rows):
        for i in range(n_clusters):
            dist = 0.0
            if cluster_sizes[i] == 0:
                # If cluster is empty, max distance
                dist = float(cols)
            else:
                for c in range(cols):
                    val = X[r, c]
                    if val <= max_val:
                        freq = counts[i, c, val]
                        prob = freq / cluster_sizes[i]
                        # given by: sum(1 - P(x_j | C))
                        dist += 1.0 - prob
                    else:
                        dist += 1.0  # Should not happen based on max_val logic

            distance[r, i] = dist

    return distance


def distance(
    X: np.ndarray,
    centroids: np.ndarray,
    metric: DistanceMetrics,
    labels: npt.NDArray[np.int64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    Compute distance between X and centroids using the specified metric.

    Args:
        X: (npt.NDArray[np.int64]) Data array (n_samples, n_features)
        centroids: (npt.NDArray[np.int64]) Centroids array (n_clusters, n_features)
        metric: (DistanceMetrics) Distance metric to use
        labels: (npt.NDArray[np.int64] | None) Labels array (n_samples,) for ng dist

    Returns:
        (npt.NDArray[np.float64]) Distance matrix (n_samples, n_clusters)
    """

    if metric == DistanceMetrics.HAMMING:
        return hamming(X, centroids)
    elif metric == DistanceMetrics.JACCARD:
        return jaccard(X, centroids)
    elif metric == DistanceMetrics.NG:
        if labels is None:
            raise ValueError("Labels are required for NG distance")
        return ng(X, centroids, labels)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
