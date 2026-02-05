import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import prange, njit
from enum import Enum



class DistanceMetrics(Enum):
    HAMMING = 'hamming'
    JACCARD = 'jaccard'
    NG = 'ng'

#we have list of centroid which we want to compare our input to
# a point is a list of xyz, in higher dimensions it has n items
# we have a list of centroids we want to compare against a list of targets

# X is data, Centroids is centroids
@njit(parallel=True, fastmath=True)
def hamming(X:np.ndarray, centroids:np.ndarray) -> npt.NDArray[np.float64]: #np.array enforces compiler check on the type you're passing in (must be an np.array)
    assert X.shape[1] == centroids.shape[1] # (taking the first index of shape, should be equal to the first index of centroids. Shape's index of [1] is the number of columns!)the first point must have the same shape (same number of arguments) as the first centroid
    rows = X.shape[0] # NUMBER of rows
    n_clusters = centroids.shape[0] #number of columns of distance matrix = number of centroids
    distance: npt.NDArray[np.float64] = np.zeros((rows, n_clusters), dtype=int)#matrix
    for i in prange(rows): # iterate from 0 through the number of rows
        for j in range(n_clusters):
            dist = 0
            for (a, b) in zip(X[i], centroids[j]):
                if a != b:
                    dist += 1
            distance[i][j] = dist
    return distance


@njit(parallel=True, fastmath=True)
def jaccard(X:np.ndarray, centroids:np.ndarray) -> npt.NDArray[np.float64]: #np.array enforces compiler check on the type you're passing in (must be an np.array)
    assert X.shape[1] == centroids.shape[1] # (taking the first index of shape, should be equal to the first index of centroids. Shape's index of [1] is the number of columns!)the first point must have the same shape (same number of arguments) as the first centroid
    rows, cols = X.shape # NUMBER of rows
    n_clusters = centroids.shape[0] #number of columns of distance matrix = number of centroids
    distance = np.zeros((rows, n_clusters), dtype=int)#matrix
    for i in prange(rows): # iterate from 0 through the number of rows
        for j in range(n_clusters):
            dist = 0
            for (a, b) in zip(X[i], centroids[j]):
                if a == b:
                    dist += 1
            distance[i][j] = dist/(cols * 2 - dist) #double the num of columns (because a union between both centroids and data points), then subtract the numner of similar elements
    return distance



def distance(X:np.ndarray, centroids:np.ndarray, metric:DistanceMetrics) -> npt.NDArray[np.float64]:
    if metric == DistanceMetrics.HAMMING:
        return hamming(X, centroids)
    elif metric == DistanceMetrics.JACCARD:
        return jaccard(X, centroids)
    elif metric == DistanceMetrics.NG:
        return ng(X, centroids)
    else:
        raise ValueError(f"Unsupported distance metric: {metric}")
