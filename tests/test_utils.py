import numpy as np
from cluster_fudge.utils import update_centroids


def test_update_centroids():
    # 2 clusters, 2 features
    # c0: [0,0], [0,1], [0,0] -> mode [0, 0]
    # c1: [1,1], [1,1], [2,1] -> mode [1, 1]

    X = np.array([[0, 0], [0, 1], [0, 0], [1, 1], [1, 1], [2, 1]], dtype=np.int64)

    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    n_clusters = 2

    new_centroids = update_centroids(X, labels, n_clusters)

    expected = np.array([[0, 0], [1, 1]], dtype=np.int64)

    np.testing.assert_array_equal(new_centroids, expected)


def test_update_centroids_empty_cluster():
    # handle empty cluster (should return 0-vec)

    X = np.array([[10]], dtype=np.int64)
    labels = np.array([0], dtype=np.int64)  # c1 empty
    n_clusters = 2

    new_centroids = update_centroids(X, labels, n_clusters)

    # c1 should be 0 (default)
    assert new_centroids[1, 0] == 0
