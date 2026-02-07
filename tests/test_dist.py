import numpy as np
import pytest
from cluster_fudge.dist import distance, hamming, jaccard, ng, DistanceMetrics


def test_hamming_simple():
    # dataset setup
    X = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.int64)
    centroids = np.array([[0, 0], [1, 1]], dtype=np.int64)

    # expected diffs:
    # x0 vs c0: 0
    # x0 vs c1: 2
    # x1 vs c0: 1
    # ...

    dist = hamming(X, centroids)

    expected = np.array([[0, 2], [1, 1], [2, 0]], dtype=np.float64)

    np.testing.assert_array_equal(dist, expected)


def test_jaccard_simple():
    # dataset setup
    X = np.array([[0, 0]], dtype=np.int64)
    centroids = np.array([[0, 0], [1, 1]], dtype=np.int64)

    # x0 vs c0: match=2, total=2. dist=0
    # x0 vs c1: match=0, total=4. dist=1

    dist = jaccard(X, centroids)
    expected = np.array([[0.0, 1.0]])
    np.testing.assert_array_almost_equal(dist, expected)


def test_ng_simple():
    X = np.array([[0, 0], [0, 1]], dtype=np.int64)
    centroids = np.array([[0, 0]], dtype=np.int64)
    labels = np.array([0, 0], dtype=np.int64)


    dist = ng(X, centroids, labels)
    expected = np.array([[0.5], [0.5]])
    np.testing.assert_array_almost_equal(dist, expected)


def test_ng_requires_labels(encoded_data, centroids):
    # error if labels missing
    with pytest.raises(ValueError):
        distance(encoded_data, centroids, DistanceMetrics.NG, labels=None)


def test_distance_dispatcher(encoded_data, centroids, labels):
    # hamming dispatch
    d_h = distance(encoded_data, centroids, DistanceMetrics.HAMMING)
    assert d_h.shape == (encoded_data.shape[0], centroids.shape[0])

    # ng dispatch
    d_n = distance(encoded_data, centroids, DistanceMetrics.NG, labels=labels)
    assert d_n.shape == (encoded_data.shape[0], centroids.shape[0])


def test_ng_empty_cluster_handling():
    # handle empty cluster gracefully
    X = np.array([[0]], dtype=np.int64)
    centroids = np.array([[0], [1]], dtype=np.int64)
    labels = np.array([0], dtype=np.int64)  # c1 empty

    dist = ng(X, centroids, labels)
    # dist to c1 should be max (1.0 for 1 feature)
    assert dist[0, 1] == 1.0


def test_dimensions_mismatch(encoded_data, centroids):
    # error on dimension mismatch
    bad_centroids = centroids[:, :-1]
    with pytest.raises(ValueError, match="features"):
        hamming(encoded_data, bad_centroids)
