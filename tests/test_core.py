import numpy as np
import pytest
import pandas as pd
from kluster_fudge.core import KModes


def test_init_validation():
    # invalid init method raises error
    with pytest.raises(ValueError, match="Unknown init method"):
        KModes(init_method="invalid")

    # invalid dist metric raises error
    with pytest.raises(ValueError, match="Unknown distance metric"):
        KModes(dist_metric="invalid")


def test_fit_predict_hamming(categorical_data):
    # init model
    model = KModes(n_clusters=3, dist_metric="hamming", max_iter=5)
    model.fit(categorical_data)

    # labels exist and match len
    assert model.labels is not None
    assert len(model.labels) == len(categorical_data)

    # check predict output shape
    preds = model.predict(categorical_data)
    assert preds.shape == (len(categorical_data),)


def test_fit_predict_jaccard(categorical_data):
    # jaccard metric runs ok
    model = KModes(n_clusters=3, dist_metric="jaccard", max_iter=5)
    model.fit(categorical_data)
    assert model.labels is not None


def test_fit_predict_ng(categorical_data):
    # ng metric runs ok
    model = KModes(n_clusters=3, dist_metric="ng", max_iter=5)
    model.fit(categorical_data)

    # labels exist
    assert model.labels is not None
    # should have >1 cluster
    assert len(np.unique(model.labels)) > 1


def test_perfect_separation():
    # create distinct groups
    # group 1: all 0s
    g1 = np.zeros((10, 4), dtype=int)
    # group 2: all 1s
    g2 = np.ones((10, 4), dtype=int)

    X = np.vstack([g1, g2])

    # fit model
    model = KModes(n_clusters=2, max_iter=10, init_method="random")
    model.fit(X)

    # first 10 same label
    assert len(np.unique(model.labels[:10])) == 1
    # last 10 same label
    assert len(np.unique(model.labels[10:])) == 1
    # distinct labels between groups
    assert model.labels[0] != model.labels[10]


def test_pandas_input(categorical_data):
    # dataframe input works
    df = pd.DataFrame(categorical_data, columns=["c1", "c2", "c3", "c4"])
    model = KModes(n_clusters=3)
    model.fit(df)

    # is_df flag set
    assert model.is_df
    assert model.labels is not None

    # predict works on df
    preds = model.predict(df)
    assert len(preds) == len(df)


def test_fit_predict_return(categorical_data):
    # fit_predict returns labels
    model = KModes(n_clusters=3)
    labels = model.fit_predict(categorical_data)

    assert labels is not None
    assert np.array_equal(labels, model.labels)


@pytest.mark.parametrize("method", ["random", "huang", "cao"])
def test_init_methods(categorical_data, method):
    # all init methods work
    model = KModes(n_clusters=3, init_method=method, max_iter=2)
    model.fit(categorical_data)
    assert model.centroids is not None


def test_determinism(categorical_data):
    # runs with same seed produce same results
    m1 = KModes(n_clusters=3, random_state=42)
    m1.fit(categorical_data)

    m2 = KModes(n_clusters=3, random_state=42)
    m2.fit(categorical_data)

    assert np.array_equal(m1.labels, m2.labels)
    assert np.array_equal(m1.centroids, m2.centroids)

    # different seed might produce same results on small data, so skipping the negative check


def test_simple_clusters():
    # easy clusters:
    # c0: all 0s
    # c1: all 1s
    # c2: all 2s

    c0 = np.zeros((10, 4), dtype=int)
    c1 = np.ones((10, 4), dtype=int)
    c2 = np.full((10, 4), 2, dtype=int)

    X = np.vstack([c0, c1, c2])

    # shuffle rows to make it slightly harder (but keeping structure)
    # actually let's keep it ordered to easily check ranges, model doesn't care about order

    model = KModes(n_clusters=3, init_method="cao")
    model.fit(X)

    # check that points 0-9 are same, 10-19 same, 20-29 same
    l0 = model.labels[:10]
    l1 = model.labels[10:20]
    l2 = model.labels[20:]

    assert len(np.unique(l0)) == 1
    assert len(np.unique(l1)) == 1
    assert len(np.unique(l2)) == 1

    # distinct groups
    assert l0[0] != l1[0]
    assert l1[0] != l2[0]
    assert l0[0] != l2[0]
