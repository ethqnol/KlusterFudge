import pytest
import numpy as np


@pytest.fixture
def categorical_data():
    """Generate synthetic categorical dataset (50, 4)."""
    np.random.seed(42)
    return np.random.choice(["a", "b", "c"], size=(50, 4))


@pytest.fixture
def encoded_data(categorical_data):
    """Return int encoded dataset"""
    X = categorical_data
    X_encoded = np.zeros(X.shape, dtype=int)
    for i in range(X.shape[1]):
        unique_vals, encoded = np.unique(X[:, i], return_inverse=True)
        X_encoded[:, i] = encoded
    return X_encoded


@pytest.fixture
def centroids(encoded_data):
    """Generate random centroids from data"""
    indices = np.random.choice(encoded_data.shape[0], 3, replace=False)
    return encoded_data[indices]


@pytest.fixture
def labels():
    """Generate random labels for clusters"""
    np.random.seed(42)
    return np.random.randint(0, 3, size=50)
