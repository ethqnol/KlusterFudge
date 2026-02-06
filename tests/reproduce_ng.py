import numpy as np
from cluster_fudge.core import ClusterFudge


def test_ng_integration():
    # Create synthetic categorical data
    # 50 samples, 4 features, values 'a', 'b', 'c'
    np.random.seed(42)
    data = np.random.choice(["a", "b", "c"], size=(50, 4))

    # Initialize ClusterFudge with NG
    print("Initializing ClusterFudge with dist_metric='ng'...")
    model = ClusterFudge(n_clusters=3, max_iter=5, dist_metric="ng")

    # Fit
    print("Fitting model...")
    try:
        model.fit(data)
        print("Fit successful!")
    except Exception as e:
        print(f"Fit failed with error: {e}")
        raise

    # Check results
    print(f"Centroids shape: {model.centroids.shape}")
    print(f"Labels shape: {model.labels.shape}")
    print(f"Unique labels: {np.unique(model.labels)}")

    assert len(np.unique(model.labels)) > 1, "Model should produce more than 1 cluster"

    print("NG Integration Test Passed!")


if __name__ == "__main__":
    test_ng_integration()
