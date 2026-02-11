import pytest
import numpy as np

try:
    import torch
except ImportError:
    torch = None

from kluster_fudge import KModes, KModesGPU


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
@pytest.mark.parametrize("metric", ["hamming", "jaccard", "ng"])
def test_kmodes_gpu_metrics(metric):
    # smoke test metrics
    np.random.seed(42)
    X = np.random.randint(0, 5, size=(200, 10))

    km = KModesGPU(n_clusters=3, n_init=1, dist_metric=metric, random_state=42)
    km.fit(X)

    assert km.labels.shape == (200,)
    assert km.cost_ >= 0


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_kmodes_gpu_consistency():
    # compare gpu vs cpu (small data)
    X = np.random.randint(0, 5, size=(100, 10))

    km_cpu = KModes(n_clusters=3, n_init=5, random_state=42)
    km_cpu.fit(X)

    # use deterministic cao init (rng differs np/torch)
    km_gpu = KModesGPU(n_clusters=3, n_init=1, init_method="cao", random_state=42)
    km_cpu_cao = KModes(n_clusters=3, n_init=1, init_method="cao", random_state=42)

    km_gpu.fit(X)
    km_cpu_cao.fit(X)

    # check identical costs
    assert np.isclose(km_gpu.cost_, km_cpu_cao.cost_)

    # check identical centroids
    np.testing.assert_array_equal(km_gpu.centroids, km_cpu_cao.centroids)


@pytest.mark.skipif(torch is None, reason="PyTorch not installed")
def test_kmodes_gpu_large():
    # smoke test large data
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("No GPU available")

    X = np.random.randint(0, 5, size=(1000, 20))
    km = KModesGPU(n_clusters=5, n_init=2, random_state=42)
    km.fit(X)
    assert km.labels.shape == (1000,)
