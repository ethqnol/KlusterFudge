import time
import numpy as np

try:
    import torch
except ImportError as e:
    print(f"PyTorch not installed. Skipping GPU benchmark. Error: {e}")
    exit(0)

from kluster_fudge import KModes, KModesGPU


def benchmark_gpu():
    # Large dataset
    n_samples = 50000
    n_features = 20
    n_clusters = 5
    n_init = 5

    print(f"Generating data: {n_samples} samples, {n_features} features...")
    np.random.seed(42)
    X = np.random.randint(0, 10, size=(n_samples, n_features))

    # Metrics to test
    metrics = ["ng"]

    for metric in metrics:
        print(f"\n--- Benchmarking Metric: {metric} ---")

        # CPU
        print(f"Running CPU KModes (n_init={n_init})...")
        start = time.time()
        km_cpu = KModes(
            n_clusters=n_clusters, n_init=n_init, dist_metric=metric, random_state=42
        )
        km_cpu.fit(X)
        cpu_time = time.time() - start
        print(f"CPU time: {cpu_time:.4f}s")

        # GPU
        print(f"Running GPU KModes (n_init={n_init})...")
        # Check device
        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        print(f"Using device: {device}")

        start = time.time()
        km_gpu = KModesGPU(
            n_clusters=n_clusters,
            n_init=n_init,
            dist_metric=metric,
            random_state=42,
            device=device,
        )
        km_gpu.fit(X)
        gpu_time = time.time() - start
        print(f"GPU time: {gpu_time:.4f}s")

        print(f"Speedup: {cpu_time / gpu_time:.2f}x")


if __name__ == "__main__":
    benchmark_gpu()
