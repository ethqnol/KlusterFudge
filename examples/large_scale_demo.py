import time
import numpy as np
from kluster_fudge import KModes, KModesGPU


def main():
    print("--- Large Scale KModes Demo ---")

    # 1. Generate large dataset
    n_samples = 100000
    n_features = 20
    n_clusters = 5

    print(f"Generating synthetic data: {n_samples} samples, {n_features} features...")
    np.random.seed(42)
    # Random integers as categorical data
    X = np.random.randint(0, 10, size=(n_samples, n_features))

    print("\nStarting benchmarks...")

    # 2. CPU Benchmark
    print("\n-> Running CPU KModes (n_init=5)...")
    start_time = time.time()
    km_cpu = KModes(n_clusters=n_clusters, n_init=5, random_state=42)
    km_cpu.fit(X)
    cpu_duration = time.time() - start_time
    print(f"CPU Time: {cpu_duration:.2f}s")
    print(f"Cost: {km_cpu.cost_:.2f}")

    # 3. GPU Benchmark
    try:
        import torch

        print("\n-> Running GPU KModes (n_init=5)...")

        device = "cpu"
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"

        print(f"Using device: {device}")

        start_time = time.time()
        # Note: GPU initialization might take a moment on first run
        km_gpu = KModesGPU(
            n_clusters=n_clusters, n_init=5, random_state=42, device=device
        )
        km_gpu.fit(X)
        gpu_duration = time.time() - start_time

        print(f"GPU Time: {gpu_duration:.2f}s")
        print(f"Cost: {km_gpu.cost_:.2f}")

        speedup = cpu_duration / gpu_duration
        print(f"\n>>> Speedup: {speedup:.2f}x <<<")

        if np.isclose(km_cpu.cost_, km_gpu.cost_):
            print("(Results match CPU consistency check passed)")
        else:
            print(
                "(Note: Costs differ slightly, likely due to float/random state differences)"
            )

    except ImportError:
        print("PyTorch not installed, skipping GPU benchmark.")


if __name__ == "__main__":
    main()
