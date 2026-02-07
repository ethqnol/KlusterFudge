import time
import numpy as np
import pandas as pd
from kluster_fudge.core import KModes


def benchmark():
    # Generate large dataset
    n_samples = 5000
    n_features = 20
    n_clusters = 5
    n_categories = 10

    print(
        f"Generating synthetic data w/ {n_samples} samples, {n_features} features, {n_categories} categories"
    )
    np.random.seed(42)
    X_int = np.random.randint(0, n_categories, size=(n_samples, n_features))
    X = pd.DataFrame(X_int).astype(str).add_prefix("cat_")

    metrics = ["hamming", "jaccard", "ng"]
    init_methods = ["random", "cao", "huang"]
    results = []

    for metric in metrics:
        for init_method in init_methods:
            print(
                f"\nBenchmarking metric: {metric.upper()} w/ init: {init_method.upper()}"
            )

            model = KModes(
                n_clusters=n_clusters, dist_metric=metric, max_iter=10, random_state=42
            )

            start_time = time.time()
            model.fit(X)
            duration = time.time() - start_time

            print(f"  > Time: {duration:.4f} seconds")
            results.append(
                {"Metric": metric, "Init": init_method, "Time (s)": duration}
            )

    print("\n--- Summary ---")
    df_res = pd.DataFrame(results)
    print(df_res.sort_values("Time (s)").reset_index(drop=True))


if __name__ == "__main__":
    benchmark()
