import pandas as pd
import numpy as np
from kluster_fudge.core import KModes


def main():
    # 1. Create Synthetic Data
    np.random.seed(42)
    n_samples = 100

    # Generate data with structure:
    # Cluster A: prefer 'red', 'small'
    # Cluster B: prefer 'blue', 'large'

    colors = np.random.choice(
        ["red", "blue", "green"], size=n_samples, p=[0.45, 0.45, 0.1]
    )
    sizes = []
    for c in colors:
        if c == "red":
            sizes.append(np.random.choice(["small", "medium"], p=[0.8, 0.2]))
        elif c == "blue":
            sizes.append(np.random.choice(["large", "medium"], p=[0.8, 0.2]))
        else:
            sizes.append(np.random.choice(["small", "medium", "large"]))

    df = pd.DataFrame(
        {
            "color": colors,
            "size": sizes,
            "texture": np.random.choice(["smooth", "rough"], size=n_samples),  # noise
        }
    )

    print("--- Input Data (First 5 rows) ---")
    print(df.head(), "\n")

    # 2. Initialize and Fit Model (CPU)
    print("Fitting KModes (CPU)...")
    cdf = KModes(n_clusters=2, random_state=42)
    cdf.fit(df)

    print("Labels computed (CPU).")

    # 3. Summarize Clusters
    print("\n--- Cluster Centroids (CPU) ---")
    print(cdf.decoded_centroids)

    # 4. Predict on new data
    print("\n--- Prediction on New Data (CPU) ---")
    new_data = pd.DataFrame(
        [["red", "small", "smooth"], ["blue", "large", "rough"]],
        columns=["color", "size", "texture"],
    )
    preds = cdf.predict(new_data)
    print(f"New Points:\n{new_data}")
    print(f"Predicted Clusters: {preds}")

    # 5. GPU Demonstration
    try:
        import torch
        from kluster_fudge import KModesGPU

        print("\n--- GPU Acceleration Demo ---")
        if torch.backends.mps.is_available():
            print("Running on GPU (MPS)...")
            km_gpu = KModesGPU(n_clusters=2, random_state=42, device="mps")
            km_gpu.fit(df)
            print("Labels computed (GPU).")
            print(f"GPU Cost: {km_gpu.cost_:.4f}")

            # Verify match
            match = np.allclose(km_gpu.cost_, cdf.cost_)
            print(f"Cost matches CPU: {match}")

        elif torch.cuda.is_available():
            print("Running on GPU (CUDA)...")
            km_gpu = KModesGPU(n_clusters=2, random_state=42, device="cuda")
            km_gpu.fit(df)
            print("Labels computed (GPU).")
        else:
            print("No GPU detected, skipping GPU run.")

    except ImportError:
        print("\nPyTorch not installed. Skipping GPU demo.")


if __name__ == "__main__":
    main()
