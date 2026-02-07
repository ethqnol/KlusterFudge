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

    # 2. Initialize and Fit Model
    print("Fitting KModes (k=2)...")
    cdf = KModes(n_clusters=2, random_state=42)
    cdf.fit(df)

    print("Labels computed.")

    # 3. Summarize Clusters
    print("\n--- Cluster Summary Profiles ---")
    summary = cdf.summarize(df)
    print(summary)

    # 4. Predict on new data
    print("\n--- Prediction on New Data ---")
    new_data = pd.DataFrame(
        [["red", "small", "smooth"], ["blue", "large", "rough"]],
        columns=["color", "size", "texture"],
    )
    preds = cdf.predict(new_data)
    print(f"New Points:\n{new_data}")
    print(f"Predicted Clusters: {preds}")


if __name__ == "__main__":
    main()
