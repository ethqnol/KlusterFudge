# KlusterFudge

A Python library for clustering categorical data using the **K-Modes** algorithm. It supports multiple initialization methods and distance metrics, optimized with `numba` for performance.

[Documentation](https://ethqnol.github.io/KlusterFudge/) | [PyPI](https://pypi.org/project/kluster-fudge/)

## Features
- **Algorithms**: K-Modes clustering for categorical data.
- **Distance Metrics**: Hamming, Jaccard, and NG dissimilarity measures.
- **Initialization**: Random, Huang, and Cao methods.
- **Optimization**: Computationally intensive operations are accelerated using `numba`.
- **Integration**: Supports all ArrayLikes (e.g. numpy arrays, pandas DataFrames, lists of lists, etc.)

## Installation

Install `KlusterFudge` via pip:

```bash
pip install kluster-fudge
```

Or install from source:

```bash
git clone https://github.com/ethqnol/KlusterFudge.git
cd KlusterFudge
pip install .
```

## Quick Start

```python
import pandas as pd
from kluster_fudge import KModes

# 1. Load Data
df = pd.DataFrame({
    'color': ['red', 'blue', 'red', 'green', 'blue', 'green'],
    'size': ['small', 'large', 'small', 'large', 'medium', 'medium'],
    'shape': ['circle', 'square', 'circle', 'square', 'triangle', 'triangle']
})

# 2. Initialize Model
model = KModes(
    n_clusters=2, 
    init_method='cao', 
    dist_metric='hamming', 
    random_state=42
)

# 3. Fit and Predict
clusters = model.fit_predict(df)
print("Cluster Labels:", clusters)
```

## Comparisons & Benchmarks

See `benchmarks/benchmark_metrics.py` 

| Metric    | Init  | Time (s) |
|---------|---------|----------|
| jaccard | cao | 0.051574 |
| jaccard | huang | 0.051677 |
| hamming | cao | 0.052091 |
| hamming | huang | 0.052477 |
| ng | huang | 0.055497 |
| ng | cao | 0.055761 |
| jaccard | random | 0.569103 |
| ng | random | 1.325702 |
| hamming | random | 3.131030 |

*(Run on 5000 samples, 20 features, 10 categories)*

## License

MIT License