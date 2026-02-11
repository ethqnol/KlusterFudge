# KlusterFudge

A Python library for clustering categorical data using the **K-Modes** algorithm. It supports multiple initialization methods and distance metrics, optimized with `numba` for performance.

[Documentation](https://ethqnol.github.io/KlusterFudge/) | [PyPI](https://pypi.org/project/kluster-fudge/)

## Features
- **Algorithms**: K-Modes clustering for categorical data
- **GPU Acceleration**: Optional PyTorch-based GPU acceleration via `KModesGPU`
- **Distance Metrics**: Hamming, Jaccard, and NG dissimilarity measures
- **Initialization**: Random, Huang, and Cao methods
- **Optimization**: CPU operations accelerated with `numba`, GPU operations with PyTorch
- **Integration**: Supports all ArrayLikes (numpy arrays, pandas DataFrames, lists, etc.)

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

### GPU Acceleration (Optional)

For GPU acceleration, install PyTorch:

```bash
pip install torch
```

See [PyTorch installation guide](https://pytorch.org/get-started/locally/) for CUDA/ROCm support.


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

## GPU Acceleration

For large datasets, use `KModesGPU` for significant speedups (unless using NG dissimilarity):

```python
from kluster_fudge import KModesGPU
import numpy as np

# Large dataset
X = np.random.randint(0, 10, size=(100000, 20))

# GPU model (auto-detects CUDA/MPS/CPU)
model_gpu = KModesGPU(n_clusters=5, n_init=5, random_state=42)
model_gpu.fit(X)

# 3-6x faster than CPU on typical datasets
```

**Performance**: On a 100k sample dataset, GPU achieves ~3.2x speedup on Apple Silicon (MPS) and up to 6x on CUDA GPUs.

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