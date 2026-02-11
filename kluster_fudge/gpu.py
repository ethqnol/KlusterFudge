from __future__ import annotations
import numpy as np
import numpy.typing as npt
import warnings

try:
    import torch
except ImportError:
    torch = None

from kluster_fudge.core import KModes


class KModesGPU(KModes):
    def __init__(
        self,
        n_clusters: int = 8,
        n_init: int = 10,
        max_iter: int = 100,
        init_method: str = "cao",
        dist_metric: str = "hamming",
        random_state: int = 42,
        device: str | None = None,
    ) -> None:
        super().__init__(
            n_clusters=n_clusters,
            n_init=n_init,
            max_iter=max_iter,
            init_method=init_method,
            dist_metric=dist_metric,
            random_state=random_state,
        )
        if torch is None:
            raise ImportError(
                "PyTorch is required for KModesGPU. Please install it with `pip install torch`."
            )

        self.device = device
        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
                warnings.warn(
                    "No GPU detected. KModesGPU will run on CPU, which might be slower than KModes (Numba)."
                )

    def fit(self, X: npt.ArrayLike) -> None:
        """
        Fit the model to the input data using GPU acceleration.

        Args:
            X: (npt.ArrayLike) Input data, array-like

        Returns:
            None
        """
        # check if X is a pandas dataframe
        if hasattr(X, "values"):
            X = X.values
            self.is_df = True
        else:
            X = np.asarray(X)

        # cpu enc
        X_encoded_cpu = self._encode(X)
        X_gpu = torch.from_numpy(X_encoded_cpu).to(self.device)

        # check int type
        n_samples, n_features = X_gpu.shape

        best_cost = float("inf")
        best_centroids = None
        best_labels = None

        if self.n_init < 1:
            raise ValueError(f"n_init must be at least 1, got {self.n_init}")

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        for init_idx in range(self.n_init):
            current_seed = (
                self.random_state + init_idx if self.random_state is not None else None
            )

            # Oslice the numpy array for initialization on CPU
            from kluster_fudge.init import init_centroids

            centroids_cpu = init_centroids(
                X_encoded_cpu,
                self.n_clusters,
                self.init_method,
                random_state=current_seed,
            )

            centroids = torch.from_numpy(centroids_cpu).to(self.device)
            labels = torch.zeros(n_samples, dtype=torch.long, device=self.device)

            for i in range(self.max_iter):
                # dist calc (expand dims)

                # normalize metric
                metric_str = self.dist_metric
                if hasattr(metric_str, "value"):
                    metric_str = metric_str.value

                if metric_str == "hamming":
                    dists = self._hamming(X_gpu, centroids)
                elif metric_str == "jaccard":
                    dists = self._jaccard(X_gpu, centroids)
                elif metric_str == "ng":
                    if i == 0:
                        # 1st iter: hamming
                        dists = self._hamming(X_gpu, centroids)
                    else:
                        dists = self._ng(X_gpu, centroids, labels, n_features)
                else:
                    # fallback
                    raise ValueError(f"Unsupported metric: {self.dist_metric}")

                # assign lbls
                new_labels = torch.argmin(dists, dim=1)

                # converged?
                if torch.equal(labels, new_labels) and i > 0:
                    break

                labels = new_labels

                # update centroids (vec w/ bincount)

                max_val = int(X_gpu.max().item())
                if max_val < 0:
                    max_val = 0
                val_offset = max_val + 1

                counts = self._compute_counts(
                    X_gpu, labels, self.n_clusters, n_features, val_offset
                )

                # reshape (F, K, V)
                counts_reshaped = counts.view(n_features, self.n_clusters, val_offset)

                # mode via argmax
                new_centroids_t = counts_reshaped.argmax(dim=2)
                new_centroids = new_centroids_t.t()  # (K, F)

                # handle empty
                cluster_counts = torch.bincount(labels, minlength=self.n_clusters)
                empty_clusters = cluster_counts == 0

                if empty_clusters.any():
                    new_centroids[empty_clusters] = centroids[empty_clusters]

                centroids = new_centroids

            # final cost
            metric_str = self.dist_metric
            if hasattr(metric_str, "value"):
                metric_str = metric_str.value

            if metric_str == "hamming":
                final_dists = self._hamming(X_gpu, centroids)
            elif metric_str == "jaccard":
                final_dists = self._jaccard(X_gpu, centroids)
            elif metric_str == "ng":
                final_dists = self._ng(X_gpu, centroids, labels, n_features)
            else:
                final_dists = self._hamming(X_gpu, centroids)

            row_idx = torch.arange(n_samples, device=self.device)
            min_dists = final_dists[row_idx, labels]
            cost = min_dists.sum().item()

            if cost < best_cost:
                best_cost = cost
                best_centroids = centroids.clone()
                best_labels = labels.clone()

        self.centroids = best_centroids.cpu().numpy()
        self.labels = best_labels.cpu().numpy()
        self.cost_ = best_cost
        self.decoded_centroids = self._decode(self.centroids)

    def _compute_counts(
        self,
        X: torch.Tensor,
        labels: torch.Tensor,
        n_clusters: int,
        n_features: int,
        val_offset: int,
    ) -> torch.Tensor:
        """
        freq counts (k, f, v)

        Args:
            X: (torch.Tensor) Input data (n_samples, n_features)
            labels: (torch.Tensor) Cluster labels (n_samples)
            n_clusters: (int) Number of clusters
            n_features: (int) Number of features
            val_offset: (int) Value offset for flat indexing

        Returns:
            (torch.Tensor) Flattened counts (n_features * n_clusters * val_offset)
        """
        n_samples = X.shape[0]

        # labels: (N) -> (N, F)
        labels_expanded = labels.unsqueeze(1).expand(-1, n_features)

        # feature indices
        feature_indices = (
            torch.arange(n_features, device=self.device)
            .unsqueeze(0)
            .expand(n_samples, -1)
        )

        # flat idx
        flat_indices = (
            feature_indices * (n_clusters * val_offset)
            + labels_expanded * val_offset
            + X
        ).view(-1)

        num_bins = n_features * n_clusters * val_offset
        return torch.bincount(flat_indices, minlength=num_bins)

    def _hamming(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        hamming dist

        Args:
            X: (torch.Tensor) Input data (n_samples, n_features)
            centroids: (torch.Tensor) Centroids (n_clusters, n_features)

        Returns:
            (torch.Tensor) Distance matrix (n_samples, n_clusters)
        """
        # (N, K)
        return (X.unsqueeze(1) != centroids.unsqueeze(0)).sum(dim=2).float()

    def _jaccard(self, X: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """
        jaccard dist

        Args:
            X: (torch.Tensor) Input data (n_samples, n_features)
            centroids: (torch.Tensor) Centroids (n_clusters, n_features)

        Returns:
            (torch.Tensor) Distance matrix (n_samples, n_clusters)
        """
        n_features = X.shape[1]
        hamming = self._hamming(X, centroids)
        intersection = n_features - hamming
        union = 2 * n_features - intersection
        # no div0
        return 1.0 - (intersection / union)

    def _ng(
        self,
        X: torch.Tensor,
        centroids: torch.Tensor,
        labels: torch.Tensor,
        n_features: int,
    ) -> torch.Tensor:
        """
        ng dist (freq based)

        Args:
            X: (torch.Tensor) Input data (n_samples, n_features)
            centroids: (torch.Tensor) Centroids (n_clusters, n_features)
            labels: (torch.Tensor) Cluster labels (n_samples)
            n_features: (int) Number of features

        Returns:
            (torch.Tensor) Distance matrix (n_samples, n_clusters)
        """
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]

        # 1. counts & sizes
        max_val = int(X.max().item())
        if max_val < 0:
            max_val = 0
        val_offset = max_val + 1

        counts_flat = self._compute_counts(
            X, labels, n_clusters, n_features, val_offset
        )
        counts = counts_flat.view(n_features, n_clusters, val_offset)

        cluster_sizes = torch.bincount(labels, minlength=n_clusters).float()
        cluster_sizes[cluster_sizes == 0] = 1.0

        # probs: (F, K, V)
        probs = counts.float() / cluster_sizes.view(1, n_clusters, 1)

        # vec lookup (no loops)

        # rearrange probs
        probs_flat = (
            probs.permute(0, 2, 1)
            .contiguous()
            .view(n_features * val_offset, n_clusters)
        )

        # calc idx
        # feature_offsets: (1, F)
        feature_offsets = (
            torch.arange(n_features, device=self.device) * val_offset
        ).unsqueeze(0)

        # indices: (N, F) -> fat (N*F)
        lookup_indices = (X + feature_offsets).view(-1).long()

        # gather probs
        gathered_probs = probs_flat.index_select(0, lookup_indices)

        # reshape & sum
        gathered_probs = gathered_probs.view(n_samples, n_features, n_clusters)
        sum_probs = gathered_probs.sum(dim=1)  # (N, K)

        dists = float(n_features) - sum_probs
        return dists
