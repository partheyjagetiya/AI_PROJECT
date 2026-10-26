"""Spatial Block Cross-Validation for geospatial soil spectral data.

Prevents spatial autocorrelation leakage (samples collected in close geographic proximity
having artificially inflated validation scores) by clustering geographic coordinates
into spatial spatial blocks / folds.
"""

from typing import Generator, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold


class SpatialBlockCV:
    """Spatial Block Cross-Validation.
    
    Clusters coordinates (lat, lon) into geographic spatial blocks or groups,
    ensuring that validation folds are spatially disjoint from training folds.
    """

    def __init__(self, n_splits: int = 5, block_size_km: float = 50.0, random_state: int = 42):
        self.n_splits = n_splits
        self.block_size_km = block_size_km
        self.random_state = random_state

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set.
        
        Parameters
        ----------
        X : np.ndarray
            Spectral samples, shape (n_samples, n_features).
        y : np.ndarray, optional
            Target values, shape (n_samples, n_targets).
        coordinates : np.ndarray, optional
            Geographic coordinates [latitude, longitude] in decimal degrees, shape (n_samples, 2).
            If None or not provided, falls back to regular KFold.
            
        Yields
        ------
        train_idx : np.ndarray
        val_idx : np.ndarray
        """
        n_samples = len(X)

        if coordinates is None or len(coordinates) != n_samples:
            # Fallback to standard KFold if coordinates not provided
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            yield from kf.split(X)
            return

        coords = np.asarray(coordinates, dtype=np.float64)

        # Approximate conversion of degrees to km
        # 1 degree latitude ~ 111 km, longitude scaled by cos(lat)
        mean_lat = np.radians(np.mean(coords[:, 0]))
        coords_km = np.empty_like(coords)
        coords_km[:, 0] = coords[:, 0] * 111.0
        coords_km[:, 1] = coords[:, 1] * (111.0 * np.cos(mean_lat))

        # Determine number of spatial clusters based on bounding box and block_size_km
        span_x = np.ptp(coords_km[:, 1])
        span_y = np.ptp(coords_km[:, 0])
        approx_clusters = max(
            self.n_splits * 2,
            int(np.ceil((span_x * span_y) / (self.block_size_km ** 2)))
        )
        # Cap clusters to avoid tiny single-element clusters
        n_clusters = min(approx_clusters, max(self.n_splits, n_samples // 10))

        # Spatial clustering into blocks
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        block_labels = kmeans.fit_predict(coords_km)

        # Distribute cluster blocks across n_splits folds
        unique_blocks = np.unique(block_labels)
        rng = np.random.default_rng(self.random_state)
        rng.shuffle(unique_blocks)
        block_splits = np.array_split(unique_blocks, self.n_splits)

        for fold_blocks in block_splits:
            val_mask = np.isin(block_labels, fold_blocks)
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(~val_mask)[0]
            if len(val_idx) > 0 and len(train_idx) > 0:
                yield train_idx, val_idx


