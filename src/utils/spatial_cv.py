"""Spatial Block Cross-Validation for geospatial soil spectral data.

Mitigates spatial autocorrelation (Tobler's First Law of Geography)
by clustering geographic coordinates into distinct spatial blocks (~50 km)
and assigning contiguous regional clusters to validation folds.
"""

from typing import Generator, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold


class SpatialBlockCV:
    """Spatial Block Cross-Validation.
    
    Clusters coordinates (lat, lon) into geographic spatial blocks,
    ensuring that validation folds represent separate geographic regions.
    """

    def __init__(self, n_splits: int = 5, block_size_km: float = 50.0, random_state: int = 42):
        self.n_splits = n_splits
        self.block_size_km = block_size_km
        self.random_state = random_state

    def _to_km(self, coordinates: np.ndarray) -> np.ndarray:
        coords = np.asarray(coordinates, dtype=np.float64)
        mean_lat = np.radians(np.mean(coords[:, 0]))
        coords_km = np.empty_like(coords)
        coords_km[:, 0] = coords[:, 0] * 111.0
        coords_km[:, 1] = coords[:, 1] * (111.0 * np.cos(mean_lat))
        return coords_km

    def split(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        coordinates: Optional[np.ndarray] = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate indices to split data into training and test set."""
        n_samples = len(X)

        if coordinates is None or len(coordinates) != n_samples:
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            yield from kf.split(X)
            return

        coords_km = self._to_km(coordinates)

        span_x = np.ptp(coords_km[:, 1])
        span_y = np.ptp(coords_km[:, 0])
        approx_clusters = max(
            self.n_splits * 2,
            int(np.ceil((span_x * span_y) / (self.block_size_km ** 2)))
        )
        n_clusters = min(approx_clusters, max(self.n_splits * 2, n_samples // 12))

        # Spatial clustering into geographic blocks
        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state, n_init="auto")
        block_labels = kmeans.fit_predict(coords_km)
        centers = kmeans.cluster_centers_

        # Spatially sort clusters along principal spatial axis (e.g. West-East longitude projection)
        # to ensure folds represent contiguous regional sectors rather than interspersed blocks
        sorted_blocks = np.argsort(centers[:, 1])
        block_splits = np.array_split(sorted_blocks, self.n_splits)

        for fold_blocks in block_splits:
            val_mask = np.isin(block_labels, fold_blocks)
            val_idx = np.where(val_mask)[0]
            train_idx = np.where(~val_mask)[0]
            if len(val_idx) > 0 and len(train_idx) > 0:
                yield train_idx, val_idx

    @staticmethod
    def calculate_min_spatial_distance_km(coords: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray) -> float:
        """Computes minimum pairwise Euclidean geographic distance in km between train and val folds."""
        mean_lat = np.radians(np.mean(coords[:, 0]))
        c_km = np.empty_like(coords, dtype=np.float64)
        c_km[:, 0] = coords[:, 0] * 111.0
        c_km[:, 1] = coords[:, 1] * (111.0 * np.cos(mean_lat))

        # Subsample if large to compute quickly
        t_sample = c_km[train_idx[:min(500, len(train_idx))]]
        v_sample = c_km[val_idx[:min(200, len(val_idx))]]

        # Distance matrix
        diff = t_sample[:, np.newaxis, :] - v_sample[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff ** 2, axis=-1))
        return float(np.min(dist))
