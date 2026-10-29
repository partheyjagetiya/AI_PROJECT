"""Unified Spectral Preprocessing Pipeline for Soil Diffuse Reflectance Spectroscopy.

Mathematically correct order of operations:
1. Savitzky-Golay smoothing (on raw positive reflectance to reduce detector noise).
2. Continuum Removal (Clark & Roush, 1984 convex hull normalization on positive reflectance).
   - Isolates diagnostic absorption band depths in [0, 1].
3. Baseline Drift Correction (ALS or airPLS) - applied if continuum removal is disabled.
4. Scatter Correction (SNV or MSC) - normalizes physical particle-size scattering across samples.
5. Atmospheric / sensor water band masking (optional, e.g. 1350-1420 nm, 1800-1950 nm).
6. Per-band Standard Scaling (Z-score normalization, fit strictly on training fold).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.signal import savgol_filter

from .baseline import als_baseline, airpls_baseline
from .continuum import continuum_removal_band_ranges, continuum_removal_single
from .scatter import snv_transform, MSCTransformer


class SpectralPreprocessor:
    """End-to-end preprocessing pipeline for soil Vis-NIR hyperspectral data.
    
    Guarantees that Continuum Removal is performed on positive reflectance
    BEFORE zero-centering or scatter transformations, preventing signal destruction.
    """

    def __init__(
        self,
        sg_window: int = 11,
        sg_poly: int = 2,
        sg_deriv: int = 0,
        continuum_removal: bool = True,
        continuum_ranges: Optional[List[Tuple[float, float]]] = None,
        baseline_method: Optional[str] = None,       # 'als', 'airpls', or None (CR already handles baseline)
        als_lam: float = 1e5,
        als_p: float = 0.001,
        airpls_lam: float = 100.0,
        scatter_method: Optional[str] = "snv",       # 'snv', 'msc', or None
        mask_water_bands: bool = False,
        water_band_ranges: Optional[List[Tuple[float, float]]] = None,
        scale: bool = True
    ):
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.sg_deriv = sg_deriv
        self.continuum_removal = continuum_removal
        self.continuum_ranges = continuum_ranges
        self.baseline_method = baseline_method
        self.als_lam = als_lam
        self.als_p = als_p
        self.airpls_lam = airpls_lam
        self.scatter_method = scatter_method
        self.mask_water_bands = mask_water_bands
        self.water_band_ranges = water_band_ranges or [(1350, 1420), (1800, 1950)]
        self.scale = scale

        self.msc_ = MSCTransformer() if scatter_method == "msc" else None
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False

    def _apply_optical_physics(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies physical optical corrections: Smoothing, Continuum Removal, and Baseline."""
        is_1d = (X.ndim == 1)
        spectra = np.atleast_2d(X).copy().astype(np.float64)
        n_samples, n_bands = spectra.shape

        if wavelengths is None:
            wavelengths = np.linspace(400, 2450, n_bands)

        # 1. Savitzky-Golay smoothing on positive raw reflectance
        if self.sg_window and self.sg_window > self.sg_poly:
            window = self.sg_window if self.sg_window % 2 == 1 else self.sg_window + 1
            if window <= n_bands:
                spectra = savgol_filter(spectra, window_length=window, polyorder=self.sg_poly, deriv=self.sg_deriv, axis=1)
                spectra = np.clip(spectra, 1e-4, 1.0)

        # 2. Continuum Removal on positive reflectance (must precede SNV/zero-centering)
        if self.continuum_removal:
            if self.continuum_ranges is not None:
                spectra = continuum_removal_band_ranges(wavelengths, spectra, feature_ranges=self.continuum_ranges)
            else:
                for i in range(n_samples):
                    spectra[i] = continuum_removal_single(wavelengths, spectra[i])

        # 3. Baseline drift correction (only if continuum removal is not already removing baseline)
        if self.baseline_method == "als":
            for i in range(n_samples):
                bl = als_baseline(spectra[i], lam=self.als_lam, p=self.als_p)
                spectra[i] = spectra[i] - bl
        elif self.baseline_method == "airpls":
            for i in range(n_samples):
                bl = airpls_baseline(spectra[i], lam=self.airpls_lam)
                spectra[i] = spectra[i] - bl

        # 4. Atmospheric / UAV water band masking
        if self.mask_water_bands:
            for start_wl, end_wl in self.water_band_ranges:
                mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
                spectra[:, mask] = 0.0

        return spectra[0] if is_1d else spectra

    def fit(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None):
        """Fits MSC reference spectrum and standard scaler statistics on training fold."""
        X_proc = self._apply_optical_physics(X, wavelengths)
        X_proc_2d = np.atleast_2d(X_proc)

        # Fit MSC reference spectrum if enabled
        if self.scatter_method == "msc" and self.msc_ is not None:
            self.msc_.fit(X_proc_2d)
            X_proc_2d = self.msc_.transform(X_proc_2d)
        elif self.scatter_method == "snv":
            X_proc_2d = snv_transform(X_proc_2d)

        # Fit standard scaler (Z-score)
        if self.scale:
            self.mean_ = np.mean(X_proc_2d, axis=0)
            self.std_ = np.std(X_proc_2d, axis=0)
            self.std_[self.std_ < 1e-8] = 1.0

        self.is_fitted_ = True
        return self

    def transform(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Transforms spectra using parameters fitted strictly on training data."""
        is_1d = (X.ndim == 1)
        proc = self._apply_optical_physics(X, wavelengths)
        proc_2d = np.atleast_2d(proc)

        # Apply scatter correction
        if self.scatter_method == "snv":
            proc_2d = snv_transform(proc_2d)
        elif self.scatter_method == "msc" and self.is_fitted_ and self.msc_ is not None:
            proc_2d = self.msc_.transform(proc_2d)

        # Apply standardization
        if self.scale and self.mean_ is not None and self.std_ is not None:
            proc_2d = (proc_2d - self.mean_) / self.std_

        return proc_2d[0] if is_1d else proc_2d

    def fit_transform(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, wavelengths).transform(X, wavelengths)
