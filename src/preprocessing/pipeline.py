"""Unified Spectral Preprocessing Pipeline for Soil Diffuse Reflectance Spectroscopy."""

from typing import List, Optional, Tuple, Union
import numpy as np
from scipy.signal import savgol_filter

from .baseline import als_baseline, airpls_baseline
from .continuum import continuum_removal_band_ranges
from .scatter import snv_transform, MSCTransformer


class SpectralPreprocessor:
    """End-to-end preprocessing pipeline for soil Vis-NIR hyperspectral data.
    
    Order of operations:
    1. Savitzky-Golay smoothing or 1st/2nd derivative
    2. Scatter correction (SNV or MSC)
    3. Baseline drift correction (ALS or airPLS)
    4. Continuum removal (to isolate diagnostic organic and mineral absorption dips)
    5. Atmospheric / sensor water band masking (interpolated or zeroed)
    6. Per-band standard scaling (fit on train set)
    """

    def __init__(
        self,
        sg_window: int = 11,
        sg_poly: int = 2,
        sg_deriv: int = 0,
        scatter_method: Optional[str] = "snv",       # 'snv', 'msc', or None
        baseline_method: Optional[str] = "als",     # 'als', 'airpls', or None
        als_lam: float = 1e5,
        als_p: float = 0.001,
        airpls_lam: float = 100.0,
        continuum_removal: bool = True,
        continuum_ranges: Optional[List[Tuple[float, float]]] = None,
        mask_water_bands: bool = False,
        water_band_ranges: Optional[List[Tuple[float, float]]] = None,
        scale: bool = True
    ):
        self.sg_window = sg_window
        self.sg_poly = sg_poly
        self.sg_deriv = sg_deriv
        self.scatter_method = scatter_method
        self.baseline_method = baseline_method
        self.als_lam = als_lam
        self.als_p = als_p
        self.airpls_lam = airpls_lam
        self.continuum_removal = continuum_removal
        self.continuum_ranges = continuum_ranges or [(1350, 1500), (1650, 1750), (1850, 2050), (2150, 2300)]
        self.mask_water_bands = mask_water_bands
        self.water_band_ranges = water_band_ranges or [(1350, 1420), (1800, 1950)]
        self.scale = scale

        self.msc_ = MSCTransformer() if scatter_method == "msc" else None
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False

    def fit(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None):
        """Fit statistics (MSC reference spectrum and standard scaler) on training spectra."""
        X_proc = self._transform_spectral_physics(X, wavelengths)
        if self.scatter_method == "msc":
            self.msc_.fit(X_proc)
            X_proc = self.msc_.transform(X_proc)

        if self.scale:
            self.mean_ = np.mean(X_proc, axis=0)
            self.std_ = np.std(X_proc, axis=0)
            self.std_[self.std_ < 1e-8] = 1.0

        self.is_fitted_ = True
        return self

    def _transform_spectral_physics(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Applies physical and optical preprocessing steps (SG, Scatter, Baseline, CR)."""
        is_1d = (X.ndim == 1)
        spectra = np.atleast_2d(X).copy().astype(np.float64)
        n_samples, n_bands = spectra.shape

        if wavelengths is None:
            wavelengths = np.linspace(400, 2500, n_bands)

        # 1. Savitzky-Golay filtering
        if self.sg_window and self.sg_window > self.sg_poly:
            window = self.sg_window if self.sg_window % 2 == 1 else self.sg_window + 1
            if window <= n_bands:
                spectra = savgol_filter(spectra, window_length=window, polyorder=self.sg_poly, deriv=self.sg_deriv, axis=1)

        # 2. Scatter correction
        if self.scatter_method == "snv":
            spectra = snv_transform(spectra)
        elif self.scatter_method == "msc" and self.is_fitted_ and self.msc_ is not None:
            spectra = self.msc_.transform(spectra)

        # 3. Baseline drift correction
        if self.baseline_method == "als":
            for i in range(n_samples):
                bl = als_baseline(spectra[i], lam=self.als_lam, p=self.als_p)
                spectra[i] = spectra[i] - bl
        elif self.baseline_method == "airpls":
            for i in range(n_samples):
                bl = airpls_baseline(spectra[i], lam=self.airpls_lam)
                spectra[i] = spectra[i] - bl

        # 4. Continuum removal
        if self.continuum_removal:
            spectra = continuum_removal_band_ranges(wavelengths, spectra, feature_ranges=self.continuum_ranges)

        # 5. Mask atmospheric water bands (critical for UAV/field spectra)
        if self.mask_water_bands:
            for start_wl, end_wl in self.water_band_ranges:
                mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
                spectra[:, mask] = 0.0

        return spectra[0] if is_1d else spectra

    def transform(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        """Transform spectra using fitted pipeline parameters."""
        is_1d = (X.ndim == 1)
        proc = self._transform_spectral_physics(X, wavelengths)
        proc_2d = np.atleast_2d(proc)

        # 6. Standardization
        if self.scale and self.mean_ is not None and self.std_ is not None:
            proc_2d = (proc_2d - self.mean_) / self.std_

        return proc_2d[0] if is_1d else proc_2d

    def fit_transform(self, X: np.ndarray, wavelengths: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, wavelengths).transform(X, wavelengths)
