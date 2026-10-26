"""Scatter and path-length correction methods for soil spectroscopy.

Implements:
1. Standard Normal Variate (SNV)
2. Multiplicative Scatter Correction (MSC)
"""

from typing import Optional
import numpy as np


def snv_transform(X: np.ndarray) -> np.ndarray:
    """Standard Normal Variate (SNV) transformation.
    
    Removes multiplicative scatter interference and particle size variations by
    centering each individual spectrum to mean 0 and scaling by its own standard deviation.
    
    Parameters
    ----------
    X : np.ndarray
        Spectral array, shape (n_samples, n_bands) or (n_bands,).
        
    Returns
    -------
    np.ndarray
        SNV corrected spectra, same shape as X.
    """
    is_1d = (X.ndim == 1)
    spectra = np.atleast_2d(X).astype(np.float64)
    
    means = np.mean(spectra, axis=1, keepdims=True)
    stds = np.std(spectra, axis=1, keepdims=True)
    stds[stds == 0] = 1e-8
    
    snv_spectra = (spectra - means) / stds
    return snv_spectra[0] if is_1d else snv_spectra


class MSCTransformer:
    """Multiplicative Scatter Correction (MSC).
    
    Fits each sample spectrum linearly against an ideal reference spectrum
    (typically the mean spectrum of the training set) to decouple additive and
    multiplicative physical scattering from chemical absorption.
    """
    def __init__(self, reference_spectrum: Optional[np.ndarray] = None):
        self.reference_spectrum = reference_spectrum

    def fit(self, X: np.ndarray):
        """Fit reference spectrum from training spectra."""
        X_arr = np.atleast_2d(X)
        self.reference_spectrum = np.mean(X_arr, axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply MSC correction using the fitted reference spectrum."""
        if self.reference_spectrum is None:
            raise ValueError("MSCTransformer must be fitted before transforming.")

        is_1d = (X.ndim == 1)
        spectra = np.atleast_2d(X).astype(np.float64)
        n_samples, n_bands = spectra.shape
        corrected = np.empty_like(spectra)

        # Regress each spectrum against reference: spectrum = a + b * reference
        ref_centered = self.reference_spectrum - np.mean(self.reference_spectrum)
        denom = np.sum(ref_centered ** 2) + 1e-12

        for i in range(n_samples):
            y = spectra[i]
            y_centered = y - np.mean(y)
            slope = np.sum(y_centered * ref_centered) / denom
            intercept = np.mean(y) - slope * np.mean(self.reference_spectrum)
            
            # Correct: (y - intercept) / slope
            corrected[i] = (y - intercept) / (slope if abs(slope) > 1e-8 else 1.0)

        return corrected[0] if is_1d else corrected

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
