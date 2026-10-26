"""Continuum removal for soil diffuse reflectance spectroscopy.

Continuum removal isolates diagnostic absorption features (e.g., C-H, O-H, Clay minerals)
and neutralizes baseline albedo and soil moisture scattering shifts by fitting an upper convex hull.
"""

from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import ConvexHull


def continuum_removal_single(wavelengths: np.ndarray, reflectance: np.ndarray) -> np.ndarray:
    """Computes full-spectrum continuum removal using an upper convex hull.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Array of wavelengths (nm), shape (L,).
    reflectance : np.ndarray
        Array of spectral reflectance values, shape (L,).
        
    Returns
    -------
    np.ndarray
        Continuum-removed reflectance values in [0, 1], shape (L,).
    """
    wavelengths = np.asarray(wavelengths, dtype=np.float64)
    reflectance = np.asarray(reflectance, dtype=np.float64)
    n_points = len(wavelengths)

    if n_points < 3:
        return np.ones_like(reflectance)

    # Construct 2D points with synthetic lower bounding points to force upper hull
    # The upper convex envelope joins local reflectance maxima
    min_val = np.min(reflectance) - 1.0
    pts = np.column_stack([wavelengths, reflectance])
    
    # Add bottom corners so ConvexHull contains the upper surface boundary
    pts_augmented = np.vstack([
        pts,
        [wavelengths[-1], min_val],
        [wavelengths[0], min_val]
    ])

    hull = ConvexHull(pts_augmented)
    hull_indices = hull.vertices

    # Keep only original spectral indices (exclude artificial bottom vertices)
    valid_hull_idx = [idx for idx in hull_indices if idx < n_points]
    valid_hull_idx = sorted(valid_hull_idx)

    # Ensure start and end points are included
    if 0 not in valid_hull_idx:
        valid_hull_idx.insert(0, 0)
    if (n_points - 1) not in valid_hull_idx:
        valid_hull_idx.append(n_points - 1)

    hull_x = wavelengths[valid_hull_idx]
    hull_y = reflectance[valid_hull_idx]

    # Interpolate continuum envelope across all wavelengths
    continuum = np.interp(wavelengths, hull_x, hull_y)
    
    # Avoid zero division and cap at 1.0
    continuum = np.maximum(continuum, 1e-8)
    cr = reflectance / continuum
    return np.clip(cr, 0.0, 1.0)


def continuum_removal_band_ranges(
    wavelengths: np.ndarray,
    reflectance: np.ndarray,
    feature_ranges: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """Continuum removal applied either to specific diagnostic absorption features
    or across the entire spectrum.
    
    Parameters
    ----------
    wavelengths : np.ndarray
        Wavelengths (nm), shape (L,).
    reflectance : np.ndarray
        1D or 2D reflectance array (n_bands,) or (n_samples, n_bands).
    feature_ranges : list of (start, end) nm tuples, optional
        Diagnostic ranges to isolate, e.g.:
        - (1350, 1500): O-H water / hydroxyl
        - (1650, 1750): C-H organic carbon
        - (1850, 2050): H2O moisture
        - (2150, 2300): Al-OH clay / organic C=O
        If None, applies continuum removal across the entire spectrum.
        
    Returns
    -------
    np.ndarray
        Continuum removed spectra of the same shape as reflectance.
    """
    is_1d = (reflectance.ndim == 1)
    spectra = np.atleast_2d(reflectance)
    output = spectra.copy()

    if feature_ranges is None:
        # Full spectrum continuum removal
        for i in range(len(spectra)):
            output[i] = continuum_removal_single(wavelengths, spectra[i])
    else:
        # Targeted feature continuum removal
        for start_wl, end_wl in feature_ranges:
            mask = (wavelengths >= start_wl) & (wavelengths <= end_wl)
            if np.sum(mask) >= 3:
                sub_wl = wavelengths[mask]
                for i in range(len(spectra)):
                    output[i, mask] = continuum_removal_single(sub_wl, spectra[i, mask])

    return output[0] if is_1d else output
