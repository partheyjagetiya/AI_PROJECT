"""Baseline drift correction for soil diffuse reflectance spectroscopy.

Implements:
1. Asymmetric Least Squares (ALS) smoothing (Eilers & Boelens, 2005)
2. Adaptive Iteratively Reweighted Penalized Least Squares (airPLS) (Zhang et al., 2010)
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def als_baseline(y: np.ndarray, lam: float = 1e5, p: float = 0.001, niter: int = 15) -> np.ndarray:
    """Asymmetric Least Squares (ALS) baseline correction.
    
    Parameters
    ----------
    y : np.ndarray
        1D spectral vector of reflectance or absorbance, shape (L,).
    lam : float
        Smoothness parameter (typically 1e4 to 1e7 for Vis-NIR).
    p : float
        Asymmetry parameter (0.001 <= p <= 0.05). Reflectance features
        are treated as positive deviations if inverted or baseline is bottom-envelope.
    niter : int
        Number of iterations.
        
    Returns
    -------
    np.ndarray
        Estimated baseline, shape (L,).
    """
    y = np.asarray(y, dtype=np.float64)
    L = len(y)
    if L < 3:
        return y.copy()

    # Second difference operator D
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), dtype=np.float64)
    H = (lam * (D.T @ D)).tocsc()
    w = np.ones(L, dtype=np.float64)

    for _ in range(niter):
        W = sparse.diags(w, 0, shape=(L, L), dtype=np.float64).tocsc()
        Z = W + H
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1.0 - p) * (y <= z)

    return z


def airpls_baseline(y: np.ndarray, lam: float = 100.0, max_iter: int = 20, tol: float = 1e-3) -> np.ndarray:
    """Adaptive Iteratively Reweighted Penalized Least Squares (airPLS).
    
    airPLS dynamically updates weights using exponential penalties on negative
    residuals, making it faster and parameter-free regarding the asymmetry p.
    
    Parameters
    ----------
    y : np.ndarray
        1D spectral vector, shape (L,).
    lam : float
        Smoothness parameter (typically 10 to 1000).
    max_iter : int
        Maximum iterations.
    tol : float
        Relative convergence tolerance on sum of negative residuals.
        
    Returns
    -------
    np.ndarray
        Estimated baseline, shape (L,).
    """
    y = np.asarray(y, dtype=np.float64)
    L = len(y)
    if L < 3:
        return y.copy()

    # Second difference operator
    D = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(L - 2, L), dtype=np.float64)
    H = (lam * (D.T @ D)).tocsc()
    w = np.ones(L, dtype=np.float64)

    for i in range(1, max_iter + 1):
        W = sparse.diags(w, 0, shape=(L, L), dtype=np.float64).tocsc()
        Z = W + H
        z = spsolve(Z, w * y)
        d = y - z

        # Negative residuals indicate signals below baseline
        d_neg = d[d < 0]
        if len(d_neg) == 0:
            break

        neg_sum = np.sum(np.abs(d_neg))
        if i > 1 and (neg_sum / (np.sum(np.abs(y)) + 1e-9)) < tol:
            break

        # Exponential penalization for negative residuals
        w = np.zeros(L, dtype=np.float64)
        mask = d >= 0
        w[mask] = 0.0
        w[~mask] = np.exp(i * np.abs(d[~mask]) / (neg_sum + 1e-12))
        w[0] = np.exp(i * np.abs(d[0]) / (neg_sum + 1e-12)) if d[0] < 0 else 1.0
        w[-1] = np.exp(i * np.abs(d[-1]) / (neg_sum + 1e-12)) if d[-1] < 0 else 1.0

    return z
