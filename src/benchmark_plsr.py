"""PLSR Benchmark Evaluator on 5-Fold SpatialBlockCV.

Runs Partial Least Squares Regression (PLSR) across the exact same 5 folds of
50 km SpatialBlockCV as the 1D-CNN, reporting Mean +/- Std for SOC, N, P, K.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from src.data.lucas import generate_synthetic_soil_spectral_dataset, DEFAULT_WAVELENGTHS
from src.preprocessing.pipeline import SpectralPreprocessor
from src.utils.spatial_cv import SpatialBlockCV


TARGET_NAMES = ["SOC (g/kg)", "Total N (g/kg)", "Avail. P (mg/kg)", "Ext. K (mg/kg)"]


def get_rpd_tier(rpd: float) -> str:
    if rpd > 3.0:
        return "Tier A (Excellent)"
    elif rpd >= 2.5:
        return "Tier B (Good)"
    elif rpd >= 2.0:
        return "Tier C (Fair / Approximate)"
    else:
        return "Tier D (Unsatisfactory / Screening)"


def run_plsr_5fold(n_samples: int = 1000, n_components: int = 15):
    print("=" * 80)
    print("  PLSR BASELINE EVALUATION ACROSS 5-FOLD SPATIAL BLOCK CV (50 km Regional Folds)")
    print("=" * 80)

    X_raw, y, wavelengths, coords = generate_synthetic_soil_spectral_dataset(
        n_samples=n_samples, n_bands=486, random_state=42
    )

    cv = SpatialBlockCV(n_splits=5, block_size_km=50.0, random_state=42)
    splits = list(cv.split(X_raw, y, coords))

    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(splits, 1):
        min_dist = SpatialBlockCV.calculate_min_spatial_distance_km(coords, train_idx, test_idx)
        print(f"[*] Fold {fold}/5: Train={len(train_idx)}, Test={len(test_idx)} | Spatial Separation: {min_dist:.1f} km")

        # Preprocessing strictly on train fold
        preprocessor = SpectralPreprocessor(sg_window=11, sg_poly=2, continuum_removal=True, scatter_method="snv", scale=True)
        preprocessor.fit(X_raw[train_idx], wavelengths)
        X_tr = preprocessor.transform(X_raw[train_idx], wavelengths)
        X_te = preprocessor.transform(X_raw[test_idx], wavelengths)

        # Scale targets
        scaler = StandardScaler()
        y_tr = scaler.fit_transform(y[train_idx])

        pls = PLSRegression(n_components=n_components)
        pls.fit(X_tr, y_tr)
        y_pred = scaler.inverse_transform(pls.predict(X_te))

        metrics = {}
        for i, name in enumerate(TARGET_NAMES):
            t = y[test_idx, i]
            p = y_pred[:, i]
            r2 = float(r2_score(t, p))
            rmse = float(np.sqrt(mean_squared_error(t, p)))
            rpd = float(np.std(t) / (rmse + 1e-8))
            metrics[name] = {"r2": r2, "rmse": rmse, "rpd": rpd}
        fold_metrics.append(metrics)

    print("\n" + "=" * 80)
    print("  PLSR 5-FOLD CROSS-VALIDATION SUMMARY (Mean +/- Std)")
    print("=" * 80)
    for analyte in TARGET_NAMES:
        r2s = [m[analyte]["r2"] for m in fold_metrics]
        rmses = [m[analyte]["rmse"] for m in fold_metrics]
        rpds = [m[analyte]["rpd"] for m in fold_metrics]
        mean_rpd = float(np.mean(rpds))
        tier = get_rpd_tier(mean_rpd)
        print(f"  {analyte:18s} | R²: {np.mean(r2s):.3f} +/- {np.std(r2s):.3f} | RMSE: {np.mean(rmses):.3f} +/- {np.std(rmses):.3f} | RPD: {mean_rpd:.2f} +/- {np.std(rpds):.2f} | {tier}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_plsr_5fold()
