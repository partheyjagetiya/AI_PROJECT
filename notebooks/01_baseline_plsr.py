"""Phase 1 Baseline: Partial Least Squares Regression (PLSR) on Soil Spectra.

Demonstrates that the preprocessing pipeline (Savitzky-Golay, SNV, Continuum Removal, Baseline Drift Correction)
achieves strong linear baseline predictive accuracy prior to training deep 1D-CNNs.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

from src.data.lucas import generate_synthetic_soil_spectral_dataset
from src.preprocessing.pipeline import SpectralPreprocessor


def run_plsr_baseline():
    print("=" * 60)
    print("PHASE 1 GATE TEST: PLSR BASELINE ON PREPROCESSED VIS-NIR SPECTRA")
    print("=" * 60)

    # 1. Load / generate synthetic soil data
    print("[*] Generating synthetic soil Vis-NIR dataset (1,200 samples, 486 bands)...")
    X, y, wavelengths, coords = generate_synthetic_soil_spectral_dataset(
        n_samples=1200,
        n_bands=486,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Preprocess spectra
    print("[*] Preprocessing with SG Filter, SNV, Continuum Removal, and Baseline Correction...")
    preprocessor = SpectralPreprocessor(
        sg_window=11,
        sg_poly=2,
        baseline_method="als",
        scatter_method="snv",
        continuum_removal=True,
        scale=True
    )
    X_train_proc = preprocessor.fit_transform(X_train, wavelengths)
    X_val_proc = preprocessor.transform(X_val, wavelengths)

    # 3. Fit PLSR with 20 latent components
    print("[*] Fitting PLS Regression (n_components=20)...")
    pls = PLSRegression(n_components=20)
    pls.fit(X_train_proc, y_train)

    preds = pls.predict(X_val_proc)

    # 4. Evaluate metrics
    names = ["SOC (g/kg)", "Nitrogen (g/kg)", "Phosphorus (mg/kg)", "Potassium (mg/kg)"]
    print("\nRESULTS:")
    print("-" * 60)
    for i, name in enumerate(names):
        r2 = r2_score(y_val[:, i], preds[:, i])
        rmse = np.sqrt(mean_squared_error(y_val[:, i], preds[:, i]))
        rpd = np.std(y_val[:, i]) / rmse
        status = "PASSED" if r2 > 0.6 else "MARGINAL"
        print(f"  {name:22s} | R²: {r2:6.3f} | RMSE: {rmse:6.3f} | RPD: {rpd:5.2f} | Gate: {status}")
    print("-" * 60)


if __name__ == "__main__":
    run_plsr_baseline()
