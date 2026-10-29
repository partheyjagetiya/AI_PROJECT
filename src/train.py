"""Training loop for Soil Hyperspectral 1D-CNN Model.

Features:
- Standardized target scaling (Z-score) to prevent potassium/phosphorus scale distortion.
- Spatial Block Cross-Validation (50 km regional clusters) with 5-fold mean +/- std reporting.
- Clean Train / Validation (early stopping) / Held-out Test evaluation.
- Correct Agronomy 2024 Table 3 RPD Tier thresholds (Tier A: >3.0, Tier B: 2.5-3.0, Tier C: 2.0-2.5, Tier D: <2.0).
- Automatic serialization of model weights, preprocessor, and target scaler via joblib.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SoilSpectralDataset
from src.data.lucas import (
    generate_synthetic_soil_spectral_dataset,
    load_lucas_data,
    load_lucas_2018_properties,
    DEFAULT_WAVELENGTHS
)
from src.models.cnn1d import SoilSpectralCNN
from src.models.transformer_cnn import TransformerCNN
from src.preprocessing.pipeline import SpectralPreprocessor
from src.utils.spatial_cv import SpatialBlockCV


TARGET_NAMES = ["SOC (g/kg)", "Total N (g/kg)", "Avail. P (mg/kg)", "Ext. K (mg/kg)"]


def get_rpd_tier(rpd: float) -> str:
    """Classifies performance strictly using Table 3 from Cao et al. (Agronomy 2024)."""
    if rpd > 3.0:
        return "Tier A (Excellent)"
    elif rpd >= 2.5:
        return "Tier B (Good)"
    elif rpd >= 2.0:
        return "Tier C (Fair / Approximate)"
    else:
        return "Tier D (Unsatisfactory / Screening)"


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Computes R2, RMSE, and RPD for each soil property in physical units."""
    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        t = y_true[:, i]
        p = y_pred[:, i]
        r2 = float(r2_score(t, p))
        rmse = float(np.sqrt(mean_squared_error(t, p)))
        std_val = float(np.std(t))
        rpd = float(std_val / (rmse + 1e-8))
        metrics[name] = {"r2": r2, "rmse": rmse, "rpd": rpd, "tier": get_rpd_tier(rpd)}
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion,
    device: torch.device
) -> float:
    model.train()
    total_loss = 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return total_loss / len(loader.dataset), preds, targets


def run_training_single_fold(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    save_dir: str = "models",
    model_name: str = "cnn1d"
) -> Tuple[nn.Module, Dict[str, Dict[str, float]], SpectralPreprocessor, StandardScaler]:
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Target standardizer fit strictly on train fold
    target_scaler = StandardScaler()
    y_train_scaled = target_scaler.fit_transform(y[train_idx]).astype(np.float32)
    y_val_scaled = target_scaler.transform(y[val_idx]).astype(np.float32)
    y_test_scaled = target_scaler.transform(y[test_idx]).astype(np.float32)

    # 2. Fit SpectralPreprocessor strictly on training spectra (no leakage)
    preprocessor = SpectralPreprocessor(
        sg_window=11,
        sg_poly=2,
        continuum_removal=True,
        scatter_method="snv",
        scale=True
    )
    preprocessor.fit(X[train_idx], wavelengths)

    # 3. Create Datasets and DataLoaders
    train_ds = SoilSpectralDataset(X[train_idx], y_train_scaled, wavelengths, preprocessor=preprocessor)
    val_ds = SoilSpectralDataset(X[val_idx], y_val_scaled, wavelengths, preprocessor=preprocessor)
    test_ds = SoilSpectralDataset(X[test_idx], y_test_scaled, wavelengths, preprocessor=preprocessor)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 4. Instantiate Model
    n_bands = X.shape[1]
    n_targets = y.shape[1]
    if model_name.lower() == "transformer_cnn":
        model = TransformerCNN(n_bands=n_bands, n_targets=n_targets).to(device)
    else:
        model = SoilSpectralCNN(n_bands=n_bands, n_targets=n_targets).to(device)

    # Weighted normalized loss: slight priority to SOC and N (direct covalent dipole bonds)
    loss_weights = torch.tensor([1.2, 1.2, 0.8, 0.8], device=device)
    def loss_fn(p, t):
        return torch.mean(loss_weights * (p - t) ** 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_loss = float("inf")
    patience_cnt = 0
    best_model_file = save_path / "best_model.pt"

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, _, _ = evaluate(model, val_loader, loss_fn, device)
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_bands": n_bands,
                "wavelengths": wavelengths,
                "model_name": model_name
            }, best_model_file)
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    # Reload best model checkpoint for evaluation on HELD-OUT TEST FOLD
    checkpoint = torch.load(best_model_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Predict on held-out test set
    _, test_preds_scaled, _ = evaluate(model, test_loader, loss_fn, device)
    test_preds = target_scaler.inverse_transform(test_preds_scaled)
    test_targets = y[test_idx]

    test_metrics = compute_metrics(test_targets, test_preds)

    # Save preprocessor and target scaler artifacts for deployment
    joblib.dump(preprocessor, save_path / "preprocessor.joblib")
    joblib.dump(target_scaler, save_path / "target_scaler.joblib")

    return model, test_metrics, preprocessor, target_scaler


def run_training(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    coords: Optional[np.ndarray] = None,
    epochs: int = 35,
    batch_size: int = 32,
    save_dir: str = "models",
    model_name: str = "cnn1d",
    full_5fold: bool = False
):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    cv = SpatialBlockCV(n_splits=5, block_size_km=50.0)
    splits = list(cv.split(X, y, coords))

    if full_5fold:
        print("\n" + "=" * 75)
        print(f"RUNNING FULL 5-FOLD SPATIAL BLOCK CV [{model_name.upper()}] (50 km Regional Folds)")
        print("=" * 75)
        all_fold_metrics = []

        for fold, (train_val_idx, test_idx) in enumerate(splits, 1):
            val_split = int(len(train_val_idx) * 0.8)
            train_idx = train_val_idx[:val_split]
            val_idx = train_val_idx[val_split:]

            if coords is not None:
                min_dist = SpatialBlockCV.calculate_min_spatial_distance_km(coords, train_idx, test_idx)
                print(f"\n[*] Fold {fold}/5: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)} | Min Separation: {min_dist:.1f} km")
            else:
                print(f"\n[*] Fold {fold}/5: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

            model, metrics, _, _ = run_training_single_fold(
                X, y, wavelengths, train_idx, val_idx, test_idx,
                epochs=epochs, batch_size=batch_size, save_dir=save_dir, model_name=model_name
            )
            all_fold_metrics.append(metrics)

        # Print Mean +/- Std Summary across all 5 folds
        print("\n" + "=" * 75)
        print(f"5-FOLD CROSS-VALIDATION SUMMARY (Mean +/- Std) — {model_name.upper()}")
        print("=" * 75)
        for analyte in TARGET_NAMES:
            r2s = [m[analyte]["r2"] for m in all_fold_metrics]
            rmses = [m[analyte]["rmse"] for m in all_fold_metrics]
            rpds = [m[analyte]["rpd"] for m in all_fold_metrics]
            mean_rpd = float(np.mean(rpds))
            tier = get_rpd_tier(mean_rpd)
            print(f"  {analyte:18s} | R²: {np.mean(r2s):.3f} +/- {np.std(r2s):.3f} | RMSE: {np.mean(rmses):.3f} +/- {np.std(rmses):.3f} | RPD: {mean_rpd:.2f} +/- {np.std(rpds):.2f} | {tier}")
        print("=" * 75)

    else:
        # Standard 3-way partition: 70% Train, 15% Val, 15% Test
        train_val_idx, test_idx = splits[0]
        val_split = int(len(train_val_idx) * 0.8)
        train_idx = train_val_idx[:val_split]
        val_idx = train_val_idx[val_split:]

        if coords is not None:
            min_dist = SpatialBlockCV.calculate_min_spatial_distance_km(coords, train_idx, test_idx)
            print(f"[*] Spatial Separation Buffer between Train and Held-out Test: {min_dist:.1f} km")

        print(f"[*] Train={len(train_idx)}, Val={len(val_idx)}, Held-Out Test={len(test_idx)}")
        model, metrics, preproc, scaler = run_training_single_fold(
            X, y, wavelengths, train_idx, val_idx, test_idx,
            epochs=epochs, batch_size=batch_size, save_dir=save_dir, model_name=model_name
        )

        print("\n" + "=" * 75)
        print(f"FINAL HELD-OUT TEST EVALUATION [{model_name.upper()}] (Agronomy 2024 Table 3 Tiers)")
        print("=" * 75)
        for analyte, m in metrics.items():
            print(f"  {analyte:18s} | R²: {m['r2']:6.3f} | RMSE: {m['rmse']:6.3f} | RPD: {m['rpd']:5.2f} | {m['tier']}")
        print("=" * 75)
        return model, metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Soil Properties Model on Vis-NIR Soil Data")
    parser.add_argument("--model", type=str, default="cnn1d", choices=["cnn1d", "transformer_cnn"])
    parser.add_argument("--lucas2018", action="store_true", help="Train on official LUCAS 2018 ground truth chemical targets")
    parser.add_argument("--samples", type=int, default=1200, help="Number of samples to train on")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--5fold", action="store_true", dest="full_5fold", help="Run full 5-fold cross-validation")
    args = parser.parse_args()

    lucas_csv = Path("LUCAS-SOIL-2018-data-report-readme-v2/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv")

    if args.lucas2018 and lucas_csv.exists():
        print(f"[*] Loading official LUCAS 2018 Ground Truths from {lucas_csv}...")
        df_lucas, y_full, coords_full = load_lucas_2018_properties(str(lucas_csv))
        n = min(args.samples, len(y_full))
        y = y_full[:n]
        coords = coords_full[:n]
        wavelengths = np.linspace(400.0, 2450.0, 486)
        
        # Radiative forward simulation conditioned on real field chemical properties
        rng = np.random.default_rng(42)
        norm_wl = (wavelengths - 400.0) / (2450.0 - 400.0)
        X = np.empty((n, 486), dtype=np.float32)
        for i in range(n):
            soc_val = y[i, 0]
            base_albedo = 0.55 - 0.22 * np.tanh(soc_val / 40.0) + rng.normal(0, 0.02)
            curve = base_albedo * (0.35 + 0.65 * (norm_wl ** 0.5))
            m = rng.uniform(0.05, 0.35)
            curve -= 0.18 * m * np.exp(-((wavelengths - 1410) / 45) ** 2)
            curve -= 0.30 * m * np.exp(-((wavelengths - 1920) / 60) ** 2)
            f = min(1.0, soc_val / 60.0)
            curve -= 0.035 * f * np.exp(-((wavelengths - 1725) / 25) ** 2)
            curve -= 0.045 * f * np.exp(-((wavelengths - 2310) / 35) ** 2)
            curve -= 0.05 * np.exp(-((wavelengths - 2205) / 20) ** 2)
            curve -= 0.12 * np.exp(-((wavelengths - 480) / 80) ** 2)
            curve += rng.uniform(-0.03, 0.03) * norm_wl + rng.normal(0, 0.003, size=486)
            X[i] = np.clip(curve, 0.01, 0.95).astype(np.float32)
    else:
        print("[*] Generating synthetic Vis-NIR soil dataset...")
        X, y, wavelengths, coords = generate_synthetic_soil_spectral_dataset(
            n_samples=args.samples, n_bands=486, save_dir="data/processed"
        )

    run_training(
        X, y, wavelengths, coords,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model,
        full_5fold=args.full_5fold
    )
