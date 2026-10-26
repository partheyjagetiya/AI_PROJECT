"""Training loop for Soil Hyperspectral 1D-CNN Model.

Supports:
- Preprocessed Vis-NIR spectral feature extraction
- Multi-target weighted regression loss for SOC, N, P, K
- Early stopping & model checkpointing
- Evaluation metrics: R-squared, RMSE, and RPD (Residual Prediction Deviation)
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from typing import Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SoilSpectralDataset
from src.data.lucas import generate_synthetic_soil_spectral_dataset, load_lucas_data, DEFAULT_WAVELENGTHS
from src.models.cnn1d import SoilSpectralCNN
from src.models.transformer_cnn import TransformerCNN
from src.preprocessing.pipeline import SpectralPreprocessor
from src.utils.spatial_cv import SpatialBlockCV


TARGET_NAMES = ["SOC", "Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Computes R2, RMSE, and RPD for each soil property."""
    metrics = {}
    for i, name in enumerate(TARGET_NAMES):
        t = y_true[:, i]
        p = y_pred[:, i]
        r2 = float(r2_score(t, p))
        rmse = float(np.sqrt(mean_squared_error(t, p)))
        std_val = float(np.std(t))
        rpd = float(std_val / (rmse + 1e-8))
        metrics[name] = {"r2": r2, "rmse": rmse, "rpd": rpd}
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

    for x, y in tqdm(loader, desc="Train Epoch", leave=False):
        x = x.to(device)
        y = y.to(device)

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

    for x, y in tqdm(loader, desc="Evaluating", leave=False):
        x = x.to(device)
        y = y.to(device)

        pred = model(x)
        loss = criterion(pred, y)
        total_loss += loss.item() * x.size(0)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y.cpu().numpy())

    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    return total_loss / len(loader.dataset), preds, targets


def run_training(
    X: np.ndarray,
    y: np.ndarray,
    wavelengths: np.ndarray,
    coords: Optional[np.ndarray] = None,
    epochs: int = 40,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    save_dir: str = "models",
    use_spatial_split: bool = True,
    model_name: str = "cnn1d"
):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device} | Architecture: {model_name.upper()}")

    # 1. Train / Validation Split
    if use_spatial_split and coords is not None:
        cv = SpatialBlockCV(n_splits=5, block_size_km=60.0)
        train_idx, val_idx = next(cv.split(X, y, coords))
    else:
        indices = np.arange(len(X))
        train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    print(f"[*] Train set size: {len(train_idx)} | Validation set size: {len(val_idx)}")

    # 2. Fit Preprocessor strictly on training data
    print("[*] Fitting SpectralPreprocessor on training split...")
    preprocessor = SpectralPreprocessor(
        sg_window=11,
        sg_poly=2,
        baseline_method="als",
        scatter_method="snv",
        continuum_removal=True,
        scale=True
    )
    preprocessor.fit(X[train_idx], wavelengths)

    # 3. Create Datasets and DataLoaders
    train_ds = SoilSpectralDataset(X, y, wavelengths, preprocessor=preprocessor, indices=train_idx)
    val_ds = SoilSpectralDataset(X, y, wavelengths, preprocessor=preprocessor, indices=val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 4. Instantiate Model (1D-CNN or Agronomy 2024 Transformer-CNN)
    n_bands = X.shape[1]
    n_targets = y.shape[1]
    if model_name.lower() == "transformer_cnn":
        model = TransformerCNN(n_bands=n_bands, n_targets=n_targets, embed_dim=64, num_heads=4, dropout=0.1).to(device)
    else:
        model = SoilSpectralCNN(n_bands=n_bands, n_targets=n_targets, dropout=0.25).to(device)

    # Multi-target weighted loss: SOC and Nitrogen prioritized
    target_weights = torch.tensor([1.0, 4.0, 0.1, 0.05], device=device)

    def weighted_loss(pred, target):
        return torch.mean(target_weights * (pred - target) ** 2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    best_val_loss = float("inf")
    patience_cnt = 0
    best_model_path = os.path.join(save_dir, "best_model.pt")

    print(f"[*] Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, weighted_loss, device)
        val_loss, preds, targets = evaluate(model, val_loader, weighted_loss, device)
        scheduler.step()

        metrics = compute_metrics(targets, preds)
        soc_r2 = metrics["SOC"]["r2"]
        n_r2 = metrics["Nitrogen (N)"]["r2"]

        print(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"SOC R²: {soc_r2:.3f} | "
            f"N R²: {n_r2:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "n_bands": n_bands,
                "wavelengths": wavelengths,
                "metrics": metrics
            }, best_model_path)
            print(f"  --> Saved new best checkpoint to {best_model_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"[*] Early stopping triggered after {patience_cnt} epochs without improvement.")
                break

    # Load best checkpoint and report final detailed metrics
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, final_preds, final_targets = evaluate(model, val_loader, weighted_loss, device)
    final_metrics = compute_metrics(final_targets, final_preds)

    def get_rpd_tier(rpd: float) -> str:
        """Classifies performance using Table 3 from Cao et al. (Agronomy 2024)."""
        if rpd > 3.0:
            return "Tier A (Excellent)"
        elif rpd >= 2.5:
            return "Tier B (Good)"
        elif rpd >= 2.0:
            return "Tier C (Approximate)"
        else:
            return "Tier D (Unsatisfactory)"

    print("\n" + "=" * 65)
    print(f"FINAL VALIDATION EVALUATION [{model_name.upper()}] (Agronomy 2024 Tiers)")
    print("=" * 65)
    for target_name, m in final_metrics.items():
        tier = get_rpd_tier(m['rpd'])
        print(f"  {target_name:18s} | R²: {m['r2']:6.3f} | RMSE: {m['rmse']:6.3f} | RPD: {m['rpd']:5.2f} | {tier}")
    print("=" * 65)

    return model, final_metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train Soil Properties Model on Vis-NIR Soil Data")
    parser.add_argument("--model", type=str, default="cnn1d", choices=["cnn1d", "transformer_cnn"],
                        help="Model architecture: 'cnn1d' (Edge 1D-CNN) or 'transformer_cnn' (Agronomy 2024 paper)")
    parser.add_argument("--lucas2018", action="store_true", help="Train on official LUCAS 2018 dataset ground truths")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to train on")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    args = parser.parse_args()

    lucas_csv = Path("LUCAS-SOIL-2018-data-report-readme-v2/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv")

    if args.lucas2018 and lucas_csv.exists():
        from src.data.lucas import load_lucas_2018_properties
        print(f"[*] Loading official LUCAS 2018 Ground Truths from {lucas_csv}...")
        df_lucas, y_full, coords_full = load_lucas_2018_properties(str(lucas_csv))
        n = min(args.samples, len(y_full))
        y = y_full[:n]
        coords = coords_full[:n]
        print(f"[*] Simulating physical Vis-NIR spectra matched to {n} official LUCAS 2018 field measurements...")
        wavelengths = np.linspace(400, 2500, 486)
        
        # Simulate physical reflectance curves conditioned on the real OC, N, P, K from LUCAS 2018
        rng = np.random.default_rng(42)
        norm_wl = (wavelengths - 400.0) / (2500.0 - 400.0)
        X = np.empty((n, 486), dtype=np.float32)
        for i in range(n):
            soc_val = y[i, 0]
            base_albedo = 0.55 - 0.22 * np.tanh(soc_val / 40.0) + rng.normal(0, 0.02)
            curve = base_albedo * (0.35 + 0.65 * (norm_wl ** 0.5))
            # Soil moisture troughs at 1410 nm and 1920 nm
            m = rng.uniform(0.05, 0.35)
            curve -= 0.18 * m * np.exp(-((wavelengths - 1410) / 45) ** 2)
            curve -= 0.30 * m * np.exp(-((wavelengths - 1920) / 60) ** 2)
            # Diagnostic C-H & C=O pits
            f = min(1.0, soc_val / 60.0)
            curve -= 0.035 * f * np.exp(-((wavelengths - 1725) / 25) ** 2)
            curve -= 0.045 * f * np.exp(-((wavelengths - 2310) / 35) ** 2)
            curve -= 0.05 * np.exp(-((wavelengths - 2205) / 20) ** 2)
            curve -= 0.12 * np.exp(-((wavelengths - 480) / 80) ** 2)
            curve += rng.uniform(-0.03, 0.03) * norm_wl + rng.normal(0, 0.003, size=486)
            X[i] = np.clip(curve, 0.01, 0.95).astype(np.float32)
    else:
        print("[*] Generating synthetic Vis-NIR soil dataset for verification...")
        X, y, wavelengths, coords = generate_synthetic_soil_spectral_dataset(
            n_samples=args.samples,
            n_bands=486,
            save_dir="data/processed"
        )

    run_training(
        X, y, wavelengths, coords,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model
    )
