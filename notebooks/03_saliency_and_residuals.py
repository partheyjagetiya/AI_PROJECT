"""Phase 3 Evaluation: Saliency Map & Residual Analysis for Soil Vis-NIR Properties.

Computes input spectral gradients (|dSOC/dReflectance|) to confirm that the 1D-CNN
attends to chemically meaningful absorption bands (C-H, Al-OH, moisture troughs).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.lucas import generate_synthetic_soil_spectral_dataset
from src.models.cnn1d import SoilSpectralCNN
from src.preprocessing.pipeline import SpectralPreprocessor


def run_saliency_and_residuals(
    model_path: str = "models/best_model.pt",
    output_dir: str = "runs/analysis"
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")

    # 1. Load data
    print("[*] Generating validation samples for saliency & residual analysis...")
    X, y, wavelengths, coords = generate_synthetic_soil_spectral_dataset(
        n_samples=200,
        n_bands=486,
        random_state=123
    )

    preprocessor = SpectralPreprocessor(
        sg_window=11,
        sg_poly=2,
        baseline_method="als",
        scatter_method="snv",
        continuum_removal=True,
        scale=True
    )
    X_proc = preprocessor.fit_transform(X, wavelengths)

    # 2. Load model
    print(f"[*] Loading model from {model_path}...")
    model = SoilSpectralCNN(n_bands=len(wavelengths), n_targets=4)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Compute Saliency Maps
    print("[*] Computing input gradient saliency maps for SOC...")
    x_tensor = torch.from_numpy(X_proc[:10]).unsqueeze(1).float()
    x_tensor.requires_grad_(True)

    preds = model(x_tensor)
    soc_preds = preds[:, 0]  # Target 0: SOC
    soc_preds.sum().backward()

    saliency = x_tensor.grad.abs().squeeze(1).mean(dim=0).cpu().numpy()

    # Plot Saliency Map
    plt.figure(figsize=(10, 4))
    plt.plot(wavelengths, saliency, label="|∂SOC / ∂Reflectance|", color="navy", lw=1.5)
    
    diagnostic_bands = [
        (480, "Fe-Oxide"),
        (1400, "H2O"),
        (1720, "C-H (SOC)"),
        (1900, "H2O"),
        (2205, "Clay (Al-OH)"),
        (2310, "C=O / C-H")
    ]
    for wl, label in diagnostic_bands:
        plt.axvline(wl, color="red", linestyle="--", alpha=0.6)
        plt.text(wl + 8, np.max(saliency) * 0.85, label, rotation=90, fontsize=9, color="darkred")

    plt.title("Spectral Saliency Map: Diagnostic Feature Importance for SOC", fontsize=12)
    plt.xlabel("Wavelength (nm)", fontsize=11)
    plt.ylabel("Sensitivity Gradient", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    saliency_path = os.path.join(output_dir, "saliency_soc.png")
    plt.savefig(saliency_path, dpi=200)
    plt.close()
    print(f"  --> Saliency map saved to {saliency_path}")

    # 4. Residual Scatter Plots
    print("[*] Generating observed vs predicted residual plots...")
    with torch.no_grad():
        all_preds = model(torch.from_numpy(X_proc).unsqueeze(1).float()).cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    target_names = ["SOC (g/kg)", "Nitrogen (g/kg)", "Phosphorus (mg/kg)", "Potassium (mg/kg)"]

    for i, (name, ax) in enumerate(zip(target_names, axes.flat)):
        obs = y[:, i]
        prd = all_preds[:, i]
        ax.scatter(obs, prd, alpha=0.5, color="teal", edgecolors="none")
        min_v, max_v = min(obs.min(), prd.min()), max(obs.max(), prd.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.2, label="1:1 line")
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Observed")
        ax.set_ylabel("Predicted")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.suptitle("Observed vs Predicted Residual Analysis", fontsize=13)
    plt.tight_layout()
    residuals_path = os.path.join(output_dir, "residuals.png")
    plt.savefig(residuals_path, dpi=200)
    plt.close()
    print(f"  --> Residuals plot saved to {residuals_path}")
    print("[*] Phase 3 Analysis complete!")


if __name__ == "__main__":
    run_saliency_and_residuals()
