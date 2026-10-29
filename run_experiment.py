"""Unified Experiment Runner: 5-Fold Spatial CV, PLSR Benchmark, Preprocessing Ablation, and ONNX Export.

Usage:
  python run_experiment.py --synthetic --epochs 30
  python run_experiment.py --props data/raw/LUCAS-SOIL-2018.csv --spectra data/raw/lucas_topsoil_spectra.csv
"""

import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import warnings
import numpy as np

from src.data.lucas import (
    load_lucas_with_spectra,
    load_lucas_2018_properties,
    generate_synthetic_soil_spectral_dataset,
    DEFAULT_WAVELENGTHS
)
from src.train import run_training
from src.ablation import run_ablation_study
from src.benchmark_plsr import run_plsr_5fold
from src.export_onnx import export_to_onnx
from src.quantize import benchmark_end_to_end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified AeroSoil AI Experiment Runner")
    parser.add_argument("--props", type=str, help="Path to LUCAS properties CSV")
    parser.add_argument("--spectra", type=str, help="Path to ESDAC raw spectra CSV")
    parser.add_argument("--synthetic", action="store_true", default=True, help="Validate pipeline using forward physical modeling")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation study")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory to save model artifacts")
    args = parser.parse_args()

    print("=" * 80)
    print("  AEROSOIL AI: UNIFIED REPRODUCIBLE EXPERIMENT SUITE")
    print("=" * 80)

    if args.props and args.spectra and Path(args.props).exists() and Path(args.spectra).exists():
        print(f"[*] Ingesting REAL ESDAC laboratory spectra from {args.spectra}...")
        X, y, wavelengths, coords = load_lucas_with_spectra(args.props, args.spectra)
    else:
        print("[*] Pipeline Validation Mode: Physical radiative simulation conditioned on LUCAS 2018 properties.")
        X, y, wavelengths, coords = generate_synthetic_soil_spectral_dataset(
            n_samples=args.samples, n_bands=486, save_dir="data/processed"
        )

    print(f"[*] Input Data Loaded: Spectra Matrix={X.shape}, Target Properties={y.shape}")

    # 1. Run PLSR Benchmark across 5 Folds
    print("\n[PHASE 1/4] Running 5-Fold PLSR SpatialBlockCV Baseline...")
    run_plsr_5fold(n_samples=min(args.samples, len(X)))

    # 2. Run Preprocessing Ablation Study
    if not args.skip_ablation:
        print("\n[PHASE 2/4] Running Preprocessing Ablation Study...")
        run_ablation_study(n_samples=min(args.samples, len(X)))

    # 3. Run 1D-CNN Training
    print("\n[PHASE 3/4] Training 1D-CNN Model across Spatial Block Folds...")
    model, metrics = run_training(
        X, y, wavelengths, coords,
        epochs=args.epochs,
        batch_size=32,
        save_dir=args.save_dir,
        model_name="cnn1d",
        full_5fold=False
    )

    # 4. Export and Verify ONNX Engine
    print("\n[PHASE 4/4] Serializing and Benchmarking Edge ONNX Engine...")
    export_to_onnx(
        checkpoint_path=f"{args.save_dir}/best_model.pt",
        output_path=f"{args.save_dir}/soil_cnn.onnx",
        n_bands=len(wavelengths)
    )
    benchmark_end_to_end(
        checkpoint_path=f"{args.save_dir}/best_model.pt",
        onnx_path=f"{args.save_dir}/soil_cnn.onnx",
        n_bands=len(wavelengths)
    )

    print("\n" + "=" * 80)
    print("  EXPERIMENT SUITE COMPLETED SUCCESSFULLY")
    print("=" * 80)
