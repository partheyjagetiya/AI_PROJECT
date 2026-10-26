"""INT8 Dynamic Quantization and Latency Benchmarking for Edge UAV Deployment."""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import time
import numpy as np
import torch
import torch.nn as nn

from src.models.cnn1d import SoilSpectralCNN


def benchmark_model(model: nn.Module, sample_input: torch.Tensor, n_runs: int = 50) -> float:
    """Measures average inference latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(sample_input)

        start = time.perf_counter()
        for _ in range(n_runs):
            _ = model(sample_input)
        elapsed = time.perf_counter() - start

    return (elapsed / n_runs) * 1000.0


def quantize_and_benchmark(
    checkpoint_path: str = "models/best_model.pt",
    output_path: str = "models/soil_cnn_int8.pt",
    n_bands: int = 486,
    batch_size: int = 1
):
    print(f"[*] Loading model from {checkpoint_path}...")
    model = SoilSpectralCNN(n_bands=n_bands, n_targets=4)
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    else:
        print(f"  [!] Checkpoint not found at {checkpoint_path}, using architecture with initialized weights.")

    model.eval()

    # Dynamic Quantization (Linear layers quantized to INT8)
    print("[*] Applying Dynamic INT8 Quantization...")
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), output_path)
    print(f"  --> Saved INT8 quantized model to {output_path}")

    # File size comparison
    fp32_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024) if Path(checkpoint_path).exists() else 0.0
    int8_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # Latency benchmarking
    x = torch.randn(batch_size, 1, n_bands, dtype=torch.float32)
    lat_fp32 = benchmark_model(model, x)
    lat_int8 = benchmark_model(quantized_model, x)

    print("\n" + "=" * 50)
    print("EDGE DEPLOYMENT BENCHMARK SUMMARY")
    print("=" * 50)
    if fp32_size_mb > 0:
        print(f"  FP32 Checkpoint Size : {fp32_size_mb:.2f} MB")
    print(f"  INT8 Checkpoint Size : {int8_size_mb:.2f} MB")
    print(f"  FP32 Latency (batch={batch_size}) : {lat_fp32:.2f} ms")
    print(f"  INT8 Latency (batch={batch_size}) : {lat_int8:.2f} ms")
    print(f"  Target (< 50ms) Met   : {'PASS (Speed: < 50ms)' if lat_int8 < 50.0 else 'NEEDS OPTIMIZATION'}")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="INT8 Dynamic Quantization")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--output", default="models/soil_cnn_int8.pt")
    parser.add_argument("--n_bands", type=int, default=486)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    quantize_and_benchmark(args.checkpoint, args.output, args.n_bands, args.batch_size)
