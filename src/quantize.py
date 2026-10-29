"""Full-Pipeline and Edge Inference Latency Profiler for UAV Deployment.

Measures:
1. End-to-End Latency: Preprocessing (Savitzky-Golay + Continuum Removal) + Inference.
2. Isolated ONNX Runtime CPU Latency vs PyTorch FP32.
3. INT8 Dynamic Quantization memory footprint.
"""

import argparse
import os
import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.cnn1d import SoilSpectralCNN
from src.preprocessing.pipeline import SpectralPreprocessor


def benchmark_end_to_end(
    checkpoint_path: str = "models/best_model.pt",
    onnx_path: str = "models/soil_cnn.onnx",
    int8_path: str = "models/soil_cnn_int8.pt",
    n_bands: int = 486,
    n_runs: int = 50
):
    print("=" * 65)
    print("  AEROSOIL AI: END-TO-END LATENCY & HARDWARE PROFILER")
    print("=" * 65)

    # 1. Profile Preprocessing Pipeline
    preprocessor = SpectralPreprocessor(sg_window=11, sg_poly=2, continuum_removal=True, scale=True)
    dummy_spectrum = np.random.uniform(0.1, 0.8, size=(n_bands,)).astype(np.float64)
    wavelengths = np.linspace(400.0, 2450.0, n_bands)
    preprocessor.fit(dummy_spectrum.reshape(1, -1), wavelengths)

    # Warmup
    for _ in range(5):
        _ = preprocessor.transform(dummy_spectrum, wavelengths)

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = preprocessor.transform(dummy_spectrum, wavelengths)
    prep_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    # 2. Profile PyTorch Model
    model = SoilSpectralCNN(n_bands=n_bands, n_targets=4)
    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()

    sample_tensor = torch.randn(1, 1, n_bands, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(10):
            _ = model(sample_tensor)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = model(sample_tensor)
        pytorch_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    # 3. Dynamic INT8 Quantization
    quantized_model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    Path(int8_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(quantized_model.state_dict(), int8_path)

    with torch.no_grad():
        for _ in range(10):
            _ = quantized_model(sample_tensor)
        t0 = time.perf_counter()
        for _ in range(n_runs):
            _ = quantized_model(sample_tensor)
        int8_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0

    # 4. Profile ONNX Runtime Engine
    onnx_ms = None
    if Path(onnx_path).exists():
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            inp_name = session.get_inputs()[0].name
            inp_data = sample_tensor.numpy()
            for _ in range(10):
                _ = session.run(None, {inp_name: inp_data})
            t0 = time.perf_counter()
            for _ in range(n_runs):
                _ = session.run(None, {inp_name: inp_data})
            onnx_ms = ((time.perf_counter() - t0) / n_runs) * 1000.0
        except Exception:
            pass

    # Total End-to-End Latency
    e2e_ms = prep_ms + (onnx_ms if onnx_ms is not None else pytorch_ms)
    fp32_size = os.path.getsize(checkpoint_path) / (1024 * 1024) if Path(checkpoint_path).exists() else 0.0
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024) if Path(onnx_path).exists() else 0.0
    int8_size = os.path.getsize(int8_path) / (1024 * 1024) if Path(int8_path).exists() else 0.0

    print(f"  Component Breakdown (Single Spectrum, 486 channels):")
    print(f"    * Preprocessing (SG + Continuum Removal) : {prep_ms:6.2f} ms")
    print(f"    * PyTorch FP32 CPU Inference            : {pytorch_ms:6.2f} ms")
    if onnx_ms is not None:
        print(f"    * ONNX Runtime Engine CPU Inference     : {onnx_ms:6.2f} ms")
    print(f"    * INT8 Quantized CPU Inference          : {int8_ms:6.2f} ms")
    print("-" * 65)
    print(f"  TOTAL END-TO-END PIPELINE LATENCY         : {e2e_ms:6.2f} ms")
    print(f"  Target Real-Time UAV Budget (< 50 ms)     : {'PASS' if e2e_ms < 50.0 else 'OPTIMIZE'}")
    print("-" * 65)
    print(f"  Model Footprints:")
    print(f"    * PyTorch Checkpoint (.pt)              : {fp32_size:.2f} MB")
    print(f"    * ONNX Runtime Engine (.onnx)           : {onnx_size:.2f} MB")
    print(f"    * INT8 Quantized Checkpoint (.pt)       : {int8_size:.2f} MB")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latency and Hardware Profiler")
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--onnx", default="models/soil_cnn.onnx")
    parser.add_argument("--int8", default="models/soil_cnn_int8.pt")
    parser.add_argument("--n_bands", type=int, default=486)
    args = parser.parse_args()

    benchmark_end_to_end(args.checkpoint, args.onnx, args.int8, args.n_bands)
