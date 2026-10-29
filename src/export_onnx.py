"""Export Soil Hyperspectral 1D-CNN Model to ONNX format for embedded/UAV edge deployment.

Includes:
- Strict checkpoint validation (no silent fallback to random weights).
- Full numerical parity verification (assert_allclose between PyTorch and ONNX Runtime).
- Dynamic batch axis for variable streaming throughput.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.models.cnn1d import SoilSpectralCNN


def export_to_onnx(
    checkpoint_path: str = "models/best_model.pt",
    output_path: str = "models/soil_cnn.onnx",
    n_bands: int = 486,
    opset_version: int = 14
):
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(
            f"PyTorch checkpoint not found at '{checkpoint_path}'. "
            "Please run train.py first to train and produce the checkpoint."
        )

    print(f"[*] Loading trained weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = SoilSpectralCNN(n_bands=n_bands, n_targets=4)
    model.load_state_dict(state_dict)
    model.eval()
    print("  --> Model loaded and set to eval mode.")

    dummy_input = torch.randn(1, 1, n_bands, dtype=torch.float32)

    # PyTorch reference output
    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Exporting ONNX model to {output_path} (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["spectra"],
        output_names=["soil_properties"],
        dynamic_axes={
            "spectra": {0: "batch_size"},
            "soil_properties": {0: "batch_size"}
        },
        dynamo=False
    )
    print("  --> ONNX computation graph serialized successfully.")

    # Numerical Parity Verification with ONNX Runtime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)
        onnx_output = ort_outs[0]

        max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
        np.testing.assert_allclose(torch_output, onnx_output, rtol=1e-4, atol=1e-4)

        print("\n" + "=" * 55)
        print("ONNX NUMERICAL PARITY VERIFICATION: PASSED")
        print("=" * 55)
        print(f"  PyTorch Output Shape   : {torch_output.shape}")
        print(f"  ONNX Output Shape      : {onnx_output.shape}")
        print(f"  Max Absolute Difference: {max_abs_diff:.2e} (Strict Tolerance < 1e-4)")
        print("=" * 55)
    except ImportError:
        print("  [*] onnxruntime not installed; skipping parity verification.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SoilSpectralCNN to ONNX")
    parser.add_argument("--checkpoint", default="models/best_model.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default="models/soil_cnn.onnx", help="Path to output ONNX file")
    parser.add_argument("--n_bands", type=int, default=486, help="Number of spectral bands")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.n_bands)
