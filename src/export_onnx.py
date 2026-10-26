"""Export Soil Hyperspectral 1D-CNN Model to ONNX format for embedded/UAV edge deployment."""

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
    print(f"[*] Loading checkpoint from {checkpoint_path}...")
    ckpt_file = Path(checkpoint_path)
    
    model = SoilSpectralCNN(n_bands=n_bands, n_targets=4)
    if ckpt_file.exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
        print("  --> Loaded model weights successfully.")
    else:
        print(f"  [!] Checkpoint not found at {checkpoint_path}. Exporting architecture with initialized weights.")

    model.eval()

    dummy_input = torch.randn(1, 1, n_bands, dtype=torch.float32)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Exporting ONNX model to {output_path}...")
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
    print("  --> ONNX export complete.")

    # Verification with onnxruntime if available
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outs = session.run(None, ort_inputs)
        print("  --> ONNXRuntime test inference passed!")
        print(f"  --> Output shape: {ort_outs[0].shape} | Sample output: {ort_outs[0][0]}")
    except ImportError:
        print("  [*] onnxruntime not yet installed; skipping onnxruntime verification step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export SoilSpectralCNN to ONNX")
    parser.add_argument("--checkpoint", default="models/best_model.pt", help="Path to PyTorch checkpoint")
    parser.add_argument("--output", default="models/soil_cnn.onnx", help="Path to output ONNX file")
    parser.add_argument("--n_bands", type=int, default=486, help="Number of spectral bands")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output, args.n_bands)
