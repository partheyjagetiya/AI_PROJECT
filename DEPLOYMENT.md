# Field Spectrometer & UAV Hyperspectral Integration Guide

## 1. Soil Organic Carbon & NPK Estimation Pipeline

This pipeline neutralization protocol compensates for:
- Soil moisture scattering and water vapor interference (~1400 nm & ~1900 nm)
- Instrument baseline drift
- Variable soil surface roughness / grain size

---

## 2. Preprocessing Parameters (Must Match Training)

When feeding in raw reflectance curves from field ASD, Ocean Optics, or UAV hyperspectral cameras (e.g. Headwall, Corning microHSI, Cubert):

| Stage | Method | Configuration / Parameters |
|---|---|---|
| **Smoothing** | Savitzky-Golay | `window_length=11`, `polyorder=2`, `deriv=0` |
| **Scatter Correction** | SNV | Standard Normal Variate (mean 0, std 1 per spectrum) |
| **Baseline Drift** | ALS | $\lambda = 10^5$, $p = 0.001$, `niter=15` |
| **Continuum Removal** | Upper Convex Hull | Isolated absorption ranges: `(1350-1500nm)`, `(1650-1750nm)`, `(1850-2050nm)`, `(2150-2300nm)` |
| **Water Band Mask** | UAV / Field | Deep moisture absorption masks: `1350-1420 nm` & `1800-1950 nm` (if saturated) |

---

## 3. High-Speed Edge Inference (Python & ONNX Runtime)

```python
import numpy as np
import onnxruntime as ort
from src.preprocessing.pipeline import SpectralPreprocessor

# 1. Initialize ONNX runtime session (runs under 10ms on CPU)
session = ort.InferenceSession("models/soil_cnn.onnx", providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

# 2. Input spectrum: (486 bands, 400 - 2500 nm)
# Preprocess with the trained preprocessor
preprocessor = ... # Loaded fitted preprocessor
proc_spectrum = preprocessor.transform(raw_spectrum, wavelengths)

# 3. Predict
input_tensor = proc_spectrum.astype(np.float32).reshape(1, 1, -1)
outputs = session.run(None, {input_name: input_tensor})[0][0]

soc, nitrogen, phosphorus, potassium = outputs
print(f"SOC: {soc:.2f} g/kg | N: {nitrogen:.2f} g/kg | P: {phosphorus:.2f} mg/kg | K: {potassium:.2f} mg/kg")
```

---

## 4. UAV Hyperspectral Camera Adaptation

1. **Spectral Resampling**:
   - UAV sensors typically capture 100–270 channels. Interpolate raw bands to the 486 training bands via cubic spline (`scipy.interpolate.CubicSpline`) or resample the 1D-CNN input channels.
2. **Atmospheric Correction**:
   - Convert UAV radiance ($L$) to surface reflectance ($\rho$) using empirical line method or ATCOR before running baseline correction.
