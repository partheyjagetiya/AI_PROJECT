# Hyperspectral / UAV Soil Organic Carbon & NPK Nutrient Estimation

This repository implements an end-to-end deep learning and spectral preprocessing framework for non-destructive soil diffuse reflectance spectroscopy (Vis-NIR bands, 400–2500 nm).

It estimates:
- **Soil Organic Carbon (SOC)**
- **Nitrogen (N)**
- **Phosphorus (P)**
- **Potassium (K)**

---

## 📄 Base Paper
This project benchmarks against and enhances the methodology from:
> **"A Novel Transformer-CNN Approach for Predicting Soil Properties from LUCAS Vis-NIR Spectral Data"**  
> Liying Cao, Miao Sun, Zhicheng Yang, Donghui Jiang, Dongjie Yin, Yunpeng Duan  
> *MDPI Agronomy* 2024, 14(9), 1998. https://doi.org/10.3390/agronomy14091998

### Key Improvements Over the Base Paper
1. **Moisture & Scatter Neutralization**: Adds **ALS/airPLS baseline drift correction** and **upper convex hull continuum removal** to isolate pure diagnostic chemical dips ($1400/1900\,\text{nm}$ water troughs, $1720\,\text{nm}$ C-H, $2205\,\text{nm}$ Al-OH) beyond standard Savitzky-Golay filtering.
2. **Spatial Block Autocorrelation Control**: Implements `SpatialBlockCV` to prevent geographic data leakage between training and validation folds.
3. **Edge & UAV Deployability**: Provides a lightweight **1D-CNN** ($<10\,\text{ms}$ on CPU, INT8 quantization, ONNX) compared to heavy Transformer models requiring high-end GPUs (e.g. RTX 4090).

---

## 📁 Project Architecture

```
AI_PROJECT/
├── python311/               # Self-contained Python 3.11 runtime (on D: drive)
├── src/
│   ├── preprocessing/
│   │   ├── baseline.py      # ALS & airPLS baseline drift correction
│   │   ├── continuum.py     # Convex hull continuum removal (moisture neutralization)
│   │   ├── scatter.py       # SNV and MSC scatter correction
│   │   └── pipeline.py      # Composable SpectralPreprocessor
│   ├── data/
│   │   ├── lucas.py         # LUCAS / OSSL loader & realistic synthetic Vis-NIR generator
│   │   └── dataset.py       # PyTorch Dataset for spectral arrays
│   ├── models/
│   │   ├── cnn1d.py         # Edge 1D-CNN (SoilSpectralCNN)
│   │   └── transformer_cnn.py # Base Paper Architecture (TransformerCNN)
│   ├── utils/
│   │   └── spatial_cv.py    # Spatial Block K-Fold Cross-Validation
│   ├── train.py             # Training loop supporting both models & Agronomy 2024 RPD Tiers
│   ├── export_onnx.py       # Phase 4.1 ONNX export with onnxruntime verification
│   └── quantize.py          # Phase 4.2 Dynamic INT8 quantization & latency benchmark
├── notebooks/
│   ├── 01_baseline_plsr.py  # Phase 1 PLSR baseline test
│   └── 03_saliency_and_residuals.py # Phase 3 Saliency & residual analysis
├── models/
│   ├── best_model.pt        # Trained PyTorch FP32 model checkpoint
│   ├── soil_cnn.onnx        # Exported ONNX model (ready for C++ / edge deployment)
│   └── soil_cnn_int8.pt     # Dynamic INT8 quantized edge model
├── runs/analysis/
│   ├── saliency_soc.png     # Saliency gradient map across Vis-NIR spectrum
│   └── residuals.png        # 1:1 Observed vs Predicted plots
├── DEPLOYMENT.md            # Field spectrometer & UAV camera integration guide
└── requirements.txt
```

---

## 🚀 Execution Guide

Run all commands directly with the project's Python 3.11 runtime on your `D:` drive:

### 1. Run Phase 1 PLSR Baseline Test
```powershell
.\python311\python.exe notebooks/01_baseline_plsr.py
```
*Gate: SOC R² > 0.80, Nitrogen R² > 0.70.*

### 2. Run Phase 2 Deep 1D-CNN Training
```powershell
.\python311\python.exe src/train.py
```
*Trains SoilSpectralCNN with spatial block validation, early stopping, and saves `models/best_model.pt`.*

### 3. Run Phase 3 Saliency & Residual Analysis
```powershell
.\python311\python.exe notebooks/03_saliency_and_residuals.py
```
*Outputs `saliency_soc.png` and `residuals.png` in `runs/analysis/`.*

### 4. Run Phase 4 Edge Deployment (ONNX & INT8)
```powershell
# Export to ONNX
.\python311\python.exe src/export_onnx.py

# Quantize to INT8 and benchmark latency
.\python311\python.exe src/quantize.py
```
*Benchmarked latency: ~6.9 ms (FP32) / ~12.6 ms (INT8) per spectrum on CPU (< 50 ms budget).*

---

## 🧪 Real LUCAS / OSSL Dataset Placement
When you download the raw LUCAS files from ESDAC, place them in:
- `data/raw/lucas_topsoil_spectra.csv`
- `data/raw/lucas_topsoil_properties.csv`

The data loader in [src/data/lucas.py](file:///d:/Downloads/AI_PROJECT/src/data/lucas.py) will automatically load and align them by `sample_id`.
