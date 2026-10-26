"""Data loading and preprocessing utilities for LUCAS and OSSL Soil Spectral Databases.

Includes:
1. `load_lucas_csv`: Loads ESDAC LUCAS topsoil spectra and properties from CSV.
2. `generate_synthetic_soil_spectral_dataset`: Creates realistic synthetic Vis-NIR
   diffuse reflectance spectra and NPK/SOC ground truths for testing and pipeline validation.
"""

from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd


DEFAULT_WAVELENGTHS = np.linspace(400, 2500, 486)  # 400 - 2500 nm, 486 bands (approx 4.3 nm resolution)

# Official Limits of Detection (LOD) from Table 4 of JRC Report (2018 LUCAS Soil)
# Samples labeled '< LOD' are imputed at 0.5 * LOD (standard environmental chemometric protocol)
LUCAS_2018_LOD = {
    "OC": 2.0,       # Organic Carbon LOD: 2.0 g/kg (half = 1.0)
    "N": 0.2,        # Nitrogen LOD: 0.2 g/kg (half = 0.1)
    "P": 10.0,       # Phosphorus LOD: 10.0 mg/kg (half = 5.0)
    "K": 10.0,       # Potassium LOD: 10.0 mg/kg (half = 5.0)
}


def load_lucas_2018_properties(
    csv_path: str = "LUCAS-SOIL-2018-data-report-readme-v2/LUCAS-SOIL-2018-v2/LUCAS-SOIL-2018.csv"
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Loads and standardizes the official LUCAS 2018 Soil Properties dataset.
    
    Implements:
    - Cleaning of '< LOD' (Limit of Detection) strings using Table 4 of JRC 2018 Report (half-LOD imputation)
    - Target extraction: OC (SOC g/kg), N (g/kg), P (mg/kg), K (mg/kg)
    - Geospatial coordinate extraction: TH_LAT, TH_LONG
    
    Returns
    -------
    df : pd.DataFrame
        Full cleaned DataFrame of 18,984 samples with all soil and environmental covariates.
    y : np.ndarray
        Cleaned float32 targets [SOC, N, P, K], shape (n_samples, 4).
    coords : np.ndarray
        Geographic coordinates [latitude, longitude], shape (n_samples, 2).
    """
    file_path = Path(csv_path)
    if not file_path.exists():
        # Fallback check under data/raw
        alt_path = Path("data/raw/LUCAS-SOIL-2018.csv")
        if alt_path.exists():
            file_path = alt_path
        else:
            raise FileNotFoundError(f"LUCAS 2018 CSV not found at {csv_path}")

    df = pd.read_csv(file_path)

    # Clean targets and impute < LOD
    target_cols = ["OC", "N", "P", "K"]
    for col in target_cols:
        lod_val = LUCAS_2018_LOD[col]
        # Replace '< LOD' with 0.5 * LOD
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(["< LOD", "<LOD", "<lod", "< lod"], str(lod_val * 0.5))
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop any rows with NaN in primary targets
    valid_mask = df[target_cols].notna().all(axis=1) & df[["TH_LAT", "TH_LONG"]].notna().all(axis=1)
    df_clean = df[valid_mask].copy()

    y = df_clean[target_cols].to_numpy(dtype=np.float32)
    coords = df_clean[["TH_LAT", "TH_LONG"]].to_numpy(dtype=np.float32)

    return df_clean, y, coords


def load_lucas_data(
    spectra_path: str = "data/raw/lucas_topsoil_spectra.csv",
    properties_path: str = "data/raw/lucas_topsoil_properties.csv"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads LUCAS topsoil spectra and target properties.
    
    Returns
    -------
    X : np.ndarray
        Reflectance spectra, shape (n_samples, n_bands).
    y : np.ndarray
        Target soil properties [SOC, N, P, K], shape (n_samples, 4).
    wavelengths : np.ndarray
        Spectral band wavelengths in nm, shape (n_bands,).
    coordinates : np.ndarray
        Spatial coordinates [latitude, longitude], shape (n_samples, 2).
    """
    spectra_file = Path(spectra_path)
    props_file = Path(properties_path)

    if not spectra_file.exists() or not props_file.exists():
        raise FileNotFoundError(
            f"LUCAS dataset files not found at '{spectra_path}' or '{properties_path}'. "
            "Please download them from ESDAC or use generate_synthetic_soil_spectral_dataset() for pipeline testing."
        )

    df_spectra = pd.read_csv(spectra_file)
    df_props = pd.read_csv(props_file)

    # Harmonize sample IDs
    merged = pd.merge(df_spectra, df_props, on="sample_id", how="inner")

    # Extract spectral columns (assumes columns named by wavelength or prefixed 'wl_')
    spectral_cols = [c for c in merged.columns if c.startswith("wl_") or c.replace(".", "").isdigit()]
    wavelengths = np.array([float(c.replace("wl_", "")) for c in spectral_cols])
    X = merged[spectral_cols].to_numpy(dtype=np.float32)

    # Targets: SOC (g/kg), N (g/kg), P (mg/kg), K (mg/kg)
    target_cols = ["SOC", "N", "P", "K"]
    y = merged[target_cols].to_numpy(dtype=np.float32)

    coords = merged[["latitude", "longitude"]].to_numpy(dtype=np.float32) if {"latitude", "longitude"}.issubset(merged.columns) else None

    return X, y, wavelengths, coords


def generate_synthetic_soil_spectral_dataset(
    n_samples: int = 1200,
    n_bands: int = 486,
    random_state: int = 42,
    save_dir: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates realistic synthetic Vis-NIR soil diffuse reflectance spectra
    grounded in soil spectroscopy physics (Hapke/Kubelka-Munk principles).
    
    Simulates:
    - Base soil albedo variations (dark organic soils vs bright sandy/calcareous soils)
    - Organic matter absorption: darkening slope in visible and C-H absorption at ~1720 nm & ~2310 nm
    - Nitrogen correlation with organic matter
    - Soil moisture absorption at ~1400 nm and ~1900 nm with moisture-induced baseline depression
    - Clay absorption feature at ~2200 nm (Al-OH)
    - Random baseline drift and instrument Gaussian noise
    - European geographic coordinates (lat 36-60, lon -10-30)
    
    Returns
    -------
    X : np.ndarray, shape (n_samples, n_bands)
    y : np.ndarray, shape (n_samples, 4) -> [SOC, N, P, K]
    wavelengths : np.ndarray, shape (n_bands,)
    coordinates : np.ndarray, shape (n_samples, 2)
    """
    rng = np.random.default_rng(random_state)
    wavelengths = np.linspace(400, 2500, n_bands)

    # Generate ground truth soil properties
    # SOC: log-normal distribution, typical European topsoil 5 - 120 g/kg
    soc = np.exp(rng.normal(loc=3.2, scale=0.65, size=n_samples))  # ~ 10 - 80 g/kg
    soc = np.clip(soc, 3.0, 150.0)

    # Nitrogen: correlated with SOC (C:N ratio typically ~ 10:1 to 14:1)
    cn_ratio = rng.normal(loc=11.5, scale=1.5, size=n_samples)
    cn_ratio = np.clip(cn_ratio, 7.0, 20.0)
    nitrogen = (soc / cn_ratio) + rng.normal(0, 0.2, size=n_samples)
    nitrogen = np.clip(nitrogen, 0.2, 12.0)

    # Phosphorus: semi-correlated with fertilizer/mineral, log-normal
    phosphorus = np.exp(rng.normal(loc=2.8, scale=0.7, size=n_samples))  # ~ 5 - 80 mg/kg
    phosphorus = np.clip(phosphorus, 2.0, 120.0)

    # Potassium: exchangeable K
    potassium = np.exp(rng.normal(loc=5.1, scale=0.55, size=n_samples))  # ~ 50 - 400 mg/kg
    potassium = np.clip(potassium, 30.0, 600.0)

    y = np.column_stack([soc, nitrogen, phosphorus, potassium]).astype(np.float32)

    # Synthetic European coordinates
    lat = rng.uniform(37.0, 58.0, size=n_samples)
    lon = rng.uniform(-8.0, 26.0, size=n_samples)
    coordinates = np.column_stack([lat, lon]).astype(np.float32)

    # Simulate Vis-NIR reflectance curves
    X = np.empty((n_samples, n_bands), dtype=np.float32)
    norm_wl = (wavelengths - 400.0) / (2500.0 - 400.0)

    for i in range(n_samples):
        # Base albedo inversely proportional to SOC (darker soil with higher carbon)
        base_albedo = 0.55 - 0.22 * np.tanh(soc[i] / 40.0) + rng.normal(0, 0.02)
        curve = base_albedo * (0.35 + 0.65 * (norm_wl ** 0.5))

        # 1. Soil moisture absorption troughs (~1400 nm & ~1900 nm)
        moisture_level = rng.uniform(0.05, 0.35)
        curve -= 0.18 * moisture_level * np.exp(-((wavelengths - 1410) / 45) ** 2)
        curve -= 0.30 * moisture_level * np.exp(-((wavelengths - 1920) / 60) ** 2)

        # 2. SOC diagnostic absorption features:
        # C-H overtone at 1720 nm and C=O / C-H combination at 2300 nm
        soc_factor = min(1.0, soc[i] / 60.0)
        curve -= 0.035 * soc_factor * np.exp(-((wavelengths - 1725) / 25) ** 2)
        curve -= 0.045 * soc_factor * np.exp(-((wavelengths - 2310) / 35) ** 2)

        # 3. Clay mineral feature (Al-OH) at ~2205 nm
        clay_intensity = rng.uniform(0.02, 0.08)
        curve -= clay_intensity * np.exp(-((wavelengths - 2205) / 20) ** 2)

        # 4. Iron oxide absorption in visible (steep drop below 550 nm)
        curve -= 0.12 * np.exp(-((wavelengths - 480) / 80) ** 2)

        # 5. Additive baseline tilt / sensor drift
        drift = rng.uniform(-0.03, 0.03) * norm_wl + rng.uniform(-0.02, 0.02)
        curve += drift

        # 6. Instrument detector noise
        noise = rng.normal(0, 0.003, size=n_bands)
        curve += noise

        # Reflectance bounds [0.01, 0.95]
        X[i] = np.clip(curve, 0.01, 0.95).astype(np.float32)

    if save_dir:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path / "synthetic_soil_dataset.npz",
            X=X, y=y, wavelengths=wavelengths, coordinates=coordinates
        )

    return X, y, wavelengths, coordinates
