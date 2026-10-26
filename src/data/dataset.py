"""PyTorch Dataset for Soil Diffuse Reflectance Spectroscopy."""

from typing import Optional, Sequence
import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None
    Dataset = object
    _TORCH_AVAILABLE = False


class SoilSpectralDataset(Dataset):
    """PyTorch Dataset for 1D soil diffuse reflectance spectra."""

    def __init__(
        self,
        spectra: np.ndarray,
        targets: np.ndarray,
        wavelengths: Optional[np.ndarray] = None,
        preprocessor=None,
        indices: Optional[Sequence[int]] = None
    ):
        """
        Parameters
        ----------
        spectra : np.ndarray
            Reflectance spectra of shape (N, n_bands).
        targets : np.ndarray
            Target soil properties of shape (N, n_targets).
        wavelengths : np.ndarray, optional
            Wavelength array of shape (n_bands,).
        preprocessor : SpectralPreprocessor, optional
            Preprocessing pipeline instance.
        indices : sequence of int, optional
            Subset indices (for cross-validation folds).
        """
        if indices is not None:
            self.spectra = np.asarray(spectra[indices], dtype=np.float32)
            self.targets = np.asarray(targets[indices], dtype=np.float32)
        else:
            self.spectra = np.asarray(spectra, dtype=np.float32)
            self.targets = np.asarray(targets, dtype=np.float32)

        self.wavelengths = wavelengths if wavelengths is not None else np.linspace(400, 2500, self.spectra.shape[1])
        self.preprocessor = preprocessor

        # If preprocessor is already fitted, preprocess the entire dataset in memory for high-speed training
        if self.preprocessor is not None and getattr(self.preprocessor, "is_fitted_", False):
            self.processed_spectra = self.preprocessor.transform(self.spectra, self.wavelengths).astype(np.float32)
        else:
            self.processed_spectra = None

    def __len__(self) -> int:
        return len(self.spectra)

    def __getitem__(self, idx: int):
        if self.processed_spectra is not None:
            x = self.processed_spectra[idx]
        elif self.preprocessor is not None:
            x = self.preprocessor.transform(self.spectra[idx:idx+1], self.wavelengths)[0].astype(np.float32)
        else:
            x = self.spectra[idx]

        # Shape for 1D-CNN: (1, n_bands)
        x_tensor = torch.from_numpy(x).unsqueeze(0).float()
        y_tensor = torch.from_numpy(self.targets[idx]).float()

        return x_tensor, y_tensor
