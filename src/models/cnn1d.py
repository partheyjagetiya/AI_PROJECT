"""1D Convolutional Neural Network for Soil Vis-NIR Hyperspectral Property Estimation.

Predicts Soil Organic Carbon (SOC) and Nitrogen, Phosphorus, Potassium (NPK)
from preprocessed diffuse reflectance spectra.
"""

from typing import Optional
import torch
import torch.nn as nn


class SoilSpectralCNN(nn.Module):
    """1D-CNN for soil Vis-NIR hyperspectral spectroscopy."""

    def __init__(
        self,
        n_bands: int = 486,
        n_targets: int = 4,
        dropout: float = 0.3,
        base_channels: int = 64
    ):
        """
        Parameters
        ----------
        n_bands : int
            Number of spectral bands (default 486, e.g. Vis-NIR resampled).
        n_targets : int
            Number of output soil properties (typically 4: SOC, N, P, K).
        dropout : float
            Dropout probability for regularization.
        base_channels : int
            Number of initial feature map filters.
        """
        super().__init__()
        self.n_bands = n_bands
        self.n_targets = n_targets

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            # Block 1: Captures broad spectral baseline and continuum contours
            nn.Conv1d(1, c1, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2: Captures diagnostic absorption band slopes (OH, CH, NH overtones)
            nn.Conv1d(c1, c2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3: Sharp diagnostic peak / dip features (e.g. 1400, 1900, 2200 nm)
            nn.Conv1d(c2, c3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten()
        )

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(c3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 1, n_bands) or (batch_size, n_bands).
            
        Returns
        -------
        torch.Tensor
            Predictions of shape (batch_size, n_targets).
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        feat = self.features(x)
        out = self.regressor(feat)
        return out


class ResidualBlock1D(nn.Module):
    """Optional residual block for deeper soil spectral architectures."""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return self.relu(out + residual)
