"""1D Convolutional Neural Network for Soil Vis-NIR Hyperspectral Property Estimation.

Predicts Soil Organic Carbon (SOC) and Nitrogen, Phosphorus, Potassium (NPK)
from preprocessed diffuse reflectance spectra.
Features optional dual global pooling (avg + max) to preserve peak absorption depths.
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
        base_channels: int = 64,
        pool: str = "avg"
    ):
        """
        Parameters
        ----------
        n_bands : int
            Number of spectral bands (default 486, 400 - 2450 nm).
        n_targets : int
            Number of output soil properties (SOC, N, P, K).
        dropout : float
            Dropout probability for regularization.
        base_channels : int
            Number of initial feature map filters.
        pool : str
            'avg' for standard global average pooling (0.98 MB model),
            'avgmax' for dual average + max pooling (preserves peak absorption depths).
        """
        super().__init__()
        assert pool in ("avg", "avgmax"), "pool must be 'avg' or 'avgmax'"
        self.n_bands = n_bands
        self.n_targets = n_targets
        self.pool = pool

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.features = nn.Sequential(
            # Block 1: Broad baseline contours
            nn.Conv1d(1, c1, kernel_size=11, padding=5, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 2: Diagnostic absorption slopes
            nn.Conv1d(c1, c2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            # Block 3: Sharp diagnostic peaks (1400, 1720, 1900, 2205, 2310 nm)
            nn.Conv1d(c2, c3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
        )

        feat_dim = c3 * (2 if pool == "avgmax" else 1)

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        f = self.features(x)
        g = f.mean(dim=2)  # Global average pool (ONNX ReduceMean)
        if self.pool == "avgmax":
            g = torch.cat([g, f.amax(dim=2)], dim=1)  # ONNX ReduceMax
        out = self.regressor(g)
        return out


class ResidualBlock1D(nn.Module):
    """Residual block for deeper soil spectral architectures."""

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
