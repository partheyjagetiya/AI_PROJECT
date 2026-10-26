"""PyTorch Implementation of the Transformer-CNN Architecture from:
Cao et al., "A Novel Transformer-CNN Approach for Predicting Soil Properties from LUCAS Vis-NIR Spectral Data",
Agronomy 2024, 14, 1998. https://doi.org/10.3390/agronomy14091998

This model runs a Multi-Head Attention Transformer branch in parallel with a 1D-CNN branch,
fusing local convolutional features with global sequence self-attention representations.
"""

import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Transformer Encoder Block with Multihead Attention, LayerNorm, and FeedForward."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # FeedForward layer as specified in Section 2.2.1 of Agronomy paper:
        # First layer: 4x embedding dimension with LeakyReLU
        # Second layer: 2x embedding dimension back to embed_dim
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-Attention + Residual + LayerNorm
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FeedForward + Residual + LayerNorm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TransformerCNN(nn.Module):
    """Parallel Transformer-CNN model from Cao et al. (Agronomy 2024).
    
    Combines:
    - 1D-CNN branch: captures local spectral peak/trough patterns
    - Transformer branch: models long-range multi-head attention dependencies between bands
    - Feature Fusion: concatenates CNN and Transformer vectors into a dense regressor head
    """

    def __init__(
        self,
        n_bands: int = 486,
        n_targets: int = 4,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_bands = n_bands
        self.n_targets = n_targets

        # ----------------------------------------------------
        # Branch 1: CNN Branch (Section 2.2.2)
        # 3 Conv1D layers with 64, 128, 256 filters, kernel_size=5, stride=2, AvgPool
        # ----------------------------------------------------
        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.AvgPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.AvgPool1d(2),

            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.01),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )

        # ----------------------------------------------------
        # Branch 2: Transformer Branch (Section 2.2.1)
        # Projects spectral sequence into embed_dim and applies self-attention
        # ----------------------------------------------------
        self.input_projection = nn.Linear(1, embed_dim)
        self.transformer_block1 = TransformerEncoderBlock(embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_block2 = TransformerEncoderBlock(embed_dim, num_heads=num_heads, dropout=dropout)
        self.transformer_pool = nn.AdaptiveAvgPool1d(1)

        # ----------------------------------------------------
        # Branch 3: Feature Fusion and Dense Regressor (Table 2)
        # Concatenates CNN (256) + Transformer (embed_dim) -> 256 + 64 = 320
        # ----------------------------------------------------
        fused_dim = 256 + embed_dim
        self.regressor = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Dropout(dropout),
            nn.Linear(128, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : torch.Tensor of shape (B, 1, n_bands) or (B, n_bands)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 1. CNN Feature Extraction
        cnn_feat = self.cnn_branch(x)  # Shape: (B, 256)

        # 2. Transformer Sequence Modeling
        # Reshape to (B, n_bands, 1) for sequence tokenization
        x_seq = x.permute(0, 2, 1)
        seq_embed = self.input_projection(x_seq)  # Shape: (B, n_bands, embed_dim)
        trans_out = self.transformer_block1(seq_embed)
        trans_out = self.transformer_block2(trans_out)
        
        # Pool across sequence length: (B, n_bands, embed_dim) -> (B, embed_dim, n_bands) -> (B, embed_dim)
        trans_feat = self.transformer_pool(trans_out.permute(0, 2, 1)).squeeze(-1)

        # 3. Parallel Feature Fusion
        fused = torch.cat([cnn_feat, trans_feat], dim=-1)

        # 4. Final Soil Properties Prediction
        output = self.regressor(fused)
        return output
