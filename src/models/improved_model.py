from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for model training. Install torch first.") from exc

from .transformer import PositionalEncoding


class ConvTransformerForecaster(nn.Module):
    """CNN + Transformer model for household power forecasting.

    Input:
        x: [batch_size, input_len, feature_dim]

    Output:
        y_hat: [batch_size, horizon]·
    """

    def __init__(
        self,
        feature_dim: int,
        horizon: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        kernel_size: int = 5,
        pooling: str = "last",
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size should be odd to preserve sequence length.")

        if pooling not in {"last", "mean"}:
            raise ValueError("pooling must be either 'last' or 'mean'.")

        self.pooling = pooling
        padding = kernel_size // 2

        self.local_encoder = nn.Sequential(
            nn.Conv1d(feature_dim, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.BatchNorm1d(d_model),
            nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
        )

        self.position = PositionalEncoding(d_model=d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq_len, feature_dim]
        x = x.transpose(1, 2)

        # x: [batch, d_model, seq_len]
        x = self.local_encoder(x)

        # x: [batch, seq_len, d_model]
        x = x.transpose(1, 2)

        x = self.position(x)
        encoded = self.encoder(x)

        if self.pooling == "last":
            pooled = encoded[:, -1, :]
        else:
            pooled = encoded.mean(dim=1)

        return self.head(pooled)