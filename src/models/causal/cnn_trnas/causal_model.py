"""~2M-parameter backbone for the do-calculus (counterfactual-invariance) track.

Same I/O contract as every other track — ``(B, in_channels, L) -> (B, 12)`` in
CLR space — so it drops straight into ``common.train_runner`` / ``common.evaluate``
via the registry (track ``causal_xl``). It is a wider/deeper sibling of
``NasaInaraTransformer`` (d_model 192, 4 pre-LN GELU encoder layers, learned
attention pooling) sized to ~2M params so it has the capacity to actually exploit
the exact-intervention invariance objective:

    python -m common.train_runner causal_xl --cf-invariance --seed 0

trains it with L = task + λ·‖f(x) − f(x^{do(env=j)})‖² (see common/counterfactuals.py).
It is a plain retrieval net — the "causal" content lives entirely in the training
objective, not in any special layer — which is exactly the honest framing the audit
asked for (an exact-intervention invariance method on a standard backbone), now with
enough capacity to be a fair headline model rather than the 0.5M-param transformer.

Design notes:
  * Downsample length is computed ARITHMETICALLY (3× MaxPool2 ⇒ //8), not via a
    dummy forward pass — so construction consumes no global RNG (keeps the
    multi-seed study clean) and prints nothing.
  * BatchNorm in the stem, pre-LN transformer (norm_first=True) and GELU for
    stable optimisation at this depth on GPU.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class _SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):                       # x: (B, T, d)
        return x + self.pe[:, : x.size(1), :]


class _AttentionPool(nn.Module):
    """Learned attention pooling over the sequence (upgrade on plain GAP)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x):                        # x: (B, T, d)
        w = torch.softmax(self.score(x), dim=1)  # (B, T, 1)
        return (w * x).sum(dim=1)                # (B, d)


class NasaInaraCausalNet(nn.Module):
    """~2M-param CNN-stem → Transformer → attention-pool → MLP retrieval net."""

    def __init__(self, in_channels: int = 2, sequence_length: int = 4378,
                 d_model: int = 192, n_heads: int = 6, num_layers: int = 4,
                 dim_feedforward: int = 832, output_dim: int = 12, dropout: float = 0.1):
        super().__init__()

        # CNN stem: 3× MaxPool2 ⇒ 8× downsample; progressive kernels; BN + GELU.
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=7, padding=3), nn.BatchNorm1d(64), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2), nn.BatchNorm1d(128), nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, d_model, kernel_size=3, padding=1), nn.BatchNorm1d(d_model), nn.GELU(),
            nn.MaxPool1d(2),
        )
        ds_len = sequence_length
        for _ in range(3):                       # matches the 3 MaxPool1d(2) above
            ds_len //= 2

        self.pre_norm = nn.LayerNorm(d_model)
        self.pos = _SinusoidalPositionalEncoding(d_model, max_len=ds_len + 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = _AttentionPool(d_model)
        self.post_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        x = self.stem(x)               # (B, d_model, T)
        x = x.permute(0, 2, 1)         # (B, T, d_model)
        x = self.pre_norm(x)
        x = self.pos(x)
        x = self.encoder(x)
        x = self.pool(x)               # (B, d_model)
        x = self.post_norm(x)
        return self.head(x)            # (B, output_dim)


if __name__ == "__main__":
    m = NasaInaraCausalNet(in_channels=2, sequence_length=4378)
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    y = m(torch.randn(4, 2, 4378))
    print(f"params={n:,}  output={tuple(y.shape)}")
