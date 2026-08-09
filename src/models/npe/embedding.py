"""Spectrum embedding for the NPE flow's conditioning context.

WHY THIS ARCHITECTURE
    It is a deliberate copy of the ``optimized1dcnn`` trunk (same four conv blocks,
    same tanh→relu→relu→relu, same MaxPool schedule, same AdaptiveAvgPool1d(16), same
    256-wide fc1). The point is that the NPE's access to the spectrum is comparable BY
    CONSTRUCTION to the best-performing regressor, so an NPE-vs-CNN difference is
    attributable to the posterior head rather than to the feature extractor.

    It is replicated rather than imported because ``NasaInaraModel`` prints on
    construction, bakes in a 12-way regression head, and runs a dummy forward pass to
    size its linear layer — all of which are wrong for an embedding used inside a flow.

EXPOSURE CONDITIONING
    log10(α) is appended to the pooled features. The nets already see α implicitly
    through the stellar-SNR channel; the flow needs it explicitly because the posterior
    WIDTH is a strong function of exposure and an amortized flow must be told which
    exposure it is conditioning on. One flow then covers the whole sweep instead of
    seven separately-trained ones, and it can interpolate between them.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _mps_safe_adaptive_avg_pool1d(x, output_size):
    """AdaptiveAvgPool1d that also works on Apple MPS.

    MPS has no kernel for adaptive pooling when the input length is not divisible by
    the output size (pytorch#96056) — it raises rather than falling back. Same detour
    as ``models/optimized1dcnn/model.py``: run this one op on CPU, numerically
    identical, only on MPS.
    """
    if x.device.type == "mps" and x.shape[-1] % output_size != 0:
        return F.adaptive_avg_pool1d(x.cpu(), output_size).to(x.device)
    return F.adaptive_avg_pool1d(x, output_size)


class SpectrumEmbedding(nn.Module):
    """[B, C, L] encoded observable (+ scalar log10 α) → [B, out_dim + 1] context."""

    def __init__(self, in_channels: int = 2, out_dim: int = 256, pool: int = 16):
        super().__init__()
        self.pool_to = pool
        self.out_dim = out_dim
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc1 = nn.Linear(256 * pool, out_dim)
        self.pool1 = nn.MaxPool1d(2)
        self.pool2 = nn.MaxPool1d(2)
        self.pool3 = nn.MaxPool1d(2)

    @property
    def context_dim(self) -> int:
        return self.out_dim + 1        # + log10(alpha)

    def forward(self, x: torch.Tensor, log_alpha: torch.Tensor) -> torch.Tensor:
        h = self.pool1(torch.tanh(self.bn1(self.conv1(x))))
        h = self.pool2(F.relu(self.bn2(self.conv2(h))))
        h = self.pool3(F.relu(self.bn3(self.conv3(h))))
        h = F.relu(self.bn4(self.conv4(h)))
        h = _mps_safe_adaptive_avg_pool1d(h, self.pool_to)
        h = F.relu(self.fc1(h.flatten(1)))
        if log_alpha.dim() == 1:
            log_alpha = log_alpha.unsqueeze(1)
        return torch.cat([h, log_alpha.to(h.dtype)], dim=1)
