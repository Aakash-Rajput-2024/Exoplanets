"""DecloudUNet1D — a 1D U-Net that restores the clear contrast spectrum.

INPUT   encoded observable  [B, in_channels, L]
          ch0 = clouded contrast (asinh-normed, the retrieval model's ch0)
          ch1 = stellar-SNR context (asinh-normed). The SNR channel is
                cloud-INVARIANT (its numerator is F_star = star_planet − planet,
                which the cloud leaves unchanged), so it is pure per-λ
                reliability side-info — free to hand the network.
OUTPUT  declouded contrast   [B, 1, L], produced as a RESIDUAL correction to ch0.

Why a residual with a zero-initialised head: the declouder starts as the identity
(output == clouded input), so early training only has to LEARN THE CORRECTION that
adds the muted band depth back — a much easier target than regressing the whole
spectrum from scratch, and numerically stable in the asinh-normed space.

The net is shape-agnostic in L: it pads the input up to a multiple of 2**depth for
the pooling/upsampling to line up, then crops the output back. Nothing here reads
data or touches the shared pipeline.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """GroupNorm with a divisor-safe group count (batch-size independent, so it
    behaves identically at train batch 256 and eval batch 4096)."""
    groups = max_groups
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class _ConvBlock(nn.Module):
    """Two width-preserving conv layers (GroupNorm + GELU). Kernel 7 gives a wide
    receptive field so the continuum context needed to un-mute a band is in view."""

    def __init__(self, c_in: int, c_out: int, kernel: int = 7):
        super().__init__()
        pad = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(c_in, c_out, kernel, padding=pad),
            _group_norm(c_out), nn.GELU(),
            nn.Conv1d(c_out, c_out, kernel, padding=pad),
            _group_norm(c_out), nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class DecloudUNet1D(nn.Module):
    """Residual 1D U-Net restorer.

    Parameters
    ----------
    in_channels : int
        Channels of the encoded observable fed in (2 for ``contrast_snr``).
    sequence_length : int, optional
        Accepted for a uniform ``Model(in_channels=, sequence_length=)`` construction
        signature (matches the retrieval tracks); unused — L is inferred at forward.
    base, depth, kernel : int
        Width of the first stage, number of down/up stages, conv kernel size.
    residual, res_scale : bool, float
        If ``residual`` the network predicts ``ch0 + res_scale * delta`` (identity at
        init because the output head is zero-initialised).
    """

    def __init__(self, in_channels: int = 2, sequence_length: int | None = None,
                 base: int = 32, depth: int = 3, kernel: int = 7,
                 residual: bool = True, res_scale: float = 1.0):
        super().__init__()
        self.in_channels = in_channels
        self.sequence_length = sequence_length
        self.depth = depth
        self.residual = residual
        self.res_scale = res_scale

        chs = [base * (2 ** i) for i in range(depth + 1)]   # e.g. [32, 64, 128, 256]
        self.inc = _ConvBlock(in_channels, chs[0], kernel)
        self.pool = nn.MaxPool1d(2)
        self.downs = nn.ModuleList(_ConvBlock(chs[i], chs[i + 1], kernel) for i in range(depth))

        self.reduce = nn.ModuleList(nn.Conv1d(chs[i], chs[i - 1], 1) for i in range(depth, 0, -1))
        self.ups = nn.ModuleList(_ConvBlock(chs[i - 1] * 2, chs[i - 1], kernel) for i in range(depth, 0, -1))

        self.outc = nn.Conv1d(chs[0], 1, 1)
        # Zero-init the head so a residual model is the identity at initialisation.
        nn.init.zeros_(self.outc.weight)
        nn.init.zeros_(self.outc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_contrast = x[:, 0:1, :]                 # keep the un-padded ch0 for the residual
        L = x.shape[-1]
        mult = 2 ** self.depth
        pad = (mult - L % mult) % mult
        h = F.pad(x, (0, pad), mode="replicate") if pad else x

        skips = [self.inc(h)]
        h = skips[0]
        for down in self.downs:
            h = down(self.pool(h))
            skips.append(h)

        for j in range(self.depth):
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            h = self.reduce[j](h)
            skip = skips[self.depth - 1 - j]
            h = self.ups[j](torch.cat([h, skip], dim=1))

        delta = self.outc(h)
        if pad:
            delta = delta[..., :L]
        return base_contrast + self.res_scale * delta if self.residual else delta


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # Shape self-test on a non-power-of-2 length (the real grid is 4378).
    net = DecloudUNet1D(in_channels=2)
    net.eval()
    for L in (4378, 4096, 1000):
        y = net(torch.randn(2, 2, L))
        assert y.shape == (2, 1, L), (L, y.shape)
    # Residual + zero-init head ⇒ the untrained net reproduces ch0 exactly.
    x = torch.randn(2, 2, 4378)
    identity = torch.allclose(net(x)[:, 0], x[:, 0], atol=1e-6)
    print(f"DecloudUNet1D OK — {count_params(net):,} params; identity at init: {identity}")
