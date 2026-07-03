"""Label-space transforms for compositional abundance targets (fixes C2).

The 12 mole fractions live on the simplex Δ¹¹ (they sum to exactly 1). Training
and reporting in *linear* space is doubly wrong: the loss gradient is dominated
~10 decades by the major gases (trace species go essentially unsupervised), and
Euclidean statistics are incoherent on a simplex. We therefore transform to a
log space before standardizing.

Three interchangeable transforms (choose per experiment; CLR is the default):

  linear : identity (the old behaviour; kept for the log-labels-off ablation).
  log10  : t = log10(x). Inverse 10**t is always positive. Field-standard
           "log mixing ratio"; per-species interpretable (RMSE in dex).
  clr    : centered log-ratio, t_i = ln x_i - mean_j ln x_j. The Aitchison
           isometry simplex -> hyperplane {sum t = 0}. Its inverse is softmax,
           and because the labels sum to 1, softmax(clr(x)) == x exactly, so
           predicting in CLR and softmax-decoding GUARANTEES a valid simplex
           (positive, sums to 1) — negative abundances become impossible and the
           N2 fill-gas is handled honestly.

A ``LabelPipeline`` bundles (transform -> per-species standardization fitted on
TRAIN). ``encode`` maps linear labels to standardized training targets;
``decode`` maps a model's standardized output back to linear mole fractions.
decode(encode(y)) == y (to float tolerance) for all three kinds.
"""

from __future__ import annotations

import torch

# Positivity floor: min label is ~9e-11, all strictly positive, so this only
# guards against exact zeros / numerical underflow — it never clips real data.
LOG_FLOOR = 1e-12
_STD_EPS = 1e-8


def _transform(y: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "linear":
        return y
    if kind == "log10":
        return torch.log10(torch.clamp(y, min=LOG_FLOOR))
    if kind == "clr":
        ly = torch.log(torch.clamp(y, min=LOG_FLOOR))
        return ly - ly.mean(dim=-1, keepdim=True)
    raise ValueError(f"unknown label transform {kind!r}")


def _inverse(t: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "linear":
        return t
    if kind == "log10":
        return torch.pow(torch.tensor(10.0, dtype=t.dtype, device=t.device), t)
    if kind == "clr":
        return torch.softmax(t, dim=-1)
    raise ValueError(f"unknown label transform {kind!r}")


class LabelPipeline:
    """transform + train-fitted per-species standardization for abundance labels."""

    def __init__(self, kind: str, mean: torch.Tensor, std: torch.Tensor):
        self.kind = kind
        self.mean = mean          # (1, D) in transformed space
        self.std = std            # (1, D)

    @classmethod
    def fit(cls, y_lin_train: torch.Tensor, kind: str) -> "LabelPipeline":
        t = _transform(y_lin_train.float(), kind)
        return cls(kind, t.mean(dim=0, keepdim=True), t.std(dim=0, keepdim=True))

    def encode(self, y_lin: torch.Tensor) -> torch.Tensor:
        """linear mole fractions -> standardized transformed training targets."""
        t = _transform(y_lin.float(), self.kind)
        return (t - self.mean) / (self.std + _STD_EPS)

    def decode(self, y_std: torch.Tensor) -> torch.Tensor:
        """standardized model output -> linear mole fractions."""
        t = y_std * (self.std + _STD_EPS) + self.mean
        return _inverse(t, self.kind)

    def state(self) -> dict:
        return {"kind": self.kind, "mean": self.mean, "std": self.std}

    @classmethod
    def from_state(cls, s: dict) -> "LabelPipeline":
        return cls(s["kind"], s["mean"], s["std"])
