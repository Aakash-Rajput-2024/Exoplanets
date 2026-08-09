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

ILR — required by anything that puts a DENSITY on the labels
    CLR output is exactly zero-sum, so the 12 coordinates are rank-11 and any
    probability density over them is singular: a normalizing flow, a KDE prior, or a
    nested sampler's unit-cube transform will diverge or silently collapse on it. That
    is fine for the point-estimating regressors (they never need a density) but not for
    the Bayesian tiers in ``evaluation.bayes`` / ``models.npe``.

    ``to_ilr`` / ``from_ilr`` map to an orthonormal basis of the zero-sum hyperplane
    (Helmert contrasts), giving 11 unconstrained coordinates with a non-degenerate
    density. The round trip through ``from_ilr`` -> softmax lands on the simplex
    interior by construction, so every posterior sample is a valid composition — the
    same guarantee CLR gives the regressors.
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


def helmert_basis(d: int = 12, dtype=torch.float32, device=None) -> torch.Tensor:
    """Orthonormal basis of {t ∈ R^d : Σ t = 0}, as a (d-1, d) matrix H.

    Satisfies H @ Hᵀ = I_{d-1} and H @ 1 = 0, so Hᵀ @ H is the projector onto the
    zero-sum hyperplane. Row k contrasts the first k coordinates against the (k+1)-th:

        H[k] = (1, …, 1, −k, 0, …, 0) / sqrt(k(k+1))     (k ones)

    Built in float64 and cast at the end — the 1/sqrt(k(k+1)) normalisation loses
    orthogonality in float32 for the later rows otherwise.
    """
    H = torch.zeros(d - 1, d, dtype=torch.float64)
    for k in range(1, d):
        H[k - 1, :k] = 1.0
        H[k - 1, k] = -float(k)
        H[k - 1] /= (k * (k + 1.0)) ** 0.5
    return H.to(dtype=dtype, device=device)


def to_ilr(y_lin: torch.Tensor, H: torch.Tensor | None = None) -> torch.Tensor:
    """Linear mole fractions (..., d) -> ILR coordinates (..., d-1).

    Composition of the CLR map with a rotation into the Helmert basis, so it inherits
    CLR's scale invariance while dropping the redundant coordinate.
    """
    t = _transform(y_lin, "clr")
    if H is None:
        H = helmert_basis(y_lin.shape[-1], dtype=t.dtype, device=t.device)
    return t @ H.T


def from_ilr(z: torch.Tensor, H: torch.Tensor | None = None) -> torch.Tensor:
    """ILR coordinates (..., d-1) -> linear mole fractions (..., d) on the simplex.

    Exact inverse of ``to_ilr`` for any composition: rotating back gives the CLR vector
    (already zero-sum by construction, since Hᵀz lies in the hyperplane) and softmax is
    CLR's inverse. Output is strictly positive and sums to 1 for ANY finite z, which is
    what makes it safe as a sampler/flow output space.
    """
    if H is None:
        H = helmert_basis(z.shape[-1] + 1, dtype=z.dtype, device=z.device)
    return torch.softmax(z @ H, dim=-1)


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
