"""Gaussian contrast likelihood of an observation against a library of spectra.

THE LIKELIHOOD
    The nets are trained on  C_noisy = C + (σ_C/α)·ε ,  ε~N(0,1)  (common.observable),
    so the matching per-observation likelihood of a candidate spectrum C_j is

        log L_ij = -½ Σ_λ [ (C_obs_i(λ) − C_j(λ)) / σ_i(λ) ]² + const,   σ_i = σ_C_i/α

    The const (−Σ log σ) is dropped: it is independent of j, and every consumer
    normalizes the weights over j.

WHY IT FACTORS
    Expanding the square turns an [N_obs, N_lib, L] tensor into two matmuls, which is
    what makes 11k observations × 88k library spectra × 4378 bins tractable at all.
    Writing o = C_obs − C̄ and δ_j = C_j − C̄ for a common per-λ reference C̄:

        log L_ij = -½ [ Σ_λ o_i²w_i  −  2 (o_i·w_i) @ δ_jᵀ  +  w_i @ (δ_j²)ᵀ ]

    ⚠ THE TRAP, and it is a real one — MEASURED, not hypothetical. The first term is
    the noise energy ‖ε‖² ~ L ~ 4e3·α², enormous and IDENTICAL for every j, while the
    discriminating signal is the small remainder. Computing it and subtracting is
    catastrophic cancellation. In float32 the resulting absolute error in log L is:

        α = 1  → 0.064  (weights off by 7%,     tolerable)
        α = 3  → 0.53   (weights off by 70%,    NOT tolerable)
        α = 10 → 7.4    (weights off by 1585×,  garbage)
        α = 300→ 4.8e3  (weights off by e^4800, total loss)

    THE FIX IS TO NOT COMPUTE IT. Σ_λ o_i²w_i depends only on i, so it is an additive
    per-row constant, and every consumer normalizes over j (``normalized_log_weights``)
    — it cancels exactly. Dropping it leaves two terms that are both O(the
    discriminating scale) rather than O(the noise energy), and the cancellation
    disappears. Centering on C̄ shrinks them further.

    Even so, use float64 (the default). It is a CPU-only path — MPS has no float64 —
    and the accuracy is worth far more here than the device. ``check_factorization``
    reports ABSOLUTE error, because that is what maps to a weight ratio e^err; a
    relative-error check looks reassuring and hides exactly this failure.
"""
from __future__ import annotations

import numpy as np
import torch

# Below this ESS a per-observation posterior is 1-nearest-neighbour in disguise: the
# weights have collapsed onto a few library members and the "posterior" carries no
# width. 200 gives a Monte-Carlo standard error of ~1/√200 ≈ 7% on the posterior mean,
# which is small against the RMSE being measured. Measured on INARA: ESS ~1e4 at α=1,
# ~1e2 at α=3, and 1 by α=10 — so this threshold is what separates the regime where T0
# is a ceiling from the regime where it is a meaningless 1-NN readout.
ESS_MIN = 200.0


def _as_t(x, dtype=torch.float32):
    return x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x), dtype=dtype)


def library_center(C_lib):
    """Per-λ reference subtracted from both sides before expanding the square.

    Any per-λ vector works (the difference C_obs − C_j is invariant); the median is
    used because it is robust to the library's heavy-tailed contrast distribution and
    minimises the magnitude of the expanded terms.
    """
    return C_lib.median(dim=0).values


def gaussian_logl(C_obs, sigma, C_lib, center=None, chunk=8192, device="cpu",
                  dtype=torch.float64, return_device="cpu"):
    """log L[i, j] of observation i under library spectrum j, up to a per-i constant.

    C_obs : [N, L]   noisy observed contrast
    sigma : [N, L]   per-λ 1σ contrast noise ALREADY divided by α
    C_lib : [M, L]   library of candidate (noiseless) contrasts
    center: [L]      optional per-λ reference; defaults to ``library_center(C_lib)``

    The returned values are shifted by an arbitrary per-row constant (the dropped
    Σ o²w term). That is intentional and safe — see the module docstring — but it
    means the output is ONLY meaningful after ``normalized_log_weights``. Do not
    compare rows against each other, and do not interpret it as an absolute log
    evidence.

    ``dtype`` defaults to float64 and ``device`` to cpu because float32 corrupts the
    weights for α ≳ 3 and MPS cannot do float64. Pass float32/mps only for a fast
    exploratory pass at α ≤ 1, and check with ``check_factorization``.

    ``chunk`` blocks the LIBRARY, not the observations. Casting the whole library to
    float64 and squaring it materialises 2·M·L·8 bytes — 6.2 GB for INARA's 88k×4378
    train split — which on an 18 GB machine OOM-kills the process once the 3.1 GB train
    tensor and the [N, M] output are also resident (measured: SIGKILL at ~12 GB peak).
    Blocking the library caps that term at 2·chunk·L·8 (0.6 GB at the default) and costs
    nothing, because each library row is still cast exactly once. The per-observation
    terms W and OW are [N, L] — 0.8 GB even for the full 11k test split — so they are
    built once up front rather than blocked.
    """
    C_obs, sigma, C_lib = _as_t(C_obs), _as_t(sigma), _as_t(C_lib)
    if center is None:
        center = library_center(C_lib)
    center = _as_t(center).to(device=device, dtype=dtype)

    N, M = C_obs.shape[0], C_lib.shape[0]
    # σ is a physical 1σ and strictly positive, but the cache can carry exact zeros at
    # bins where the noise column is 0 (see data.TRIM_TAIL_BINS); those would make w
    # infinite and poison the whole row, so clamp.
    W = 1.0 / torch.clamp(sigma.to(device=device, dtype=dtype), min=1e-30) ** 2   # [N, L]
    OW = (C_obs.to(device=device, dtype=dtype) - center) * W                      # [N, L]
    out = torch.empty(N, M, dtype=dtype, device=return_device)

    for j in range(0, M, chunk):
        Lib = (C_lib[j:j + chunk].to(device=device, dtype=dtype) - center)        # [b, L]
        # NOTE the missing +Σ o²w term — dropped on purpose (constant in j).
        ll = -0.5 * (W @ (Lib * Lib).T - 2.0 * (OW @ Lib.T))
        out[:, j:j + chunk] = ll.to(return_device)
        del Lib, ll
    return out


def logl_direct(C_obs, sigma, C_lib):
    """Unexpanded float64 reference: -½ Σ ((C_obs − C_lib)/σ)². O(N·M·L) memory.

    Only for verification on small N — this is the form ``gaussian_logl`` must match
    up to a per-row additive constant.
    """
    C_obs, sigma, C_lib = (_as_t(x).double() for x in (C_obs, sigma, C_lib))
    w = 1.0 / torch.clamp(sigma, min=1e-30) ** 2
    d = C_obs[:, None, :] - C_lib[None, :, :]
    return -0.5 * ((d * d) * w[:, None, :]).sum(-1)


def check_factorization(C_obs, sigma, C_lib, tol=1e-2, **kw):
    """Max ABSOLUTE error of the factored form vs the direct float64 form, after
    row-centering (the only thing the weights depend on).

    Absolute, not relative: an error of ``e`` in log L is a weight ratio of exp(e), so
    ``e`` IS the quantity of interest. A relative check against the row scale looks
    reassuring at 1e-6 while the weights are off by 1585× — that is precisely how the
    float32 path was nearly shipped. Returns (max_abs_err, max_weight_ratio, ok).
    """
    a = gaussian_logl(C_obs, sigma, C_lib, return_device="cpu", **kw).double()
    b = logl_direct(C_obs, sigma, C_lib)
    a = a - a.max(dim=1, keepdim=True).values
    b = b - b.max(dim=1, keepdim=True).values
    err = float((a - b).abs().max())
    return err, float(np.exp(min(err, 700.0))), err <= tol


def normalized_log_weights(logl):
    """Row-normalized log weights: log w_ij = logL_ij − logsumexp_j logL_ij."""
    return logl - torch.logsumexp(logl, dim=1, keepdim=True)


def ess(logw):
    """Kish effective sample size per row, ESS = 1 / Σ_j w_ij².

    ESS ≈ M means the likelihood is flat and the posterior is the prior (no
    information). ESS ≈ 1 means all weight sits on one library member — the estimate
    has degenerated to nearest-neighbour and its "posterior" is meaningless. Only the
    band in between is a resolved posterior.
    """
    return torch.exp(-torch.logsumexp(2.0 * logw, dim=1))


def weighted_mean(logw, values):
    """Σ_j w_ij · values[j] → [N, D]. ``values`` is the library's label matrix [M, D]."""
    return torch.exp(logw) @ _as_t(values, dtype=logw.dtype)


def weighted_quantile(logw, values, q):
    """Per-observation, per-dimension weighted quantile → [N, D].

    Loops over observations: the sort is [M log M] per row and this is only used for
    credible intervals on a reporting subset, never in an inner loop.
    """
    w = torch.exp(logw)
    vals = _as_t(values, dtype=logw.dtype)
    order = torch.argsort(vals, dim=0)                     # [M, D]
    out = torch.empty(w.shape[0], vals.shape[1], dtype=logw.dtype)
    for i in range(w.shape[0]):
        for d in range(vals.shape[1]):
            idx = order[:, d]
            cw = torch.cumsum(w[i][idx], dim=0)
            k = int(torch.searchsorted(cw, torch.tensor(q, dtype=cw.dtype)).clamp(max=len(idx) - 1))
            out[i, d] = vals[idx[k], d]
    return out


def resample(logw, values, n_samples, seed=0):
    """Draw ``n_samples`` posterior samples per observation → [S, N, D].

    This is the shape ``evaluation.metrics_extra`` (PIT / SBC / TARP / coverage /
    reliability) consumes, so a T0 posterior drops straight into the calibration
    battery that currently runs on MC-dropout draws.

    Sampling is WITH replacement from the library under the normalized weights, which
    is the standard SIR step. Where ESS is small the draws will be near-duplicates —
    that is the honest representation of a collapsed posterior, not a bug, and callers
    are expected to gate on ``ess`` before interpreting them.
    """
    g = torch.Generator().manual_seed(int(seed))
    w = torch.exp(logw)                                     # [N, M]
    idx = torch.multinomial(w, n_samples, replacement=True, generator=g)   # [N, S]
    vals = _as_t(values, dtype=logw.dtype)
    return vals[idx].permute(1, 0, 2).contiguous()          # [S, N, D]
