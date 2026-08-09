"""T2 sampler layer — nested sampling / MCMC over arbitrary θ with a swappable backend.

Pairs with ``emulator.py`` (fast, PSG-faithful, adequacy-tested) or with either physical
engine. Produces the same weighted-sample structure T0 does, so both tiers feed Section
K and the ``metrics_extra`` battery through one contract.

PARAMETER SPACE
    11 ILR coordinates + the physical nuisances. ILR, never CLR: CLR is exactly
    zero-sum, so a unit-cube prior transform onto 12 CLR coordinates is ill-defined and
    the sampler explores a measure-zero sheet. ``common.transforms.from_ilr`` maps any
    finite point to a strictly-positive composition summing to 1, so every live point is
    a valid planet by construction and no rejection step is needed.

THE PRIOR IS AN APPROXIMATION — and this is the honest difference from T0
    T0 uses INARA's prior EXACTLY, because its proposal *is* a draw from that prior.
    A sampler cannot do that: it needs a density it can invert, so ``EmpiricalPrior``
    fits a multivariate normal in whitened (ILR + scaled-nuisance) space. That captures
    the means, variances and — importantly — the cross-correlations between species and
    between composition and physical state, but it does not capture skew, multimodality,
    or hard bounds in the true INARA prior.

    Consequence: T2 and T0 will not agree perfectly even when both are working. The
    T0-vs-T2 comparison at α ∈ [1, 3], where T0 is exact, is therefore the calibration
    of T2's prior approximation, and any T2 number at high α inherits whatever
    disagreement is measured there. Run that check before quoting T2.

BACKENDS
    nautilus  — recommended. Neural-boosted importance nested sampling; needs roughly an
                order of magnitude fewer likelihood calls than dynesty/ultranest at this
                dimensionality, and returns weighted samples + ESS, i.e. exactly T0's
                structure.
    dynesty   — the field standard; use as an independent cross-check on a handful of
                targets (posterior means agreeing to <0.05 dex is the pass criterion).
    nestle    — already installed inside MultiREx-public/.venv as a TauREx dependency;
                the zero-install option for the cross-code tier.
"""
from __future__ import annotations

import numpy as np
import torch

from common.transforms import from_ilr, helmert_basis, to_ilr

SUPPORTED = ("nautilus", "dynesty", "nestle")


class EmpiricalPrior:
    """Multivariate-normal fit to INARA's joint (ILR composition, nuisance) prior."""

    def __init__(self, mean, cov, n_ilr=11, jitter=1e-8):
        self.mean = np.asarray(mean, dtype=np.float64)
        self.cov = np.asarray(cov, dtype=np.float64)
        self.n_ilr = int(n_ilr)
        d = self.cov.shape[0]
        self.chol = np.linalg.cholesky(self.cov + jitter * np.eye(d))

    @property
    def ndim(self):
        return self.mean.shape[0]

    @classmethod
    def fit(cls, y_lin, nuisance=None):
        """y_lin [N, 12] linear VMRs; nuisance [N, P] already on a sane (log) scale."""
        z = to_ilr(torch.as_tensor(np.asarray(y_lin), dtype=torch.float64)).numpy()
        X = z if nuisance is None else np.concatenate([z, np.asarray(nuisance)], axis=1)
        return cls(X.mean(axis=0), np.cov(X, rowvar=False), n_ilr=z.shape[1])

    def transform(self, u):
        """Unit hypercube → parameter vector (the nested-sampling prior transform).

        Gaussian quantile per whitened dimension, then the Cholesky factor re-imposes
        the empirical correlations. Vectorized over a leading batch axis if present.
        """
        from scipy.special import ndtri
        u = np.clip(np.asarray(u, dtype=np.float64), 1e-12, 1 - 1e-12)
        return self.mean + ndtri(u) @ self.chol.T

    def to_vmr(self, theta):
        """Parameter vector → linear 12-species VMR (the ILR block, softmax-decoded)."""
        t = torch.as_tensor(np.atleast_2d(theta)[:, :self.n_ilr], dtype=torch.float64)
        H = helmert_basis(self.n_ilr + 1, dtype=torch.float64)
        return from_ilr(t, H).numpy()


def make_loglike(C_obs, sigma, forward, batch=None):
    """Gaussian contrast log-likelihood closure for one observation.

    C_obs, sigma : [L] for this planet, σ ALREADY divided by α
    forward      : θ → C(λ). Must accept [P] and may accept [B, P] (then set ``batch``).

    Returns f(θ) → float (or [B] when vectorized), dropping the θ-independent
    normalisation. NaN/inf model output is mapped to -1e300 rather than propagating: a
    sampler that receives a NaN likelihood will either crash or silently accept the
    point, and both failure modes are hard to spot after the fact.
    """
    w = 1.0 / np.clip(np.asarray(sigma, dtype=np.float64), 1e-30, None) ** 2
    c = np.asarray(C_obs, dtype=np.float64)

    def logl(theta):
        m = np.asarray(forward(theta), dtype=np.float64)
        d = c - m
        out = -0.5 * np.sum(d * d * w, axis=-1)
        return np.where(np.isfinite(out), out, -1e300)

    logl.vectorized = bool(batch)
    return logl


def torch_forward(emulator, prior, nuisance_fill=None):
    """Adapt a ``SpectrumEmulator`` to the numpy θ → C(λ) signature ``make_loglike`` wants.

    The emulator was trained on CLR composition + nuisances, while the sampler works in
    ILR, so this converts ILR → VMR → CLR on the way in. ``nuisance_fill`` supplies any
    nuisance the sampler is not varying.
    """
    H = helmert_basis(prior.n_ilr + 1, dtype=torch.float64)

    def forward(theta):
        th = np.atleast_2d(np.asarray(theta, dtype=np.float64))
        z = torch.as_tensor(th[:, :prior.n_ilr], dtype=torch.float64)
        vmr = from_ilr(z, H)
        ly = torch.log(vmr.clamp(min=1e-12))
        clr = (ly - ly.mean(-1, keepdim=True)).float()
        nz = th[:, prior.n_ilr:]
        if nuisance_fill is not None and nz.shape[1] == 0:
            nz = np.repeat(np.asarray(nuisance_fill)[None, :], th.shape[0], axis=0)
        x = torch.cat([clr, torch.as_tensor(nz, dtype=torch.float32)], dim=1)
        with torch.no_grad():
            out = emulator(x).numpy()
        return out[0] if np.ndim(theta) == 1 else out

    return forward


def run_nested(logl, prior, backend="nautilus", n_live=1000, seed=0, **kw):
    """Run one retrieval. Returns dict(samples [S, D], log_w [S], logz, ess, backend).

    Backends are imported lazily so a missing sampler only breaks the tier that asks
    for it, never the eval package as a whole.
    """
    if backend not in SUPPORTED:
        raise ValueError(f"backend must be one of {SUPPORTED}, got {backend!r}")

    if backend == "nautilus":
        from nautilus import Sampler
        s = Sampler(prior.transform, logl, n_dim=prior.ndim, n_live=n_live,
                    seed=seed, **kw)
        s.run(verbose=False)
        pts, logw, _ = s.posterior()
        return {"samples": np.asarray(pts), "log_w": np.asarray(logw),
                "logz": float(s.log_z), "ess": _ess(logw), "backend": backend}

    if backend == "dynesty":
        from dynesty import NestedSampler
        s = NestedSampler(logl, prior.transform, prior.ndim, nlive=n_live,
                          rstate=np.random.default_rng(seed), **kw)
        s.run_nested(print_progress=False)
        r = s.results
        return {"samples": np.asarray(r.samples), "log_w": np.asarray(r.logwt - r.logz[-1]),
                "logz": float(r.logz[-1]), "ess": _ess(r.logwt - r.logz[-1]),
                "backend": backend}

    import nestle
    r = nestle.sample(logl, prior.transform, prior.ndim, npoints=n_live,
                      rstate=np.random.default_rng(seed), **kw)
    logw = r.logwt - r.logz
    return {"samples": np.asarray(r.samples), "log_w": np.asarray(logw),
            "logz": float(r.logz), "ess": _ess(logw), "backend": backend}


def _ess(log_w):
    lw = np.asarray(log_w, dtype=np.float64)
    lw = lw - lw.max()
    w = np.exp(lw)
    w /= w.sum()
    return float(1.0 / np.sum(w ** 2))


def posterior_summary(result, prior, n_draw=200, seed=0):
    """Weighted nested-sampling output → the (samples_log, point_log) contract.

    Resamples to equal weight so the draws slot into ``metrics_extra`` unchanged, and
    returns the posterior mean in LOG10 space — the metric-matched point estimate,
    matching T0's ``post_mean_log``.
    """
    lw = result["log_w"] - result["log_w"].max()
    w = np.exp(lw)
    w /= w.sum()
    idx = np.random.default_rng(seed).choice(len(w), size=n_draw, replace=True, p=w)
    vmr = prior.to_vmr(result["samples"][idx])
    s_log = np.log10(np.clip(vmr, 1e-12, None))
    return {"samples_log": s_log, "point_log": s_log.mean(axis=0),
            "logz": result["logz"], "ess": result["ess"]}
