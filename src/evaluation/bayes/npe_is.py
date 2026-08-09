"""T4 — importance-sampling correction of the NPE posterior (Dax+2023, arXiv:2312.08295).

THE IDEA
    T0 is exact but its proposal is the prior, which collapses (ESS→1) as soon as the
    likelihood is informative. T1 is fast and covers every α but is an approximation
    with no error bar on its own approximation. Composing them fixes both:

        θ_s ~ q(θ | x)                              [T1 flow — a GOOD proposal]
        w_s ∝ p(θ_s) · L(x | θ_s) / q(θ_s | x)      [T0 weighting — exactness]

    Self-normalized importance sampling is exact for ANY proposal with the right
    support, so the weighted samples are asymptotically the true posterior no matter how
    imperfect the flow is. The flow only has to be *close*, and closeness is measured
    rather than assumed: the ESS of these weights is a direct, per-planet certificate of
    how good the NPE posterior was.

WHAT THIS BUYS OVER EACH PARENT
    vs T0 — the proposal is now concentrated where the likelihood is, so ESS survives at
            high α. This is the only route in this repo to an exact posterior at the
            noiseless reference, where the headline R² lives and where T0 is dead.
    vs T1 — a correctness guarantee. A low ESS here is proof the flow's posterior is
            wrong for that planet, and the corrected samples are right anyway.

WHAT IT COSTS
    A likelihood, hence a forward model over arbitrary θ — the emulator. So T4 inherits
    the emulator's adequacy limit (``emulator.chi2_adequacy``) exactly as T2 does, and
    is only valid where χ²_emul ≪ 1. It also inherits ``nested.EmpiricalPrior``'s
    Gaussian approximation to p(θ). Both are stated in the returned diagnostics; neither
    is optional to check.
"""
from __future__ import annotations

import numpy as np
import torch

from common.transforms import from_ilr, helmert_basis
from evaluation.bayes.likelihood import ESS_MIN


def prior_logpdf(theta, prior):
    """log p(θ) under ``nested.EmpiricalPrior`` (multivariate normal)."""
    d = prior.ndim
    diff = np.atleast_2d(np.asarray(theta, dtype=np.float64)) - prior.mean
    sol = np.linalg.solve(prior.chol, diff.T).T           # whiten
    logdet = 2.0 * np.sum(np.log(np.diag(prior.chol)))
    return -0.5 * (np.sum(sol * sol, axis=1) + logdet + d * np.log(2 * np.pi))


def correct(model, cfg, x_encoded, log_alpha, C_obs, sigma, prior, forward,
            n_proposal=2000, device="cpu", seed=0):
    """Importance-correct one planet's NPE posterior.

    x_encoded : [1, C, L] the encoded observable the flow conditions on
    C_obs, sigma : [L] contrast and its 1σ (σ already divided by α)
    forward   : θ → C(λ), from ``nested.torch_forward(emulator, prior)``

    Returns dict(samples_log [S,12], point_log [12], ess, ess_frac, log_evidence).
    """
    H = helmert_basis(prior.n_ilr + 1, dtype=torch.float64)
    with torch.no_grad():
        ctx = model.embed(x_encoded.to(device),
                          torch.full((1,), float(log_alpha), device=device))
        dist = model.flow(ctx)
        u = dist.sample((n_proposal,))                    # [S, 1, 11] standardized
        log_q = dist.log_prob(u).squeeze(-1).cpu().numpy()
        z = model.unstandardize(u).squeeze(1).double().cpu()   # [S, 11] ILR
    # The flow's density is over STANDARDIZED ILR; the prior's is over raw ILR. The
    # Jacobian of that affine map is constant in θ, so it cancels in the normalized
    # weights — but only because it is constant. Do not add a non-affine reparam here
    # without putting its log-determinant back.
    theta = z.numpy()
    if prior.ndim > prior.n_ilr:                          # pad any fixed nuisances
        theta = np.concatenate(
            [theta, np.repeat(prior.mean[None, prior.n_ilr:], theta.shape[0], axis=0)], axis=1)

    m = np.asarray(forward(theta), dtype=np.float64)
    w_l = 1.0 / np.clip(np.asarray(sigma, dtype=np.float64), 1e-30, None) ** 2
    d = np.asarray(C_obs, dtype=np.float64)[None, :] - m
    log_L = -0.5 * np.sum(d * d * w_l[None, :], axis=1)

    log_w = log_L + prior_logpdf(theta, prior) - log_q
    log_w = np.where(np.isfinite(log_w), log_w, -np.inf)
    mx = log_w.max()
    if not np.isfinite(mx):
        raise FloatingPointError("all importance weights underflowed — proposal and "
                                 "likelihood have disjoint support")
    w = np.exp(log_w - mx)
    log_evidence = float(mx + np.log(w.mean()))
    w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2))

    idx = np.random.default_rng(seed).choice(len(w), size=len(w), replace=True, p=w)
    vmr = from_ilr(torch.as_tensor(theta[idx, :prior.n_ilr], dtype=torch.float64), H).numpy()
    s_log = np.log10(np.clip(vmr, 1e-12, None))
    return {
        "samples_log": s_log,
        "point_log": s_log.mean(axis=0),
        "ess": ess,
        "ess_frac": ess / n_proposal,
        "log_evidence": log_evidence,
        # ESS here certifies the FLOW, unlike T0's ESS which certifies the prior as a
        # proposal. A low value means this planet's NPE posterior was poor — the
        # corrected samples remain valid, just noisier.
        "flow_ok": bool(ess >= ESS_MIN),
    }
