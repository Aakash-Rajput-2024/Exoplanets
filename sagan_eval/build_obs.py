"""Turn a catalog albedo Ag(lambda) into the model's (raw_x, noise) observable.

Reflected light at full phase:  Fp(lam) = Ag(lam) * Fsun(lam) * (Rp/a)^2
                                Fstar(lam) = Fsun(lam)

evaluation/synth.build_raw median-matches the planet and star channels SEPARATELY onto
the real INARA scale, so (Rp/a)^2 -- and the absolute albedo normalisation, which the
catalog's own sources define inconsistently (Venus/Mercury look low vs the textbook
geometric albedo) -- are washed out by construction. Only the SPECTRAL SHAPE survives.
That is the same contract Section E already runs under, and it is what makes bodies with
wildly different brightness comparable.

Consequence worth stating: the contrast channel is
    C = Fp/(Fstar+Fp) = k_p*Ag*Fsun / (k_s*Fsun + k_p*Ag*Fsun) = k_p*Ag / (k_s + k_p*Ag)
so Fsun CANCELS EXACTLY in ch0. The solar SED only sets the stellar-SNR channel (ch1).
``verify_sed_cancellation()`` checks this numerically rather than trusting the algebra.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))

from evaluation import synth, core                      # noqa: E402
from sagan_eval import ingest                           # noqa: E402

CACHE_V2 = os.path.join(REPO, "data", "cache_v2")


def body_raw(ag_grid, sed=None, cache_v2=CACHE_V2, seed=0):
    """(raw_x[1,2,L], noise[1,L]) at the real INARA scale for one body."""
    sed = ingest.solar_sed_on_grid() if sed is None else sed
    planet = ag_grid * sed
    star = sed
    return synth.build_raw(planet[None], star[None], cache_v2, seed=seed)


def all_bodies_raw(bodies=None, cache_v2=CACHE_V2, seed=0):
    """Stack every body into one (raw_x[N,2,L], noise[N,L]) batch + name list + metas."""
    sed = ingest.solar_sed_on_grid()
    built = ingest.build_all(bodies)
    names = list(built)
    planets = np.stack([built[b][0] * sed for b in names])
    stars = np.stack([sed for _ in names])
    raw_x, noise = synth.build_raw(planets, stars, cache_v2, seed=seed)
    return raw_x, noise, names, {b: built[b][1] for b in names}


def verify_sed_cancellation(body="Earth", tol=1e-6):
    """The contrast channel's SHAPE must not depend on the assumed stellar SED.

    Fsun cancels per-lambda, but build_raw's median-matching sets
    k_p/k_s = (m_planet/m_star) * median(Fsun)/median(Ag*Fsun), which is weakly
    SED-dependent. So a wrong SED rescales C by a constant; it cannot deform it.
    We assert shape invariance (Pearson == 1) and merely REPORT the amplitude.
    """
    from common.observable import make_observable
    grid = ingest.inara_grid()
    ag, _ = ingest.albedo_on_grid(ingest.CANONICAL[body], grid)

    seds = {"catalog": ingest.solar_sed_on_grid(grid),
            "planck": ingest._planck_lambda(ingest.SUN_TEFF, grid),
            "flat": np.ones_like(grid)}
    out = {t: make_observable(*body_raw(ag, sed=s), "contrast_snr",
                              alpha=300.0)[0, 0].numpy() for t, s in seds.items()}

    ref = out["catalog"]
    shape = {t: float(np.corrcoef(v, ref)[0, 1]) for t, v in out.items()}
    amp = {t: float(np.median(v) / np.median(ref)) for t, v in out.items()}
    lin = float(np.corrcoef(ref, ag[:len(ref)])[0, 1])
    ok = all(1.0 - v < tol for v in shape.values())
    return ok, shape, amp, lin


if __name__ == "__main__":
    ok, shape, amp, lin = verify_sed_cancellation()
    print("contrast-channel SED independence:")
    for k in shape:
        print(f"   vs {k:<8}  shape pearson={shape[k]:.8f}   amplitude x{amp[k]:.4f}")
    print(f"   C is linear in Ag: pearson(C, Ag) = {lin:.8f}")
    print(f"   => {'PASS' if ok else 'FAIL'}: ch0 shape depends only on Ag(lambda)\n")

    raw_x, noise, names, metas = all_bodies_raw()
    print(f"raw_x {tuple(raw_x.shape)}  noise {tuple(noise.shape)}  bodies={len(names)}")
    print(f"finite: raw_x {bool(torch.isfinite(raw_x).all())}  noise {bool(torch.isfinite(noise).all())}")
    sp, pl = raw_x[:, 0], raw_x[:, 1]
    print(f"star+planet median {sp.median():.3e}   planet median {pl.median():.3e}")
    print(f"planet < star+planet everywhere: {bool((pl <= sp + 1e-30).all())}")
