"""Assemble a v2-format cross-generator eval cache for petitRADTRANS.

The v2 models are 2-channel (contrast + stellar-SNR) with a per-λ asinh input norm
and a CLR label pipeline — none of which the v1 `crosstest.py` path produces.
Instead of re-implementing the v2 eval, we build a cache directory that the
*existing* v2 evaluator (`common.evaluate` / `common.pipeline`) consumes directly:

  data/cache_v2_prt/
    test_x.pt      [N,2,L]  pRT (star_planet, planet), rescaled to INARA raw scale
    test_y.pt      [N,12]   real INARA 12-vec labels (bootstrapped by the sampler)
    test_noise.pt  [N,L]    REAL INARA LUVOIR 1σ noise rows (bootstrapped). The
                            v2 SNR channel = α·F_star/noise is leak-free (planet-
                            independent), so using real INARA noise keeps it
                            in-distribution and isolates the measured domain gap
                            to the CONTRAST channel — the actual generator physics.
    test_ids.json  [N]      placeholder ids (load_raw requires the file to exist)
    <symlinks to data/cache_v2>: wavelength.pt + the INARA-fit derived caches
      (norm_*_v2trim1.pt, label_pipe_*_v2trim1.pt, continuum_*_v2trim1.pt). Because
      get_norm()/label_pipeline() short-circuit on these cached files, the norm and
      label pipeline are the INARA-*train* statistics with NO refit on synthetic
      data — we score the trained weights fairly.

Scale alignment: each pRT channel is median-matched (one global scalar) to the
corresponding cache_v2 raw channel median. This washes out arbitrary absolute-
scale conventions (albedo, radius units) while preserving the pRT spectral SHAPE
— the real cross-generator domain shift. ch0 is then set to (star+planet) to match
the v2 `star_planet` channel semantics exactly (observable.stellar() reconstructs
F_star = star_planet − planet).

Run in venvNLP (torch). Requires the pRT `both` cache from build_eval_cache.py.
"""
import os
import json
import argparse
import numpy as np
import torch

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
DEFAULT_PRT_BOTH = os.path.join(REPO, "data", "cache_crossgen_prt_both")
DEFAULT_CACHE_V2 = os.path.join(REPO, "data", "cache_v2")
DEFAULT_OUT = os.path.join(REPO, "data", "cache_v2_prt")

# Derived-cache / grid files to symlink so the v2 evaluator reuses INARA-train
# statistics verbatim (no refit on synthetic spectra).
_SYMLINK_GLOBS = ("wavelength.pt", "norm_", "label_pipe_", "continuum_")


def _channel_median(x):  # x: [N, L] torch tensor
    return float(x.median())


def build(prt_both, cache_v2, out, n=None, noise_split="test", seed=0):
    # --- pRT synthetic (both = [star_planet_shape, planet_shape], INARA-calibrated) ---
    X = np.load(os.path.join(prt_both, "val_x.npy"))   # [N,2,L]  ch0=bare-star, ch1=planet
    Y = np.load(os.path.join(prt_both, "val_y.npy"))   # [N,12]   real INARA 12-vec labels
    X = torch.from_numpy(X).float()
    Y = torch.from_numpy(Y).float()
    if n is not None:
        X, Y = X[:n], Y[:n]
    N, C, L = X.shape
    assert C == 2, f"expected 2-channel pRT 'both' cache, got C={C}"

    # --- target scale from the real v2 cache (what the INARA-fit norm expects) ---
    v2_test_x = torch.load(os.path.join(cache_v2, "test_x.pt"), map_location="cpu")  # [M,2,L]
    assert v2_test_x.shape[2] == L, f"grid length mismatch: pRT L={L}, cache_v2 L={v2_test_x.shape[2]}"
    m_star = _channel_median(v2_test_x[:, 0, :])     # star_planet (F_star+F_p) median
    m_planet = _channel_median(v2_test_x[:, 1, :])   # planet (F_p) median
    del v2_test_x

    # --- median-match each pRT channel (global scalar; preserves spectral shape) ---
    star = X[:, 0, :] * (m_star / _channel_median(X[:, 0, :]))     # -> INARA star scale
    planet = X[:, 1, :] * (m_planet / _channel_median(X[:, 1, :]))  # -> INARA planet scale
    star_planet = star + planet                                    # F_star + F_p semantics
    test_x = torch.stack([star_planet, planet], dim=1).contiguous().float()  # [N,2,L]

    # --- bootstrap REAL INARA noise rows (leak-free SNR channel stays in-distribution) ---
    v2_noise = torch.load(os.path.join(cache_v2, f"{noise_split}_noise.pt"), map_location="cpu")  # [M,L]
    M = v2_noise.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.choice(M, size=N, replace=(N > M))
    test_noise = v2_noise[idx].contiguous().float()                # [N,L]
    del v2_noise

    # --- write real files ---
    os.makedirs(out, exist_ok=True)
    torch.save(test_x, os.path.join(out, "test_x.pt"))
    torch.save(Y, os.path.join(out, "test_y.pt"))
    torch.save(test_noise, os.path.join(out, "test_noise.pt"))
    with open(os.path.join(out, "test_ids.json"), "w") as f:
        json.dump([f"prt_{i}" for i in range(N)], f)

    # --- symlink INARA derived caches + grid (so norm/labels are INARA-train fit) ---
    linked = []
    for fn in sorted(os.listdir(cache_v2)):
        if any(fn == g or fn.startswith(g) for g in _SYMLINK_GLOBS):
            src = os.path.join(cache_v2, fn)
            dst = os.path.join(out, fn)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(src, dst)
            linked.append(fn)

    manifest = {
        "engine": "prt", "observable": "reflected_light", "n": int(N),
        "channels": ["star_planet", "planet"], "seq_len": int(L),
        "noise": f"bootstrapped real INARA {noise_split}_noise ({M} rows, seed={seed})",
        "scale_match": {"star_planet_median": m_star, "planet_median": m_planet},
        "label_source": "real INARA 12-vec (sampler bootstrap)",
        "symlinked_from_cache_v2": linked,
    }
    with open(os.path.join(out, "manifest_prt.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    C_med = float((test_x[:, 1, :] / torch.clamp(test_x[:, 0, :], min=1e-30)).median())
    print(f"wrote {out}")
    print(f"  test_x {tuple(test_x.shape)}  test_y {tuple(Y.shape)}  test_noise {tuple(test_noise.shape)}")
    print(f"  scale targets: star_planet median={m_star:.3e}  planet median={m_planet:.3e}")
    print(f"  median contrast C=F_p/F_starplanet = {C_med:.3e}  (INARA ref ~{m_planet/m_star:.3e})")
    print(f"  symlinked: {linked}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prt-both", default=DEFAULT_PRT_BOTH)
    ap.add_argument("--cache-v2", default=DEFAULT_CACHE_V2)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=None, help="cap number of pRT test planets")
    ap.add_argument("--noise-split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.prt_both, a.cache_v2, a.out, n=a.n, noise_split=a.noise_split, seed=a.seed)
