"""Build the PSG sanity-anchor cache (VALIDATION_PLAN B2) — fully offline.

The anchor routes REAL held-out INARA/PSG spectra through the BYTE-IDENTICAL assembly
path that build_cache_v2_prt.py uses for the pRT cross-gen cache (bare-star/planet split →
per-channel median-match to the cache_v2 test medians → star_planet = star + planet → real
INARA noise bootstrap → symlink the INARA-train derived stats). The ONLY thing removed vs
the pRT cache is the alternative generator: the spectra here ARE real PSG.

So if crosstest / Section-D covered-species R² on this anchor collapses relative to the
native INARA test R², the confound is the EVAL PATH (median-match / noise bootstrap /
symlink), not generator physics — and no cross-generator gap can be trusted until it's
fixed. Passing anchor ⇒ the pRT/TauREx collapse is a real generator effect.

Zero PSG API calls: the source is the already-cached real INARA test tensor.

Run in venvNLP (torch):
  python src/evaluation/crossgen/build_cache_v2_psg.py --cache-v2 data/cache_v2 --out data/cache_v2_psg --n 2000
"""
import os
import json
import argparse
import numpy as np
import torch

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
DEFAULT_CACHE_V2 = os.path.join(REPO, "data", "cache_v2")
DEFAULT_OUT = os.path.join(REPO, "data", "cache_v2_psg")
_SYMLINK_GLOBS = ("wavelength.pt", "norm_", "label_pipe_", "continuum_")


def _median(x):
    return float(x.median())


def build(cache_v2, out, n=2000, split="test", seed=0):
    X = torch.load(os.path.join(cache_v2, f"{split}_x.pt"), map_location="cpu").float()   # [M,2,L]
    Y = torch.load(os.path.join(cache_v2, f"{split}_y.pt"), map_location="cpu").float()   # [M,12]
    noise_all = torch.load(os.path.join(cache_v2, f"{split}_noise.pt"), map_location="cpu").float()
    M, C, L = X.shape
    assert C == 2, f"expected 2-channel cache_v2, got C={C}"

    rng = np.random.default_rng(seed)
    sel = rng.choice(M, size=min(n, M), replace=False)
    X, Y = X[sel], Y[sel]
    N = X.shape[0]

    # Real INARA channels: ch0 = star_planet, ch1 = planet. Recover the bare star exactly
    # as observable.stellar() does, to mirror the pRT 'both' cache (ch0=bare-star, ch1=planet).
    star_planet0, planet0 = X[:, 0, :], X[:, 1, :]
    bare_star = torch.clamp(star_planet0 - planet0, min=1e-30)

    # Target medians from the SAME real cache_v2 test tensor (identical to build_cache_v2_prt).
    m_star, m_planet = _median(star_planet0), _median(planet0)

    star = bare_star * (m_star / _median(bare_star))      # median-match (path step)
    planet = planet0 * (m_planet / _median(planet0))
    star_planet = star + planet
    test_x = torch.stack([star_planet, planet], dim=1).contiguous().float()

    # Bootstrap REAL INARA noise rows (identical path; leak-free SNR stays in-distribution).
    idx = rng.choice(noise_all.shape[0], size=N, replace=(N > noise_all.shape[0]))
    test_noise = noise_all[idx].contiguous()

    os.makedirs(out, exist_ok=True)
    torch.save(test_x, os.path.join(out, "test_x.pt"))
    torch.save(Y, os.path.join(out, "test_y.pt"))
    torch.save(test_noise, os.path.join(out, "test_noise.pt"))
    with open(os.path.join(out, "test_ids.json"), "w") as f:
        json.dump([f"psg_anchor_{i}" for i in range(N)], f)

    linked = []
    for fn in sorted(os.listdir(cache_v2)):
        if any(fn == g or fn.startswith(g) for g in _SYMLINK_GLOBS):
            dst = os.path.join(out, fn)
            if os.path.islink(dst) or os.path.exists(dst):
                os.remove(dst)
            os.symlink(os.path.join(cache_v2, fn), dst)
            linked.append(fn)

    manifest = {"engine": "psg_anchor", "observable": "reflected_light", "n": int(N),
                "channels": ["star_planet", "planet"], "seq_len": int(L),
                "source": f"real INARA {split} split through the pRT assembly path",
                "noise": f"bootstrapped real INARA {split}_noise (seed={seed})",
                "scale_match": {"star_planet_median": m_star, "planet_median": m_planet},
                "symlinked_from_cache_v2": linked}
    with open(os.path.join(out, "manifest_psg.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {out}  test_x {tuple(test_x.shape)}  test_y {tuple(Y.shape)}")
    print(f"  scale targets: star_planet median={m_star:.3e}  planet median={m_planet:.3e}")
    print(f"  symlinked: {linked}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-v2", default=DEFAULT_CACHE_V2)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build(a.cache_v2, a.out, n=a.n, split=a.split, seed=a.seed)
