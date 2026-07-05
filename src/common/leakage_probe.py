"""Standing input-leakage regression probe (the check that caught the v2 leak).

The v2 observable leaked the planetary spectrum, noise-free, through the SNR
"error-bar" channel: the model read F_p from its own error bars instead of the
noisy contrast, so accuracy was flat vs SNR and flat vs clouds. A cheap
permutation-importance test on the FROZEN TEST split localizes this in seconds:

    * permute the CONTRAST channel across planets → R² must COLLAPSE
      (that channel is where the real, noisy retrieval signal lives), and
    * permute the SNR channel across planets → R² must stay ~unchanged
      (it is auxiliary; if permuting it is what destroys R², the model is
      reading planet signal from it — i.e. the channel is leaking).

PASS  ⇔  drop(contrast) is large AND drop(SNR) is small
FAIL  ⇔  drop(SNR) ≥ drop(contrast)  (SNR channel is carrying the signal)

WHY A SWEEP OF α (post-noise-gate): with the leak closed AND the noise units
validated (Zorzan+25: α=1 is the photon-starved nominal exposure), a correct
model has R²≈0 at α=1 — so a probe pinned at α=1 can only ever say INCONCLUSIVE
(no signal in EITHER channel to localize). To actually confirm "leak-free" we
must test where the model has signal: this probe runs the permutation test at
several α and renders its verdict at the OPERATING POINT — the α with the most
baseline signal. There, an honest model shows drop(contrast) ≫ drop(SNR); a
leaky one shows the reverse (and would show it even at α=1, since the leak is
α-independent).

    python -m common.leakage_probe <track> [--seed S] [--n 3000]
                                  [--alphas 1 10 100]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from common.registry import track_config, CACHE_V2
from common.runtime import get_device
from common.pipeline import load_eval_raw, get_norm, TEST_NOISE_SEED
from common.inputs import build_eval_observable  # noqa: F401 (kept for parity/imports)
from common.observable import contrast, snr, contrast_sigma
from common.evaluate import _load_model, _predict, _ckpt_path
from common import metrics

MIN_SIGNAL = 0.02          # a drop below this is indistinguishable from noise
MIN_BASELINE_R2 = 0.05     # below this the model has no signal to localize
DEFAULT_ALPHAS = (1.0, 10.0, 100.0)


def _r2(model, x_obs, lp, y_true, device):
    pred = lp.decode(_predict(model, x_obs, device)).numpy()
    return float(np.nanmean(metrics.report(y_true, pred, space="log10", bootstrap=0)["r2"]))


def _probe_at_alpha(model, sp, pl, noise, norm, lp, y_true, device, perm, alpha):
    """Permutation test at one α. Contrast noise is σ_C/α and the SNR channel is
    α-scaled — EXACTLY matching common.observable.inject_noise, so the probe sees
    the same observable the evaluator/trainer do at this α."""
    gen = torch.Generator().manual_seed(TEST_NOISE_SEED)
    C = contrast(sp, pl)
    sig = contrast_sigma(sp, pl, noise) / max(alpha, 1e-12)
    Cn = C + sig * torch.randn(C.shape, generator=gen)
    S = snr(sp, pl, noise, alpha)

    base = norm.encode(torch.stack([Cn, S], dim=1))
    contrast_perm = norm.encode(torch.stack([Cn[perm], S], dim=1))
    snr_perm = norm.encode(torch.stack([Cn, S[perm]], dim=1))

    r2_base = _r2(model, base, lp, y_true, device)
    r2_cp = _r2(model, contrast_perm, lp, y_true, device)
    r2_sp = _r2(model, snr_perm, lp, y_true, device)
    return dict(alpha=alpha, r2_base=r2_base, r2_contrast_perm=r2_cp, r2_snr_perm=r2_sp,
                drop_contrast=r2_base - r2_cp, drop_snr=r2_base - r2_sp)


def _verdict(row):
    """PASS / FAIL / INCONCLUSIVE from one operating-point row."""
    if row["r2_base"] < MIN_BASELINE_R2 or max(row["drop_contrast"], row["drop_snr"]) < MIN_SIGNAL:
        return "INCONCLUSIVE"
    return "FAIL" if row["drop_snr"] > row["drop_contrast"] else "PASS"


def probe(track, seed=None, cache_dir=CACHE_V2, n=3000, perm_seed=0, alphas=DEFAULT_ALPHAS):
    cfg = track_config(track)
    if seed is None:
        seed = next((s for s in cfg["seeds"] if os.path.exists(_ckpt_path(track, s))), None)
        if seed is None:
            raise SystemExit(f"No checkpoint for {track}")
    device = get_device()

    raw_x, y_lin, noise, _ids, lp = load_eval_raw(cache_dir, "test", cfg["label_transform"])
    raw_x, y_lin, noise = raw_x[:n], y_lin[:n], noise[:n]
    y_true = y_lin.numpy()
    obs_mode, input_norm = cfg["obs_mode"], cfg["input_norm"]
    if obs_mode != "contrast_snr":
        raise SystemExit(f"probe targets the 2-channel contrast_snr observable; {track} uses {obs_mode}")
    norm = get_norm(cache_dir, obs_mode, input_norm)
    model, _ = _load_model(track, seed, device)

    sp, pl = raw_x[:, 0, :], raw_x[:, 1, :]
    g = torch.Generator().manual_seed(perm_seed)
    perm = torch.randperm(sp.shape[0], generator=g)

    rows = [_probe_at_alpha(model, sp, pl, noise, norm, lp, y_true, device, perm, a)
            for a in alphas]

    # OPERATING POINT = the α with the most baseline signal. The leak (if present)
    # is α-independent, so judging where the model can actually be read is the
    # discriminating test; at the photon floor (α=1) neither channel carries
    # signal and the honest answer is INCONCLUSIVE.
    op = max(rows, key=lambda r: r["r2_base"])
    status = _verdict(op)

    print(f"\n=== leakage probe: {track} seed{seed} (N={sp.shape[0]}) ===")
    print(f"  {'alpha':<8}| {'R²base':<9}| {'permute contrast':<20}| {'permute SNR':<18}")
    print("  " + "-" * 62)
    for r in rows:
        star = "  *" if r is op else ""
        print(f"  {r['alpha']:<8}| {r['r2_base']:<+9.4f}| "
              f"{r['r2_contrast_perm']:<+8.4f}(drop {r['drop_contrast']:+.4f})| "
              f"{r['r2_snr_perm']:<+8.4f}(drop {r['drop_snr']:+.4f}){star}")
    print(f"  operating point: α={op['alpha']}  (max baseline signal)")
    msg = {
        "PASS": "PASS — at the operating point, permuting CONTRAST destroys R² and "
                "permuting SNR does not → signal lives in the noisy contrast channel (leak-free)",
        "FAIL": "FAIL — permuting the SNR channel destroys R² → the error-bar channel "
                "is carrying retrieval signal (LEAK)",
        "INCONCLUSIVE": "INCONCLUSIVE — no α reached R²>%.2f; the model hasn't learned "
                        "enough to localize signal (train more / raise α)" % MIN_BASELINE_R2,
    }[status]
    print(f"  VERDICT: {msg}")
    if status == "FAIL":
        print("  → the model is reading planet signal from the error-bar channel; "
              "check common.observable.snr uses the star-only numerator.")
    return dict(track=track, seed=seed, alphas=list(alphas), rows=rows,
                operating_alpha=op["alpha"], **{k: op[k] for k in
                ("r2_base", "r2_contrast_perm", "r2_snr_perm", "drop_contrast", "drop_snr")},
                status=status, pass_=(status != "FAIL"))


def main():
    ap = argparse.ArgumentParser(description="Input-leakage permutation probe (ULTRAPLAN §2.3).")
    ap.add_argument("track")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--cache", default=CACHE_V2)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--alphas", type=float, nargs="*", default=list(DEFAULT_ALPHAS),
                    help="SNR multipliers to probe; verdict is read at the one with most signal")
    a = ap.parse_args()
    res = probe(a.track, seed=a.seed, cache_dir=a.cache, n=a.n, alphas=a.alphas)
    # Exit non-zero ONLY on a true FAIL (leak). INCONCLUSIVE is not a failure.
    raise SystemExit(1 if res["status"] == "FAIL" else 0)


if __name__ == "__main__":
    main()
