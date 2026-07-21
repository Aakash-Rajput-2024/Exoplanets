"""Evaluate every retrieval track on the 19 real Sagan-catalog bodies, +/- declouding.

    PYTHONPATH=src:. python3 sagan_eval/run_sagan.py            # all tracks
    PYTHONPATH=src:. python3 sagan_eval/run_sagan.py --decloud  # + declouded pass

Three probes, one per epistemic class (see truth.py):

  ACCURACY (terrestrial)   dominant-gas correct + mean dex error on covered species.
                           Only Earth and Titan are sampled densely enough to support a
                           quantitative claim (bluegap.py: R~22 costs -0.7 to -1.1 R2).

  HONESTY (giant)          H2/He mass sits outside the simplex; must not fabricate O2/O3.

  FALSE POSITIVE (airless) The headline. These bodies have no atmosphere, so ANY confident
                           gas call is fabricated by construction. Earth (Lundock 081121)
                           and the Moon (Lundock 081121, same night, same instrument, same
                           telluric division, R=867 both) form a matched pair: whatever the
                           model says about Earth's O2 it must NOT say about a rock.

DECLOUDING: Venus (H2SO4 deck), Titan (organic haze), Jupiter/Saturn (NH3) and Earth
(~60% water cloud) are cloud/haze dominated. cloud_recovery.DecloudUNet1D restores a
clear-sky contrast channel, which we feed to the SAME frozen retrieval weights. The
declouder trained on GREY clouds over INARA spectra, so real H2SO4/haze decks are far
out of its training family -- that is the point of testing it here.

Everything is inference. Checkpoints are opened read-only; nothing is written outside
sagan_eval/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(REPO, "src"))

from evaluation import core                                    # noqa: E402
from evaluation import metrics_extra as mx                     # noqa: E402
from common.data import TARGET_COLUMNS                         # noqa: E402
from sagan_eval import build_obs, ingest, truth                # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
DECLOUD_CKPT = os.path.join(REPO, "src", "cloud_recovery", "checkpoints", "decloud_seed0.pth")

TRACKS = [
    ("original1dcnn",  [0],       ""),
    ("optimized1dcnn", [0, 1, 2], ""),
    ("transformerarch", [0],      ""),
    ("causal",         [0],       ""),
    ("causal_cfi",     [0, 1],    "_cfi"),
    ("causal_xl",      [0],       "_cfi"),
]


# --------------------------------------------------------------------------- #
# Declouding front-end
# --------------------------------------------------------------------------- #
def load_declouder(device, ckpt=DECLOUD_CKPT):
    from cloud_recovery.model import DecloudUNet1D
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    c = ck.get("config", {})
    m = DecloudUNet1D(in_channels=ck["in_channels"], sequence_length=ck["seq_len"],
                      base=c.get("base", 32), depth=c.get("depth", 3)).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


@torch.no_grad()
def decloud_encoded(declouder, x_enc, device):
    """[N,2,L] encoded observable -> same, with ch0 replaced by the restored contrast."""
    x = torch.as_tensor(x_enc, dtype=torch.float32)
    out = declouder(x.to(device)).cpu()          # [N,1,L]
    return torch.cat([out, x[:, 1:]], dim=1)


# --------------------------------------------------------------------------- #
def predict_bodies(ctx, raw_x, noise, declouder=None, noiseless=True, alpha=1.0):
    """Ensemble prediction [N,12] for every body, optionally through the declouder."""
    if declouder is None:
        out = core.predict_raw(ctx, raw_x, noise, noiseless=noiseless, alpha=alpha)
        return out["ens"]
    _, _, lt = ctx.ckpt_config()
    lp = core.inara_label_pipeline(ctx.cache_v2, lt)
    x_enc = core.encode_raw(ctx, raw_x, noise, noiseless=noiseless, alpha=alpha)
    x_dec = decloud_encoded(declouder, x_enc, ctx.device)
    _, ens = core._decode_predict(ctx, x_dec, lp)
    return ens


def summarize(name, pred, tgt):
    row = {"body": name, "body_class": tgt["body_class"], "native_R": tgt["native_R"],
           "quantitative": tgt["quantitative"], "cloudy": tgt["cloudy"],
           "dominant_pred": truth.dominant(pred), "o2_plus_o3": truth.o2_o3_sum(pred),
           "max_vmr": float(np.max(pred)),
           "pred": {c: float(pred[i]) for i, c in enumerate(TARGET_COLUMNS)}}
    if tgt["true_vector"] is not None and tgt["representable"]:
        s = mx.known_truth_summary(pred, tgt["true_vector"], TARGET_COLUMNS, tgt["covered_species"])
        row.update(dominant_true=s["dominant_true"], dominant_correct=s["dominant_correct"],
                   mean_dex_error_covered=s["mean_dex_error_covered"],
                   spearman=s["spearman_ordering"])
    elif tgt["true_vector"] is not None:
        row["dominant_true"] = "H2/He (outside simplex)"
    else:
        row["dominant_true"] = "none (airless)"
    return row


def run_track(track, seeds, suffix, raw_x, noise, names, targets, declouder=None):
    ctx = core.EvalContext(track=track, seeds=seeds, suffix=suffix)
    if not ctx.has_checkpoints:
        return None
    res = {"track": ctx.label, "seeds": ctx.seeds}
    for tag, dc in [("clear", None)] + ([("declouded", declouder)] if declouder else []):
        ens = predict_bodies(ctx, raw_x, noise, declouder=dc)
        res[tag] = [summarize(n, ens[i], targets[n]) for i, n in enumerate(names)]
    core.free_device(ctx.device)
    return res


# --------------------------------------------------------------------------- #
def print_track(res, tag="clear"):
    rows = res[tag]
    print(f"\n{'='*100}\nTRACK {res['track']}  seeds={res['seeds']}   [{tag}]\n{'='*100}")
    for cls, title in [("terrestrial", "ACCURACY — terrestrial atmospheres"),
                       ("giant", "HONESTY — H2/He giants (outside the simplex)"),
                       ("airless", "FALSE POSITIVE — airless bodies (no atmosphere at all)")]:
        sub = [r for r in rows if r["body_class"] == cls]
        print(f"\n  {title}")
        if cls == "terrestrial":
            h = f"    {'body':<9}{'R':>6}{'quant':>7}{'true':>7}{'pred':>7}{'ok':>4}{'dex_err':>9}{'O2+O3':>8}"
        else:
            h = f"    {'body':<9}{'R':>6}{'quant':>7}{'pred_dominant':>15}{'max_vmr':>9}{'O2+O3':>8}"
        print(h); print("    " + "-" * (len(h) - 4))
        for r in sub:
            q = "y" if r["quantitative"] else "NO"
            if cls == "terrestrial":
                de = r.get("mean_dex_error_covered")
                de_s = f"{de:>9.2f}" if de is not None else f"{'-':>9}"
                ok = "Y" if r.get("dominant_correct") else "n"
                print(f"    {r['body']:<9}{r['native_R']:>6.0f}{q:>7}"
                      f"{r.get('dominant_true', '-'):>7}{r['dominant_pred']:>7}{ok:>4}"
                      f"{de_s}{r['o2_plus_o3']:>8.3f}")
            else:
                print(f"    {r['body']:<9}{r['native_R']:>6.0f}{q:>7}"
                      f"{r['dominant_pred']:>15}{r['max_vmr']:>9.3f}{r['o2_plus_o3']:>8.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decloud", action="store_true", help="also run the declouded pass")
    ap.add_argument("--tracks", nargs="*", default=None)
    a = ap.parse_args()

    raw_x, noise, names, metas = build_obs.all_bodies_raw()
    targets = {t["name"]: t for t in truth.all_targets(metas)}
    print(f"bodies={len(names)}  raw_x={tuple(raw_x.shape)}")

    declouder = None
    if a.decloud:
        dev = core.get_device()
        declouder, ck = load_declouder(dev)
        n = sum(p.numel() for p in declouder.parameters())
        print(f"declouder: {n:,} params  from {os.path.relpath(DECLOUD_CKPT, REPO)}")

    todo = [t for t in TRACKS if (a.tracks is None or t[0] in a.tracks)]
    all_res = {}
    for track, seeds, suffix in todo:
        try:
            res = run_track(track, seeds, suffix, raw_x, noise, names, targets, declouder)
        except Exception as e:                       # a bad checkpoint must not kill the sweep
            print(f"\n!! {track}{suffix}: {type(e).__name__}: {e}")
            continue
        if res is None:
            print(f"\n-- {track}{suffix}: no checkpoints, skipped")
            continue
        all_res[res["track"]] = res
        for tag in (["clear", "declouded"] if a.decloud else ["clear"]):
            print_track(res, tag)

    path = os.path.join(OUT, "sagan_results.json")
    with open(path, "w") as f:
        json.dump({"bodies": names, "metas": metas, "results": all_res}, f, indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
