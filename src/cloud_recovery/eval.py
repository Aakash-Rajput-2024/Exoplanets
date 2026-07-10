"""Evaluate the declouder — reconstruction + FROZEN-retrieval R² recovery.

    PYTHONPATH=src python -m decloud.eval --track transformerarch --seed 0 \
        --decloud-ckpt src/cloud_recovery/checkpoints/decloud_seed0.pth

The honest test is NOT reconstruction error — it is whether de-clouding restores a
*frozen, clear-trained* retrieval model's accuracy. For each cloud family × exposure
α we score the SAME frozen checkpoint on three inputs:

    clear      → baseline R² (high)
    cloudy     → R² drops (the cloud–abundance degeneracy)
    declouded  → R² recovers most of the gap  ← the win

Recovery = (R²_declouded − R²_cloudy) / (R²_clear − R²_cloudy). The declouder trained
on GREY clouds only, so the held-out families (non_grey, band_selective, patchy)
measure generalisation to unseen cloud physics.

Everything is inference: the retrieval model is loaded read-only and never updated,
so this stays fully isolated. Outputs → ``src/cloud_recovery/eval_out/``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.registry import CACHE_V2, track_config, load_model_class
from common.runtime import get_device
from common.pipeline import load_eval_raw, get_norm, TEST_NOISE_SEED
from common.inputs import build_eval_observable
from common.observable import make_observable
from common.data import load_wavelength
from common import cloud_families, metrics
from decloud.model import DecloudUNet1D
from decloud.cloud_pairs import apply_family_to_raw, FAMILIES

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(THIS_DIR, "eval_out")


# ---- frozen retrieval model (loaded read-only; mirrors common.evaluate) ----------
def _retrieval_ckpt_path(track, seed, suffix=""):
    track_dir = os.path.dirname(track_config(track)["model_py"])
    named = os.path.join(track_dir, "checkpoints_v2", f"model_best_{track}_seed{seed}{suffix}.pth")
    if os.path.exists(named):
        return named
    return os.path.join(track_dir, "checkpoints_v2", f"model_best_seed{seed}{suffix}.pth")


def load_retrieval_model(track, seed, device, suffix=""):
    cfg = track_config(track)
    path = _retrieval_ckpt_path(track, seed, suffix)
    if not os.path.exists(path):
        return None, None
    ck = torch.load(path, map_location=device, weights_only=False)
    ModelClass = load_model_class(cfg["model_py"], cfg["model_cls"])
    model = ModelClass(in_channels=ck["in_channels"], sequence_length=ck["seq_len"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck.get("config", {})


def load_decloud_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ck.get("config", {})
    model = DecloudUNet1D(in_channels=ck["in_channels"], sequence_length=ck["seq_len"],
                          base=c.get("base", 32), depth=c.get("depth", 3)).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def _forward(model, x, device, batch=512):
    out = [model(x[i:i + batch].to(device)).cpu() for i in range(0, x.shape[0], batch)]
    return torch.cat(out)


def _retrieval_r2(model, x_enc, y_true, lp, device):
    pred_lin = lp.decode(_forward(model, x_enc, device)).numpy()
    return float(np.nanmean(metrics.report(y_true, pred_lin, space="log10", bootstrap=0)["r2"]))


def evaluate(decloud_ckpt, track, seed, families, alphas, n=4000, suffix="",
             cache_dir=CACHE_V2, plot_examples=0):
    device = get_device()
    os.makedirs(OUT_DIR, exist_ok=True)
    norm = get_norm(cache_dir)
    obs_mode = "contrast_snr"

    raw_x, y_lin, noise, _ids, lp = load_eval_raw(cache_dir, "test", "clr")
    if n and raw_x.shape[0] > n:
        raw_x, y_lin, noise = raw_x[:n], y_lin[:n], noise[:n]
    y_true = y_lin.numpy()
    wl = load_wavelength(cache_dir).numpy()
    cont = cloud_families.estimate_continuum(raw_x[:, 1, :].numpy())     # once, reused per family

    dmodel, dck = load_decloud_model(decloud_ckpt, device)
    rmodel, rcfg = load_retrieval_model(track, seed, device, suffix)
    has_retrieval = rmodel is not None
    if not has_retrieval:
        print(f"[note] no retrieval checkpoint for {track} seed {seed} "
              f"(looked in checkpoints_v2/) — reporting RECONSTRUCTION only. "
              f"Point --track/--seed at a trained checkpoint for the R² recovery test.")

    # Noiseless clear contrast = the reconstruction target (α-independent).
    clear_contrast = norm.encode(make_observable(raw_x, noise, obs_mode))[:, 0:1, :]

    rows = []
    for alpha in alphas:
        x_clear_enc = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=alpha, seed=TEST_NOISE_SEED)
        r2_clear = _retrieval_r2(rmodel, x_clear_enc, y_true, lp, device) if has_retrieval else None
        for family in families:
            x_cloud_raw = apply_family_to_raw(raw_x, family, wl, continuum=cont)
            x_cloud_enc = build_eval_observable(x_cloud_raw, noise, obs_mode, norm,
                                                alpha=alpha, seed=TEST_NOISE_SEED)
            declouded = _forward(dmodel, x_cloud_enc, device)               # [N,1,L] declouded contrast

            err_cloud = float(((x_cloud_enc[:, 0:1, :] - clear_contrast) ** 2).mean())
            err_decl = float(((declouded - clear_contrast) ** 2).mean())
            row = dict(alpha=float(alpha), family=family,
                       recon_mse_cloudy=err_cloud, recon_mse_declouded=err_decl,
                       recon_recovery=(1.0 - err_decl / err_cloud) if err_cloud > 0 else float("nan"))
            if has_retrieval:
                declouded_enc = torch.cat([declouded, x_cloud_enc[:, 1:2, :]], dim=1)
                r2_cloudy = _retrieval_r2(rmodel, x_cloud_enc, y_true, lp, device)
                r2_decl = _retrieval_r2(rmodel, declouded_enc, y_true, lp, device)
                gap = r2_clear - r2_cloudy
                row.update(r2_clear=r2_clear, r2_cloudy=r2_cloudy, r2_declouded=r2_decl,
                           r2_recovery=(r2_decl - r2_cloudy) / gap if abs(gap) > 1e-9 else float("nan"))
            rows.append(row)
            _print_row(row, has_retrieval)

    summary = dict(track=track, seed=seed, decloud_ckpt=os.path.abspath(decloud_ckpt),
                   decloud_best_val=dck.get("best_val"), n_test=int(raw_x.shape[0]),
                   has_retrieval=has_retrieval, families=list(families),
                   alphas=[float(a) for a in alphas], rows=rows)
    out_json = os.path.join(OUT_DIR, f"decloud_eval_{track}_seed{seed}{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    _write_table(track, seed, suffix, rows, has_retrieval)
    if has_retrieval:
        _plot_recovery(track, seed, suffix, rows, families, alphas)
    if plot_examples:
        _plot_examples(dmodel, raw_x, noise, norm, wl, cont, families, device,
                       n_examples=plot_examples)
    print(f"=> wrote {out_json}")
    return summary


def _print_row(row, has_retrieval):
    base = f"  α={row['alpha']:<6g} {row['family']:<15} recon↓{row['recon_recovery']:+.2f}"
    if has_retrieval:
        base += (f" | R² clear {row['r2_clear']:.3f} cloudy {row['r2_cloudy']:.3f} "
                 f"declouded {row['r2_declouded']:.3f}  recovery {row['r2_recovery']:+.2f}")
    print(base)


def _write_table(track, seed, suffix, rows, has_retrieval):
    L = ["=" * 92,
         f"DECLOUD EVALUATION — retrieval track '{track}' seed {seed}   (frozen, inference-only)",
         "=" * 92,
         "recon_recovery = 1 − MSE(declouded,clear)/MSE(cloudy,clear)  (fraction of cloud error removed)"]
    if has_retrieval:
        L.append("R²_recovery    = (R²_declouded − R²_cloudy)/(R²_clear − R²_cloudy)  (fraction of accuracy restored)")
        L.append("-" * 92)
        L.append(f"{'alpha':<8}{'family':<16}{'R2_clear':<10}{'R2_cloudy':<11}{'R2_declouded':<14}"
                 f"{'R2_recov':<10}{'recon_recov':<11}")
        for r in rows:
            L.append(f"{r['alpha']:<8g}{r['family']:<16}{r['r2_clear']:<10.4f}{r['r2_cloudy']:<11.4f}"
                     f"{r['r2_declouded']:<14.4f}{r['r2_recovery']:<10.3f}{r['recon_recovery']:<11.3f}")
    else:
        L.append("-" * 92)
        L.append(f"{'alpha':<8}{'family':<16}{'recon_recovery':<16}{'mse_cloudy':<14}{'mse_declouded':<14}")
        for r in rows:
            L.append(f"{r['alpha']:<8g}{r['family']:<16}{r['recon_recovery']:<16.3f}"
                     f"{r['recon_mse_cloudy']:<14.5f}{r['recon_mse_declouded']:<14.5f}")
    L.append("=" * 92)
    txt = "\n".join(L) + "\n"
    with open(os.path.join(OUT_DIR, f"decloud_eval_{track}_seed{seed}{suffix}.txt"), "w") as f:
        f.write(txt)
    print(txt)


def _plot_recovery(track, seed, suffix, rows, families, alphas):
    """One panel per family: R² vs α for clear / cloudy / declouded."""
    fams = list(families)
    fig, axes = plt.subplots(1, len(fams), figsize=(4.2 * len(fams), 4), squeeze=False)
    for k, fam in enumerate(fams):
        ax = axes[0][k]
        fr = [r for r in rows if r["family"] == fam]
        xs = [r["alpha"] for r in fr]
        ax.plot(xs, [r["r2_clear"] for r in fr], "o-", color="seagreen", label="clear")
        ax.plot(xs, [r["r2_cloudy"] for r in fr], "o-", color="firebrick", label="cloudy")
        ax.plot(xs, [r["r2_declouded"] for r in fr], "o-", color="royalblue", label="declouded")
        ax.set_xscale("log")
        ax.set_title(f"{fam}" + ("  (trained)" if fam == "grey" else "  (held-out)"))
        ax.set_xlabel("exposure α  (√(t/t_nom))")
        ax.set_ylabel("overall R² (log10)")
        ax.grid(True, which="both", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Declouding restores frozen-{track} retrieval accuracy under clouds", y=1.02)
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"recovery_{track}_seed{seed}{suffix}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"=> wrote {out}")


def _plot_examples(dmodel, raw_x, noise, norm, wl, cont, families, device, n_examples=4, alpha=10.0):
    """Eyeball a few spectra: clear vs cloudy vs declouded (asinh-normed contrast)."""
    fam = families[0]
    x_cloud_raw = apply_family_to_raw(raw_x, fam, wl, continuum=cont)
    x_cloud_enc = build_eval_observable(x_cloud_raw, noise, "contrast_snr", norm,
                                        alpha=alpha, seed=TEST_NOISE_SEED)
    clear_enc = norm.encode(make_observable(raw_x, noise, "contrast_snr"))[:, 0:1, :]
    declouded = _forward(dmodel, x_cloud_enc, device)
    idx = np.linspace(0, raw_x.shape[0] - 1, n_examples).astype(int)
    fig, axes = plt.subplots(n_examples, 1, figsize=(11, 2.4 * n_examples), squeeze=False)
    for r, i in enumerate(idx):
        ax = axes[r][0]
        ax.plot(wl, clear_enc[i, 0].numpy(), lw=0.8, color="seagreen", label="clear (target)")
        ax.plot(wl, x_cloud_enc[i, 0].numpy(), lw=0.8, color="firebrick", alpha=0.7, label="cloudy (input)")
        ax.plot(wl, declouded[i, 0].numpy(), lw=0.8, color="royalblue", alpha=0.9, label="declouded")
        ax.set_ylabel("asinh-normed C")
        if r == 0:
            ax.legend(fontsize=8, ncol=3, loc="upper right")
            ax.set_title(f"Declouder examples — family '{fam}', α={alpha:g}")
    axes[-1][0].set_xlabel("wavelength (µm)")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, f"examples_{fam}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"=> wrote {out}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate the declouder (recon + frozen-retrieval recovery).")
    ap.add_argument("--decloud-ckpt", required=True)
    ap.add_argument("--track", default="transformerarch", help="frozen retrieval track to score")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="retrieval checkpoint suffix")
    ap.add_argument("--families", nargs="*", default=list(FAMILIES))
    ap.add_argument("--alphas", type=float, nargs="*", default=[3.0, 10.0, 30.0, 100.0])
    ap.add_argument("--n", type=int, default=4000, help="max test planets (speed)")
    ap.add_argument("--cache", default=CACHE_V2)
    ap.add_argument("--plot-examples", type=int, default=0, help="save N example spectra")
    a = ap.parse_args()
    evaluate(a.decloud_ckpt, a.track, a.seed, a.families, a.alphas, n=a.n, suffix=a.suffix,
             cache_dir=a.cache, plot_examples=a.plot_examples)


if __name__ == "__main__":
    main()
