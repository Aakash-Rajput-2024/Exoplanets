"""Unified matched-budget trainer for every track (fixes C4, H6, C6, M1/M2/M3/M4).

One loop trains any architecture in the registry under the identical budget, for
each seed, on the physical observable with CLR labels and per-λ asinh encoding.
It replaces the five drifted ``train.py`` copies (M5). Per run it writes, under
``<track_dir>/{checkpoints_v2,charts_v2,logs_v2}/``:

  * ``model_best_seed{S}.pth`` — best-val checkpoint, provenance-stamped (C6),
  * ``logs_seed{S}.json`` / ``.csv`` — per-epoch train/val Huber loss, LR, and
    the real retrieval metric (val log₁₀-R², decoded through the CLR pipeline),
  * ``loss_curve_seed{S}.png``.

Usage:
    python -m common.train_runner <track> [--seed S] [--epochs E] [--smoke]
    (omitting --seed runs every seed in MATCHED["seeds"])
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, desc=""):
        return it

from common.registry import track_config, load_model_class, REPO, CACHE_V2
from common.runtime import seed_everything, get_device
from common.pipeline import build_datasets, load_eval_raw, VAL_NOISE_SEED
from common.inputs import build_eval_observable
from common.provenance import stamp_checkpoint
from common import metrics

# Fixed exposures at which the OPTIONAL --diag pass reports val R² each epoch.
# Purely diagnostic (a live view of the R²-vs-exposure curve); NEVER used for
# model selection, which stays on the averaged val loss (the matched-budget rule).
DIAG_ALPHAS = (1.0, 10.0, 100.0)


def _cosine_warmup(optimizer, warmup_epochs, total_epochs, steps_per_epoch):
    warmup = warmup_epochs * steps_per_epoch
    total = total_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warmup:
            return float(step) / float(max(1, warmup))
        progress = float(step - warmup) / float(max(1, total - warmup))
        # float(): np.cos returns np.float64, which would poison the saved LR
        # history and break weights_only checkpoint loading (PyTorch 2.6+).
        return float(max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _make_loss(name):
    return {"huber": nn.HuberLoss(delta=1.0),
            "mse": nn.MSELoss(),
            "l1": nn.L1Loss()}[name]


def _val_pass(model, loader, criterion, device, label_pipe):
    """Return (mean val loss, overall val log₁₀-R²). R² is the honest retrieval
    metric — decode both predictions and targets back to linear mole fractions
    (decode∘encode is identity for the true labels) and score in log10 space."""
    model.eval()
    vloss, preds, trues = 0.0, [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            vloss += criterion(p, y).item()
            preds.append(p.cpu())
            trues.append(y.cpu())
    vloss /= max(1, len(loader))
    p = label_pipe.decode(torch.cat(preds)).numpy()
    t = label_pipe.decode(torch.cat(trues)).numpy()
    r2 = metrics.r2_per_species(metrics.to_space(t, "log10"), metrics.to_space(p, "log10"))
    return vloss, float(np.nanmean(r2))


@torch.no_grad()
def _predict_batched(model, x, device, batch=256):
    model.eval()
    out = [model(x[i:i + batch].to(device)).cpu() for i in range(0, x.shape[0], batch)]
    return torch.cat(out)


def _diag_val_r2(model, diag_raw, obs_mode, norm, label_pipe, device,
                 alphas=DIAG_ALPHAS, seed=VAL_NOISE_SEED):
    """Per-α fixed-exposure val R²(log10) — a live view of the R²-vs-exposure
    curve during training. DIAGNOSTIC ONLY: never used for model selection (that
    stays on the averaged val loss). Non-fatal — any failure returns NaN for that
    α so training is never interrupted."""
    raw_x, noise, y_true = diag_raw
    res = {}
    for a in alphas:
        try:
            xa = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=a, seed=seed)
            pred = label_pipe.decode(_predict_batched(model, xa, device)).numpy()
            res[a] = float(np.nanmean(metrics.r2_per_species(
                metrics.to_space(y_true, "log10"), metrics.to_space(pred, "log10"))))
        except Exception as e:  # never let a diagnostic kill a training run
            print(f"  [diag] α={a} failed: {e}")
            res[a] = float("nan")
    return res


def _save_resume(path, next_epoch, model, optimizer, scheduler, loader_gen,
                 best_val, no_improve, history,
                 batch_idx=0, running_loss=0.0, epoch_gen=None, epoch_elapsed=0.0):
    """Atomically write a full resumable state: model + optimizer + scheduler +
    RNG states + loader shuffle position + bookkeeping. Written to a temp file and
    os.replace'd, so a crash mid-save cannot corrupt the resume point.

    ``batch_idx`` > 0 marks a MID-EPOCH snapshot: ``next_epoch`` is then the epoch
    still in progress rather than the next one to start, and ``epoch_gen`` holds
    the loader RNG state as it was at that epoch's FIRST batch. Restoring that
    state rebuilds the identical shuffle order, so the resumed run can fast-forward
    over the ``batch_idx`` batches it already trained on and continue on exactly
    the samples it had not reached.
    """
    blob = {
        "next_epoch": next_epoch,
        "batch_idx": batch_idx,
        "running_loss": running_loss,
        "epoch_gen": epoch_gen,
        "epoch_elapsed": epoch_elapsed,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "sched": scheduler.state_dict(),
        "best_val": best_val,
        "no_improve": no_improve,
        "history": history,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "loader_gen": loader_gen.get_state() if loader_gen is not None else None,
    }
    try:
        blob["mps_rng"] = (torch.mps.get_rng_state()
                           if hasattr(torch, "mps") and torch.backends.mps.is_available()
                           and hasattr(torch.mps, "get_rng_state") else None)
    except Exception:
        blob["mps_rng"] = None
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)


def _load_resume(path, model, optimizer, scheduler, loader_gen, device):
    """Restore a state written by :func:`_save_resume`. Returns
    (start_epoch, best_val, no_improve, history)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optim"])
    scheduler.load_state_dict(ck["sched"])
    try:  # RNG restore is best-effort; a partial restore only perturbs augmentation
        torch.set_rng_state(ck["torch_rng"].cpu())
        np.random.set_state(ck["numpy_rng"])
        random.setstate(ck["python_rng"])
        if loader_gen is not None and ck.get("loader_gen") is not None:
            loader_gen.set_state(ck["loader_gen"].cpu())
        if ck.get("mps_rng") is not None and hasattr(torch, "mps") and hasattr(torch.mps, "set_rng_state"):
            torch.mps.set_rng_state(ck["mps_rng"].cpu())
    except Exception as e:
        print(f"  [resume] RNG restore partial ({e}); trajectory may differ slightly.")
    # .get() defaults keep checkpoints written before mid-epoch support loadable:
    # they simply resume at an epoch boundary, as they always did.
    return (ck["next_epoch"], ck["best_val"], ck["no_improve"], ck["history"],
            ck.get("batch_idx", 0), ck.get("running_loss", 0.0),
            ck.get("epoch_gen"), ck.get("epoch_elapsed", 0.0))


def train_one(track, seed, cache_dir=CACHE_V2, epochs=None, smoke=False,
              num_workers=0, cloud=None, resume=False, diag=False, cf=False,
              cf_invariance=False, lambda_inv=None, save_every=300):
    cfg = track_config(track)
    if epochs is not None:
        cfg["epochs"] = epochs
    if lambda_inv is not None:
        cfg["lambda_inv"] = lambda_inv
    if cf and cf_invariance:
        raise SystemExit("--cf and --cf-invariance are mutually exclusive (augmentation vs objective).")
    if cloud and (cf or cf_invariance):
        print("  [warn] --cloud takes precedence over --cf/--cf-invariance in the "
              "pipeline; the counterfactual objective will NOT be applied this run.")
    cfg["cloud_train"] = cloud
    cfg["cf_train"] = cf
    cfg["cf_invariance"] = cf_invariance
    # Distinct checkpoint suffix PER TRAINING OBJECTIVE so a counterfactual run can
    # never silently overwrite the plain control checkpoint (2026-07-08 audit fix):
    #   _cfi = counterfactual-invariance objective, _cf = cf augmentation,
    #   (none) = control; grey clouds append _grey as before.
    obj = "_cfi" if cf_invariance else ("_cf" if cf else "")
    suffix = obj + (f"_{cloud}" if cloud else "")
    device = get_device()
    track_dir = os.path.dirname(cfg["model_py"])
    ckpt_dir = os.path.join(track_dir, "checkpoints_v2")
    charts_dir = os.path.join(track_dir, "charts_v2")
    logs_dir = os.path.join(track_dir, "logs_v2")
    for d in (ckpt_dir, charts_dir, logs_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n=== {track} | seed {seed} | {device} | {cfg['desc']} ===")
    seed_everything(seed)

    data = build_datasets(
        cache_dir, obs_mode=cfg["obs_mode"], label_transform=cfg["label_transform"],
        input_norm=cfg["input_norm"], alpha=cfg["alpha"],
        alpha_train_range=cfg.get("alpha_train_range"), train_seed=seed,
        cloud_train=cloud, cf_train=cf, cf_invariance=cf_invariance,
    )
    train_ds, val_ds = data.train_ds, data.val_ds
    if smoke:
        train_ds = Subset(train_ds, range(min(512, len(train_ds))))
        val_ds = Subset(val_ds, range(min(256, len(val_ds))))
        cfg["epochs"] = 1
        print("  [SMOKE] 1 epoch on a tiny subset")

    print(f"  in_channels={data.in_channels} seq_len={data.seq_len} "
          f"train={len(train_ds)} val={len(val_ds)}")
    ModelClass = load_model_class(cfg["model_py"], cfg["model_cls"])
    model = ModelClass(in_channels=data.in_channels, sequence_length=data.seq_len).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params:,}")

    # Resumable runs use an explicit shuffle generator so the minibatch order can
    # be restored exactly; the default (non-resume) path keeps the original
    # global-RNG shuffle, byte-for-byte unchanged.
    loader_gen = torch.Generator().manual_seed(seed) if resume else None
    loader_kwargs = dict(batch_size=cfg["batch_size"], shuffle=True,
                         pin_memory=(device.type == "cuda"), num_workers=num_workers,
                         collate_fn=data.train_collate)
    if loader_gen is not None:
        loader_kwargs["generator"] = loader_gen
    train_loader = DataLoader(train_ds, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=num_workers)

    criterion = _make_loss(cfg["loss"])
    inv_criterion = nn.MSELoss()          # counterfactual-invariance penalty (CLR space)
    lam_inv = float(cfg.get("lambda_inv", 1.0))
    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = _cosine_warmup(optimizer, cfg["warmup_epochs"], cfg["epochs"], len(train_loader))

    best_val, no_improve = float("inf"), 0
    history = []  # per-epoch dicts
    start_epoch = 0
    resume_path = os.path.join(ckpt_dir, f"last_{track}_seed{seed}{suffix}.pth")
    resume_batch, resume_loss, resume_gen, resume_elapsed = 0, 0.0, None, 0.0
    if resume and os.path.exists(resume_path) and not smoke:
        (start_epoch, best_val, no_improve, history,
         resume_batch, resume_loss, resume_gen, resume_elapsed) = _load_resume(
            resume_path, model, optimizer, scheduler, loader_gen, device)
        where = (f"epoch {start_epoch + 1}/{cfg['epochs']} batch {resume_batch}"
                 if resume_batch else f"epoch {start_epoch + 1}/{cfg['epochs']}")
        print(f"  [resume] {os.path.basename(resume_path)}: continuing at "
              f"{where} (best_val={best_val:.5f})")
        if start_epoch >= cfg["epochs"]:
            print("  [resume] run already complete — skipping training.")

    # Optional per-α diagnostic val set (raw handles built once; observables are
    # re-formed cheaply each epoch). Setup is non-fatal.
    diag_raw = None
    if diag:
        try:
            dvx, dvy, dvnoise, _ids, _lp = load_eval_raw(cache_dir, "val", cfg["label_transform"])
            diag_raw = (dvx, dvnoise, dvy.numpy())
            print(f"  [diag] per-α val R² at α={DIAG_ALPHAS} enabled (N={dvx.shape[0]})")
        except Exception as e:
            print(f"  [diag] disabled (setup failed: {e})")

    # Outer bar tracks the whole run; the inner bar (below) tracks one epoch, so a
    # long job shows both "how far into this epoch" and "how far overall".
    epoch_bar = tqdm(total=cfg["epochs"], initial=start_epoch, position=0, leave=True,
                     desc=f"{track} s{seed}", unit="ep",
                     bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}]")

    for epoch in range(start_epoch, cfg["epochs"]):
        # Fresh, reproducible noise realization this epoch (C1 augmentation).
        if data.train_collate is not None and hasattr(data.train_collate, "set_epoch"):
            data.train_collate.set_epoch(epoch)

        # The shuffle permutation is drawn from loader_gen when the iterator is
        # created, so the generator state must be positioned BEFORE that happens:
        # restore the saved start-of-epoch state when resuming into a partial
        # epoch, otherwise record the current state as this epoch's anchor.
        if resume_batch and resume_gen is not None and loader_gen is not None:
            loader_gen.set_state(resume_gen.cpu())
            epoch_gen = resume_gen
        else:
            epoch_gen = loader_gen.get_state() if loader_gen is not None else None

        model.train()
        tl = resume_loss
        t0 = time.time() - resume_elapsed
        last_save = time.time()
        n_batches = len(train_loader)
        batch_bar = tqdm(total=n_batches, initial=resume_batch, position=1, leave=False,
                         desc=f"  E{epoch+1}/{cfg['epochs']}", unit="b")
        for bi, (x, y) in enumerate(train_loader):
            # Fast-forward: these batches were already trained on before the
            # interrupt. Optimizer/scheduler/model state already reflect them, so
            # they must be consumed (to advance the iterator) but not re-applied.
            if bi < resume_batch:
                continue
            optimizer.zero_grad()
            if x.dim() == 4:
                # (2, B, C, L) anchor/counterfactual pair (CFInvarianceCollate):
                # task loss on the anchor + λ · invariance penalty across the exact
                # do(environment) intervention. Predictions must not move.
                y = y.to(device)
                pred_a = model(x[0].to(device))
                pred_cf = model(x[1].to(device))
                loss = criterion(pred_a, y) + lam_inv * inv_criterion(pred_a, pred_cf)
            else:
                x, y = x.to(device), y.to(device)
                loss = criterion(model(x), y)
            loss.backward()
            if cfg["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            tl += loss.item()
            batch_bar.update(1)

            # Mid-epoch snapshot. On a 940 s/epoch transformer run an interrupt
            # would otherwise discard up to a full epoch of work; this caps the
            # loss at `save_every` seconds.
            if resume and not smoke and (time.time() - last_save) >= save_every:
                _save_resume(resume_path, epoch, model, optimizer, scheduler,
                             loader_gen, best_val, no_improve, history,
                             batch_idx=bi + 1, running_loss=tl, epoch_gen=epoch_gen,
                             epoch_elapsed=time.time() - t0)
                last_save = time.time()
                batch_bar.set_postfix_str(f"ckpt@{bi+1}")
        batch_bar.close()
        resume_batch, resume_loss, resume_elapsed, resume_gen = 0, 0.0, 0.0, None
        tl /= max(1, n_batches)

        vl, vr2 = _val_pass(model, val_loader, criterion, device, data.label_pipe)
        lr_now = float(optimizer.param_groups[0]["lr"])
        dt = time.time() - t0
        h = dict(epoch=epoch + 1, train_loss=tl, val_loss=vl,
                 val_r2_log10=vr2, lr=lr_now, sec=dt)
        diag_str = ""
        if diag_raw is not None:
            dr = _diag_val_r2(model, diag_raw, cfg["obs_mode"], data.norm, data.label_pipe, device)
            for a in DIAG_ALPHAS:
                h[f"val_r2_a{int(a)}"] = dr.get(a, float("nan"))
            diag_str = " [" + " ".join(f"α{int(a)}={dr[a]:.3f}" for a in DIAG_ALPHAS) + "]"
        print(f"  epoch {epoch+1}: train={tl:.5f} val={vl:.5f} valR²(log10)={vr2:.4f}{diag_str} "
              f"lr={lr_now:.2e} ({dt:.0f}s)")
        history.append(h)

        is_best = vl < best_val
        best_val = min(best_val, vl)
        no_improve = 0 if is_best else no_improve + 1

        if is_best and not smoke:
            state = stamp_checkpoint({
                "epoch": epoch + 1, "seed": seed, "state_dict": model.state_dict(),
                "best_val_loss": best_val, "best_val_r2_log10": vr2,
                "in_channels": data.in_channels, "seq_len": data.seq_len,
                "history": history,
            }, {k: cfg[k] for k in cfg if k != "seeds"} | {"seed": seed}, repo_dir=REPO)
            torch.save(state, os.path.join(ckpt_dir, f"model_best_{track}_seed{seed}{suffix}.pth"))

        # Resumable snapshot AFTER the epoch (state now points at epoch+1). Only
        # when --resume; the default path writes nothing new.
        if resume and not smoke:
            # Re-generate resume_path to ensure we write to the track-specific name
            actual_resume_path = os.path.join(ckpt_dir, f"last_{track}_seed{seed}{suffix}.pth")
            # batch_idx=0 marks a clean epoch boundary, so a resume from here
            # starts epoch+1 fresh rather than fast-forwarding.
            _save_resume(actual_resume_path, epoch + 1, model, optimizer, scheduler,
                         loader_gen, best_val, no_improve, history, batch_idx=0)

        epoch_bar.update(1)
        epoch_bar.set_postfix(val=f"{vl:.4f}", r2=f"{vr2:.3f}")

        if no_improve >= cfg["patience"]:
            print(f"  early stopping (no improvement for {no_improve} epochs)")
            break
    epoch_bar.close()

    # --- persist logs + curves ------------------------------------------------
    tag = f"{track}_seed{seed}{suffix}"
    if not smoke:
        with open(os.path.join(logs_dir, f"logs_{tag}.json"), "w") as f:
            json.dump({"track": track, "seed": seed, "config":
                       {k: cfg[k] for k in cfg if k != "seeds"},
                       "best_val_loss": best_val, "history": history}, f, indent=2)
        with open(os.path.join(logs_dir, f"logs_{tag}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            w.writeheader()
            w.writerows(history)

        ep = [h["epoch"] for h in history]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5))
        a1.plot(ep, [h["train_loss"] for h in history], "o-", label="train")
        a1.plot(ep, [h["val_loss"] for h in history], "o-", label="val")
        a1.set(xlabel="epoch", ylabel=f"loss ({cfg['loss']})",
               title=f"{track} seed{seed} loss"); a1.legend(); a1.grid(True)
        a2.plot(ep, [h["val_r2_log10"] for h in history], "o-", color="green",
                label="avg(α) [selection]")
        if history and any(k.startswith("val_r2_a") for k in history[0]):
            for a in DIAG_ALPHAS:
                key = f"val_r2_a{int(a)}"
                if key in history[0]:
                    a2.plot(ep, [h.get(key, float("nan")) for h in history], "o-", label=f"α={int(a)}")
            a2.legend(fontsize=8)
        a2.set(xlabel="epoch", ylabel="val R² (log10)",
               title=f"{track} seed{seed} retrieval metric"); a2.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"loss_curve_{tag}.png"), dpi=150)
        plt.close()

    print(f"=== {track} seed{seed} done. best val loss={best_val:.5f} ===")
    return best_val


def main():
    ap = argparse.ArgumentParser(description="Matched-budget trainer (ULTRAPLAN C4).")
    ap.add_argument("track")
    ap.add_argument("--seed", type=int, default=None, help="single seed; default = all MATCHED seeds")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--cache", default=CACHE_V2)
    ap.add_argument("--cloud", default=None, choices=[None, "grey"],
                    help="grey-cloud training augmentation for the C5 seen/unseen study")
    ap.add_argument("--cf", action="store_true",
                    help="causal (do-calculus) AUGMENTATION: exact environment counterfactual "
                         "— same atmosphere, re-paired host star + noise (checkpoint suffix _cf)")
    ap.add_argument("--cf-invariance", action="store_true",
                    help="causal (do-calculus) INVARIANCE objective: penalise the retrieval for "
                         "changing under an exact do(environment) intervention "
                         "(L=task+λ·‖f(x)−f(x^do)‖²). Novel vs --cf. Checkpoint suffix _cfi")
    ap.add_argument("--lambda-inv", type=float, default=None,
                    help="override counterfactual-invariance strength λ (default MATCHED['lambda_inv'])")
    ap.add_argument("--resume", action="store_true",
                    help="write a resumable snapshot each epoch and continue from it if present "
                         "(checkpoints_v2/last_seed{S}.pth); default off leaves behavior unchanged")
    ap.add_argument("--diag", action="store_true",
                    help="log per-α fixed-exposure val R² each epoch (diagnostic only; "
                         "never used for model selection)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=300, metavar="SECONDS",
                    help="mid-epoch resume snapshot interval (default 300s). Only "
                         "active with --resume. Lower = less work lost to an "
                         "interrupt, at the cost of more checkpoint writes.")
    a = ap.parse_args()

    cfg = track_config(a.track)
    seeds = [a.seed] if a.seed is not None else list(cfg["seeds"])
    for s in seeds:
        train_one(a.track, s, cache_dir=a.cache, epochs=a.epochs,
                  smoke=a.smoke, num_workers=a.num_workers, cloud=a.cloud,
                  resume=a.resume, diag=a.diag, cf=a.cf,
                  cf_invariance=a.cf_invariance, lambda_inv=a.lambda_inv,
                  save_every=a.save_every)


if __name__ == "__main__":
    main()
