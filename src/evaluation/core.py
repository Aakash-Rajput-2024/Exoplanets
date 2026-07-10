"""Shared core for the unified evaluation pipeline (``src/evaluation``).

Every eval *suite* (in_distribution, cross_generator, solar_system, real_earth,
calibration, ...) is scored through THIS module, so a trained checkpoint is
evaluated identically everywhere:

  * the v2 leak-free observable  [contrast C=F_p/F_star, stellar-SNR]  (common.observable)
  * the INARA-**train**-fit per-λ asinh input norm                     (common.pipeline.get_norm)
  * the CLR label decode → guaranteed-valid simplex predictions        (common.transforms.LabelPipeline)

It REUSES ``common.*`` rather than re-implementing the pipeline — this is the
single seam that keeps every eval consistent with training. Nothing here trains
or mutates INARA data/stats; all outputs live under
``src/evaluation/results/<track>/<suite>/``.

The two workhorses:
  * ``predict_cache``  — score a v2-format cache dir (INARA test, cross-gen, PSG anchor).
  * ``predict_raw``    — score in-memory (raw_x, noise) arrays a suite synthesised
                         itself (real Earth, solar-system analytic engine), using the
                         INARA-train norm/label pipeline so the weights are scored fairly.
"""
from __future__ import annotations

import dataclasses
import datetime
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
SRC = os.path.join(REPO, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from common.registry import track_config, CACHE_V2, MATCHED           # noqa: E402
from common.runtime import get_device                                  # noqa: E402
from common.data import label_pipeline as inara_label_pipeline         # noqa: E402
from common.pipeline import load_eval_raw, get_norm, TEST_NOISE_SEED   # noqa: E402
from common.inputs import build_eval_observable                        # noqa: E402
from common.observable import make_observable, snr_summary, SNR_CLAMP  # noqa: E402,F401
from common import metrics                                             # noqa: E402
from common.evaluate import (                                          # noqa: E402
    _load_model, _predict, _ckpt_path, _ckpt_provenance,
    DEFAULT_ALPHAS, MAX_TRAINED_ALPHA, TARGET_COLUMNS,
)

EVAL_DIR = os.path.join(SRC, "evaluation")
RESULTS_ROOT = os.path.join(EVAL_DIR, "results")
BENCH_DIR = os.path.join(EVAL_DIR, "benchmarks")

# Species that reflected/thermal 0.2–2.0 µm forward models actually imprint on the
# spectrum AND that overlap INARA's 12 targets. Uncovered species carry real labels
# but no spectral signal, so their R² is uninformative by construction — every suite
# reports a covered-species headline plus the honest all-12 number alongside it.
COVERED_DEFAULT = ["H2O", "CO2", "O2", "CH4", "O3"]


def covered_idx(covered=None):
    covered = COVERED_DEFAULT if covered is None else covered
    return [TARGET_COLUMNS.index(s) for s in covered if s in TARGET_COLUMNS]


# --------------------------------------------------------------------------- #
# Evaluation context: the (track, seeds, device) a suite runs against.
# --------------------------------------------------------------------------- #
@dataclass
class EvalContext:
    track: str
    seeds: list
    suffix: str = ""
    alphas: tuple = DEFAULT_ALPHAS
    cache_v2: str = CACHE_V2
    device: object = None

    def __post_init__(self):
        if self.device is None:
            self.device = get_device()
        # Keep only seeds whose checkpoint exists on disk (offline-safe: a partly
        # trained track still evaluates on the seeds it does have).
        self.seeds = [s for s in self.seeds
                      if os.path.exists(_ckpt_path(self.track, s, self.suffix))]
        self._models = {}

    @property
    def label(self):
        return f"{self.track}{self.suffix}"

    @property
    def n_seeds(self):
        return len(self.seeds)

    @property
    def has_checkpoints(self):
        return len(self.seeds) > 0

    def out_dir(self, suite):
        d = os.path.join(RESULTS_ROOT, self.label, suite)
        os.makedirs(d, exist_ok=True)
        return d

    def model(self, seed):
        """Load (and cache) a model for a seed — avoids re-reading the .pth across
        the many-α sweeps a single suite runs."""
        if seed not in self._models:
            self._models[seed] = _load_model(self.track, seed, self.device, self.suffix)
        return self._models[seed][0]

    def ckpt_config(self):
        """obs_mode / input_norm / label_transform, read from the checkpoint the
        model was actually trained with (robust to later registry/flag changes)."""
        _, ck_cfg = _load_model(self.track, self.seeds[0], self.device, self.suffix)
        cfg = track_config(self.track)
        return (
            ck_cfg.get("obs_mode", cfg["obs_mode"]),
            ck_cfg.get("input_norm", cfg["input_norm"]),
            ck_cfg.get("label_transform", cfg["label_transform"]),
        )

    def provenance(self):
        return _ckpt_provenance(self.track, self.seeds[0], self.device, self.suffix)


# --------------------------------------------------------------------------- #
# Prediction: the two entry points every suite uses.
# --------------------------------------------------------------------------- #
def free_device(device):
    """Release the framework's cached device memory. On MPS the caching allocator
    fragments across the many forward passes an all-section run makes, which slows
    the transformer backbone ~5× over a long process (and can look like a hang);
    emptying the cache between passes keeps every pass as fast as the first."""
    try:
        if getattr(device, "type", None) == "mps":
            torch.mps.empty_cache()
        elif getattr(device, "type", None) == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass


def _decode_predict(ctx, x_encoded, lp):
    """Run every seed on an already-encoded observable, decode via the CLR label
    pipeline, return (per_seed_preds[list of [N,12]], ensemble[N,12])."""
    preds = []
    for s in ctx.seeds:
        preds.append(lp.decode(_predict(ctx.model(s), x_encoded, ctx.device)).numpy())
        free_device(ctx.device)
    ens = np.mean(preds, axis=0)
    return preds, ens


def predict_cache(ctx, cache_dir, alpha=1.0, noiseless=False, seed=TEST_NOISE_SEED,
                  obs_mode=None, input_norm=None, label_transform=None):
    """Score a v2-format cache directory (``test_{x,y,noise}.pt`` + symlinked
    INARA-train derived stats). Returns a dict with raw handles so callers can run
    an α-sweep without reloading.

    ``noiseless=True`` builds the noiseless contrast reference (zero injected
    contrast noise, SNR channel held at MAX_TRAINED_ALPHA — in-distribution, NOT
    the OOD α→∞ clamp limit), matching common.evaluate's honest reference.
    """
    ob0, in0, lt0 = ctx.ckpt_config()
    obs_mode = obs_mode or ob0
    input_norm = input_norm or in0
    label_transform = label_transform or lt0

    raw_x, y_lin, noise, ids, lp = load_eval_raw(cache_dir, "test", label_transform)
    norm = get_norm(cache_dir, obs_mode, input_norm)
    if noiseless:
        x = norm.encode(make_observable(raw_x, noise, obs_mode, alpha=MAX_TRAINED_ALPHA))
    else:
        x = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=alpha, seed=seed)
    preds, ens = _decode_predict(ctx, x, lp)
    return dict(y_true=y_lin.numpy(), preds=preds, ens=ens, raw_x=raw_x, noise=noise,
                ids=ids, lp=lp, norm=norm, obs_mode=obs_mode, input_norm=input_norm,
                label_transform=label_transform)


def predict_raw(ctx, raw_x, noise, alpha=1.0, noiseless=False, seed=TEST_NOISE_SEED,
                obs_mode=None, input_norm=None, label_transform=None):
    """Score in-memory (raw_x[M,2,L], noise[M,L]) a suite synthesised itself
    (real Earth, solar-system analytic engine), using the INARA-**train** norm and
    label pipeline (from ctx.cache_v2) — so the trained weights are scored fairly
    and no statistic is refit on the eval spectra.
    """
    ob0, in0, lt0 = ctx.ckpt_config()
    obs_mode = obs_mode or ob0
    input_norm = input_norm or in0
    label_transform = label_transform or lt0

    if not torch.is_tensor(raw_x):
        raw_x = torch.as_tensor(np.asarray(raw_x), dtype=torch.float32)
    if not torch.is_tensor(noise):
        noise = torch.as_tensor(np.asarray(noise), dtype=torch.float32)

    lp = inara_label_pipeline(ctx.cache_v2, label_transform)
    norm = get_norm(ctx.cache_v2, obs_mode, input_norm)
    if noiseless:
        x = norm.encode(make_observable(raw_x, noise, obs_mode, alpha=MAX_TRAINED_ALPHA))
    else:
        x = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=alpha, seed=seed)
    preds, ens = _decode_predict(ctx, x, lp)
    return dict(preds=preds, ens=ens, raw_x=raw_x, noise=noise, lp=lp, norm=norm,
                obs_mode=obs_mode, input_norm=input_norm, label_transform=label_transform)


def encode_cache(ctx, cache_dir, alpha=1.0, noiseless=False, seed=TEST_NOISE_SEED,
                 obs_mode=None, input_norm=None, label_transform=None):
    """Return the ENCODED observable (numpy [N,C,L]) + raw handles for a cache dir —
    what OOD/honesty (δ/v, anomaly score) inspect. No model is run."""
    ob0, in0, lt0 = ctx.ckpt_config()
    obs_mode = obs_mode or ob0
    input_norm = input_norm or in0
    label_transform = label_transform or lt0
    raw_x, y_lin, noise, ids, lp = load_eval_raw(cache_dir, "test", label_transform)
    norm = get_norm(cache_dir, obs_mode, input_norm)
    if noiseless:
        x = norm.encode(make_observable(raw_x, noise, obs_mode, alpha=MAX_TRAINED_ALPHA))
    else:
        x = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=alpha, seed=seed)
    return x.numpy(), raw_x, noise, y_lin.numpy(), lp


def encode_raw(ctx, raw_x, noise, alpha=1.0, noiseless=False, seed=TEST_NOISE_SEED,
               obs_mode=None, input_norm=None):
    """Encode an in-memory (raw_x, noise) with the INARA-train norm (numpy [N,C,L])."""
    ob0, in0, _ = ctx.ckpt_config()
    obs_mode = obs_mode or ob0
    input_norm = input_norm or in0
    if not torch.is_tensor(raw_x):
        raw_x = torch.as_tensor(np.asarray(raw_x), dtype=torch.float32)
    if not torch.is_tensor(noise):
        noise = torch.as_tensor(np.asarray(noise), dtype=torch.float32)
    norm = get_norm(ctx.cache_v2, obs_mode, input_norm)
    if noiseless:
        x = norm.encode(make_observable(raw_x, noise, obs_mode, alpha=MAX_TRAINED_ALPHA))
    else:
        x = build_eval_observable(raw_x, noise, obs_mode, norm, alpha=alpha, seed=seed)
    return x.numpy()


# --------------------------------------------------------------------------- #
# Standard scoring helpers (thin wrappers over common.metrics).
# --------------------------------------------------------------------------- #
def score(y_true, y_pred, covered=None, bootstrap=1000):
    """Per-species R²/RMSE/MAE(dex) in log10 space + covered/all-12 means.

    Returns a plain dict (JSON-friendly). ``covered`` defaults to the species the
    forward models imprint; pass an explicit list for engine-specific coverage.
    """
    rep = metrics.report(y_true, y_pred, space="log10", bootstrap=bootstrap)
    cov = covered_idx(covered)
    r2 = np.asarray(rep["r2"])
    out = {
        "per_species": {c: {
            "r2": _f(rep["r2"][i]), "rmse_dex": _f(rep["rmse"][i]), "mae_dex": _f(rep["mae"][i]),
            **({"r2_ci": [_f(rep["r2_lo"][i]), _f(rep["r2_hi"][i])]} if "r2_lo" in rep else {}),
        } for i, c in enumerate(TARGET_COLUMNS)},
        "r2_covered": _f(np.nanmean(r2[cov])) if cov else None,
        "r2_all12": _f(np.nanmean(r2)),
        "rmse_all12_dex": _f(np.nanmean(rep["rmse"])),
        "mae_all12_dex": _f(np.nanmean(rep["mae"])),
        "covered_species": [TARGET_COLUMNS[i] for i in cov],
    }
    return out


# --------------------------------------------------------------------------- #
# SuiteResult: the common artifact every suite emits.
# --------------------------------------------------------------------------- #
@dataclass
class SuiteResult:
    section: str                 # "E"
    suite: str                   # "solar_system"  (dir slug)
    title: str                   # human title
    track: str
    status: str = "ok"           # ok | skipped | preliminary | error
    epistemic: str = ""          # one-line "what this proves" tag (ground truth / probe / ...)
    headline: dict = field(default_factory=dict)   # small metric key→value
    tables: list = field(default_factory=list)     # [{"name":..,"columns":[..],"rows":[[..]]}]
    caveats: list = field(default_factory=list)
    artifacts: list = field(default_factory=list)  # file paths written
    provenance: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)        # full machine-readable payload

    @classmethod
    def skipped(cls, section, suite, title, track, reason):
        return cls(section, suite, title, track, status="skipped", caveats=[reason])


def _f(x):
    """Coerce numpy/torch scalars to plain JSON-safe floats (NaN preserved)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def write_result(ctx, res: SuiteResult):
    """Persist a SuiteResult as result.json under results/<track>/<suite>/."""
    d = ctx.out_dir(res.suite)
    payload = dataclasses.asdict(res)
    payload["generated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    path = os.path.join(d, "result.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)
    if path not in res.artifacts:
        res.artifacts.append(path)
    return res


# --------------------------------------------------------------------------- #
# Markdown rendering (shared by run_eval's report + each suite's own txt).
# --------------------------------------------------------------------------- #
_STATUS_ICON = {"ok": "✅", "skipped": "⏭️", "preliminary": "🟡", "error": "❌"}


def render_markdown(res: SuiteResult):
    L = [f"## Section {res.section} — {res.title}  {_STATUS_ICON.get(res.status, '')}"]
    if res.epistemic:
        L.append(f"*{res.epistemic}*")
    L.append("")
    if res.status == "skipped":
        L.append(f"> **Skipped.** {'; '.join(res.caveats) or 'inputs not present'}")
        L.append("")
        return "\n".join(L)
    if res.headline:
        L.append("| metric | value |")
        L.append("|---|---|")
        for k, v in res.headline.items():
            L.append(f"| {k} | {_fmt(v)} |")
        L.append("")
    for tbl in res.tables:
        L.append(f"**{tbl.get('name', 'table')}**")
        L.append("")
        cols = tbl["columns"]
        L.append("| " + " | ".join(str(c) for c in cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for row in tbl["rows"]:
            L.append("| " + " | ".join(_fmt(c) for c in row) + " |")
        L.append("")
    if res.caveats:
        L.append("**Caveats**")
        for c in res.caveats:
            L.append(f"- {c}")
        L.append("")
    if res.artifacts:
        rels = [os.path.relpath(a, RESULTS_ROOT) for a in res.artifacts]
        L.append("<sub>artifacts: " + ", ".join(f"`{r}`" for r in rels) + "</sub>")
        L.append("")
    return "\n".join(L)


def _fmt(v):
    if isinstance(v, float):
        if v != v:            # NaN
            return "NaN"
        if abs(v) != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.3e}"
        return f"{v:.4f}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt(x) for x in v) + "]"
    return str(v)
