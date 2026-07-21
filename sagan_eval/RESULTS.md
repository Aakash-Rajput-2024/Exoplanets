# Real solar-system spectra as exoplanets — Carl Sagan Institute catalog

**Data.** Madden & Kaltenegger (2018), *Astrobiology* **18**(12) 1559, "A Catalog of Spectra,
Albedos, and Colors of Solar System Bodies for Exoplanet Comparison".
DOI [10.5281/zenodo.3930987](https://doi.org/10.5281/zenodo.3930986), CC-BY 4.0.
19 bodies, geometric albedo `Ag(λ)`, 0.45–2.5 µm.

These are **real measured reflectance spectra**. Everything Section E of the eval pipeline
scored before was synthesised by `evaluation/engines/reflected_engine.py`, a band-template
proxy — so this is the first time the models have been shown real planets.

Code: `sagan_eval/` (isolated; reads checkpoints and `data/cache_v2` read-only, writes only
here). Results: `sagan_results.json`, `bluegap_*.json`, three figures.

---

## TL;DR

1. **CH₄ detection is real and near-perfect.** Matched-resolution AUROC = **1.00** (4 of 6
   tracks) separating the 5 CH₄-rich bodies from 5 airless bodies. This is the first genuine
   detection result on real spectra in this repo.
2. **O₂ detection does not exist.** The Moon and Mercury — bare rocks — are assigned *more*
   O₂ than Earth. Earth ranks 3rd–10th of 19 in predicted O₂, never 1st. Predicted O₂ on
   rocks (≈0.40–0.46) sits at the model's INARA prior (0.328). The EVAL_REPORT observation
   that "every model predicts O₂-dominant on real Earth" is **the training prior, not a
   detection**.
3. **The declouder must not be used on real spectra.** It is not identity-safe: on a *clear*
   spectrum it drops R² from **+0.782 to −1.089**. Root cause is a one-line bug —
   `GREY_F_RANGE = (0.3, 1.0)`, so cloud opacity `f < 0.3` (including clear, `f = 0`) never
   appeared in training. Applying it to the Sagan bodies inverts the CH₄ bands and collapses
   CH₄ AUROC from 1.00 to 0.00–0.24.
4. **Resolution, not the missing blue, decides which bodies are usable.** Only **11 of 19**
   bodies are sampled densely enough for a quantitative claim. **Venus and Mars are not** —
   which removes the two most interesting CO₂ comparisons.

---

## 1. Ingest and its validation

`ingest.py`. Three properties were verified against the files rather than assumed:

- **Internal consistency.** `Spec_Sun / Albedo` recovers the *same* solar SED for every body
  (fractional difference 2 × 10⁻⁵ Moon vs Earth), so `Ag` is the only per-body quantity.
- **Telluric blow-ups.** These are ground-based spectra. In the deep telluric H₂O bands
  (1.36–1.41, 1.87–1.95 µm) the earthshine/moonshine ratio is ≈ 0/0 and the tabulated albedo
  explodes: Neptune spans −47 … +133; Earth −3.6 … +7.8. Physical geometric albedo is ≈[0,1.5].
  Such points are **masked and interpolated over, never clipped** — clipping would invent a
  flat bright band exactly where the data says nothing.
  A *global* MAD flags ~9% of every high-res body and 45% of Titan (whose albedo is genuinely
  near zero across the NIR CH₄ bands), so the spike detector uses a **local** rolling MAD.
- **The contrast channel depends only on `Ag`.** `Fp = Ag·F☉`, `Fstar = F☉`, so `F☉` cancels in
  `C = Fp/(Fstar+Fp)`. Checked numerically: shape Pearson = 1.00000000 across catalog / Planck /
  flat SEDs, and `pearson(C, Ag) = 0.9999998`. The SED choice only rescales C by a constant
  (Planck: ×0.991).

Absolute albedo normalisation differs between the catalog's underlying sources (Venus and
Mercury read low against textbook geometric albedo). This does not matter: `synth.build_raw`
median-matches each channel to the INARA scale, so only spectral **shape** survives — the same
contract Section E already ran under.

## 2. What the catalog can and cannot support (`bluegap.py`)

The catalog stops at 0.45 µm; the model's grid starts at 0.20 µm — **35% of its 4378 bins**,
and exactly where O₃'s Hartley (0.20–0.31) and Huggins (0.31–0.35) bands live. Bodies are also
sampled anywhere from R≈12 (Io, 30 points) to R≈870 (Earth, 1752 points), against a model
trained at R≈1900.

Both mutilations were applied to **INARA test planets, where truth is known**:

| variant | causal ΔR²_cov | optimized1dcnn ΔR²_cov |
|---|---|---|
| blue flat-fill below 0.45 µm | −0.078 | −0.007 |
| resolution → R=870 | −0.030 | +0.001 |
| **resolution → R=22** | **−1.108** | **−0.727** |
| bluefill + R=870 | −0.104 | −0.006 |

(`full` = 0.779 for causal, reproducing the known 0.781 ceiling.)

**The blue gap is a minor tax (−0.01 … −0.08); R≈22 sampling is fatal** (R² goes negative;
CO₂ alone loses 1.7–2.5). Two architectures agree.

Consequence: bodies at R ≥ 400 (Earth, Titan, the 4 giants, Moon, Enceladus, Dione, Rhea,
Ceres = **11 of 19**) support quantitative claims. Venus, Mars, Mercury, Io, Europa, Ganymede,
Callisto and Pluto do not — they are effectively broadband photometry. **Any "Venus CO₂"
result below is qualitative at best.**

## 3. Three epistemic classes (`truth.py`)

`solar_system_truth.json` had 8 bodies. The catalog adds 11, and they are the valuable ones:

| class | n | bodies | what it tests |
|---|---|---|---|
| terrestrial | 4 | Earth, Venus, Mars, Titan | accuracy |
| giant | 4 | Jupiter, Saturn, Uranus, Neptune | honesty (H₂/He outside the simplex) |
| **airless** | **11** | Mercury, Moon, Io, Europa, Ganymede, Callisto, Enceladus, Dione, Rhea, Ceres, Pluto | **false positives** |

The airless class is the control the pipeline never had. These bodies have **no atmosphere**;
the simplex must sum to 1, so the model has no "none of the above" head and *must* fabricate a
composition. Nothing else in the repo can distinguish "the model detected O₂" from "the model
always says O₂", because every other Section-E spectrum is generated from a gas mixture.

**Earth and the Moon are a matched pair**: Lundock observed both on 2008-11-21, same instrument,
same telluric division, both delivered at R = 867. One is 21% O₂. The other is a rock.

## 4. Results (`run_sagan.py`, `analyze.py`)

Reference: INARA test **labels** have mean O₂ = 0.300 (O₂ is heavily over-represented in the
label prior). `causal` predicts mean O₂ = 0.328 on INARA test spectra.

### CH₄ — a genuine detection

Matched resolution: 5 CH₄-rich (Titan, Jupiter, Saturn, Uranus, Neptune) vs 5 airless
(Moon, Enceladus, Dione, Rhea, Ceres), **all R = 829–867**.

| track | CH₄ AUROC (clear) | after declouding |
|---|---|---|
| original1dcnn | **1.00** | 0.16 |
| optimized1dcnn | 0.88 | 0.20 |
| transformerarch | **1.00** | 0.08 |
| causal | **1.00** | 0.00 |
| causal_cfi | **1.00** | 0.20 |
| causal_xl | 0.76 | 0.12 |

CH₄ has deep, broad bands (0.89, 1.7 µm) well inside the catalog's coverage — measured depth
in the delivered albedo: Jupiter 0.47/0.91, Titan 0.35/0.80, Moon −0.09/−0.24. The signal is
there, and the models use it.

### O₂ — fabricated

| track | Earth O₂ | Moon O₂ | Earth − Moon | Earth's rank in O₂ (of 19) |
|---|---|---|---|---|
| original1dcnn | 0.455 | 0.461 | **−0.006** | 3 |
| optimized1dcnn | 0.517 | 0.536 | **−0.019** | 4 |
| transformerarch | 0.392 | 0.465 | **−0.073** | 9 |
| causal | 0.404 | 0.446 | **−0.043** | 6 |
| causal_cfi | 0.312 | 0.305 | +0.007 | 10 |
| causal_xl | 0.311 | 0.313 | **−0.002** | 10 |

True O₂: Earth 0.2095, Moon 0. **Five of six tracks give the Moon more oxygen than Earth.**
`optimized1dcnn` calls all 19 bodies O₂-dominant, Mercury highest of all (O₂+O₃ = 0.597).
Mean O₂+O₃ on airless bodies (0.31–0.47) matches or exceeds that on terrestrials.

Be precise about the mechanism: the O₂ A-band at 0.76 µm is **not measurable in this Earth
spectrum** (depth −0.028; the Moon's is +0.032, i.e. larger — residual telluric). So the models
are not *ignoring* visible evidence. They are **asserting O₂ ≈ 0.4 in its absence**, at their
training prior. The claim supported here is fabrication, not blindness — and the fix is the
label prior, not the architecture.

### All 12 gases — what the catalog can even test (`gases.py`)

A gas is testable here only if some body genuinely imprints it (in the benchmark's own
`covered_species` **and** VMR > 10⁻³) to serve as a positive, scored against the 11 airless
bodies (true VMR = 0) as negatives.

| gas | positives | testable? | median detection AUROC |
|---|---|---|---|
| **CH₄** | Titan + 4 giants (5, all R≥829) | **YES** | **1.00** |
| O₂ | Earth only | rank, not AUROC | 0.68 (Moon > Earth) |
| H₂O | Earth only | rank, not AUROC | 0.41 |
| CO₂ | Venus + Mars (both R≈22) | **no** — unusable resolution | 0.48 |
| N₂ | none (invisible in reflection) | null control | 0.57 |
| O₃, SO₂, CO, NH₃, N₂O, C₂H₆, NO₂ | **none** | **no body imprints them** | — |

**Seven of the twelve target gases cannot be tested by this catalog at all** — no solar-system
body in it carries a detectable amount. **CH₄ is the only gas with a real multi-body,
matched-resolution test, and it passes.** Everything else is either one body (a rank, not a
detection curve), fatally low resolution (CO₂), or absent.

**N₂ is the null control.** Earth is 78% N₂ and Titan 95%, but N₂ has no reflected-light bands,
so it appears in no `covered_species` list — predicted N₂ *must* be uninformative. Median AUROC
0.57 (chance) confirms it, and the models still emit N₂ ≈ 0.26–0.44 on airless rocks: the prior.

**Fabrication is prior-recitation, gas by gas.** Mean predicted VMR on the 11 airless bodies,
against the INARA label prior:

| | O₂ | CO₂ | N₂ | H₂O | CH₄ |
|---|---|---|---|---|---|
| airless-body mean (causal) | 0.341 | 0.351 | 0.255 | 0.024 | 0.008 |
| INARA label prior | 0.300 | 0.305 | 0.301 | 0.034 | 0.034 |

The airless-body outputs track the prior column for column. The models are not measuring these
rocks; they are reciting what INARA taught them the average planet looks like. The trace gases
(O₃, SO₂, CO ≈ 0.00) are correctly near zero — but so is their prior, so that is not a detection
either.

**Surface-ice confound — checked, and clean.** Ice absorbs at 1.5/2.0 µm exactly where water
*vapour* does; 6 of the airless moons are water-ice covered. If the models mistook ice for
vapour, the icy moons would score high H₂O vs the dry rocks. They don't (AUROC ≈ 0.0–0.6,
mostly ≤ chance), and predicted H₂O is ~0.02–0.05 everywhere. Same for Io (SO₂ frost) and Pluto
(CH₄ ice): no phase confusion. The models simply don't respond to these surfaces at all — which
is the *right* answer for an atmospheric retrieval, reached for the wrong reason (prior, not
signal).

### Giants — honesty probe

All four giants are called O₂- or CO₂-dominant with max VMR 0.31–0.46. `optimized1dcnn` puts
O₂+O₃ ≈ 0.44 on every giant. The existing "O₂+O₃ < 0.1 = OK" honesty threshold in
`suites/solar_system.py` **fails for every track on real spectra.**

## 5. The declouder (`decloud_check.py`)

`cloud_recovery/DecloudUNet1D`, 1.41 M params, `decloud_seed0.pth`. Wiring is identical to
`cloud_recovery/eval.py:132` (verified before drawing any conclusion).

**It works on the family it was trained for**, and fails everywhere else. Grey-opacity sweep on
INARA test spectra (`causal`, N=1000; clear R² = +0.782):

| f | in training range? | R² cloudy | R² declouded | verdict |
|---|---|---|---|---|
| 0.0 (clear) | **no** | +0.782 | **−1.089** | HURTS |
| 0.1 | **no** | +0.256 | −0.540 | HURTS |
| 0.2 | **no** | −0.120 | −0.331 | HURTS |
| 0.3 | yes | −0.375 | −0.238 | helps |
| 0.5 | yes | −0.708 | −0.148 | helps |
| 0.7 | yes | −0.888 | −0.101 | helps (recovery +0.47) |
| 0.9 | yes | −0.979 | −0.069 | helps (+0.52) |

The cliff falls **exactly at the training floor**. `cloud_pairs.py:39` sets
`GREY_F_RANGE = (0.3, 1.0)`: opacity below 0.3 — including a clear spectrum — was never sampled.
The network therefore learned an *unconditional* de-muting correction. The README's "starts as
the identity" holds only at initialisation; after training the identity is nowhere in its
hypothesis space.

On the real bodies it **inverts genuine absorption** and **fabricates absent features**
(encoded band contrast, before → after):

| body | CH₄ 0.89 | CH₄ 1.7 | O₂ A |
|---|---|---|---|
| Jupiter | +0.133 → −0.064 | +0.534 → −0.116 | +0.059 → −0.033 |
| Titan | +0.043 → −0.047 | +0.264 → −0.355 | +0.042 → −0.051 |
| Enceladus (pure ice) | −0.006 → +0.012 | −0.639 → **+1.532** | +0.148 → −0.290 |

That is the mechanism behind CH₄ AUROC 1.00 → 0.00.

⚠️ **A trap worth naming.** Declouding makes `original1dcnn` call Venus *and* Mars CO₂-dominant
— "dominant gas correct" for both, and it would look like the declouder fixed Venus. It didn't:
the same pass calls **all 19 bodies** CO₂-dominant, including the Moon and Jupiter. Only the
airless controls expose it. Without them this would have been reported as a success.

## 6. Repo fix applied

`src/cloud_recovery/{eval,train}.py` still imported `from decloud.*` after the module was
renamed `cloud_recovery` — both files raised `ModuleNotFoundError` on import. Fixed (import
path only, no logic touched).

## 7. Deep analysis — untried methods, frozen models only

Four methods with no precedent anywhere in this repo (verified by grep: no conformal /
occlusion / stacking / TTA / contrastive nulling exists in `src/` or elsewhere). All are
inference-only; no weight was modified.

### 7.1 Continuum-twin nulling (`null_twin.py`) — kills the fabrication

For each body, build a **band-free twin** (its own continuum, via
`cloud_families.estimate_continuum`) and score `pred(body) − pred(twin)`. The twin shares
colour, slope and brightness — bands are the only difference — so the label prior cancels
exactly, per body, with zero training.

| metric | raw pred (causal) | twin-nulled | raw (optimized) | twin-nulled |
|---|---|---|---|---|
| CH₄ AUROC | 1.00 | 0.92 | 0.88 | **1.00** |
| O₂ AUROC (Earth vs airless) | 0.64 | **0.91** | 0.73 | **1.00** |
| Earth − Moon O₂ | −0.043 | +0.006 | −0.019 | +0.009 |
| mean \|O₂\| on rocks | 0.341 | **0.016** | 0.468 | **0.006** |
| mean \|CO₂\| on rocks | 0.351 | 0.013 | 0.186 | 0.007 |
| mean \|N₂\| on rocks | 0.255 | 0.003 | 0.271 | 0.004 |

Fabrication collapses **20–80×**; CH₄ survives; the Earth−Moon O₂ sign flips correct on both
architectures. Caveats kept honest: the O₂ "AUROC" has one positive (a rank), and the nulled
Earth−Moon gap (+0.006–0.009) is tiny — consistent with the A-band not being measurable in this
data. The method converts a prior-reciting retriever into a band-differential detector.

### 7.2 Occlusion evidence check — the CH₄ signal is causal

Blanking the CH₄ windows (0.86–0.92, 1.10–1.20, 1.30–1.45, 1.62–1.80 µm, continuum-filled) drops
CH₄ AUROC 1.00→0.88; blanking equal-width control windows changes nothing (1.00). The model
reads the right wavelengths; the residual 0.88 reflects CH₄'s redundancy across many bands.
A plain **band-depth ruler also scores 1.00**, so the honest claim is "the NN matches physics
on this discrimination", not "beats it".

### 7.3 Resolution requirements (`resolution_curve.py`) — the R-question settled per gas

Per-gas R² vs resolution on INARA test (causal), raw and after a per-gas affine recalibration
(fit once on held-in INARA planets — frozen model):

| R | H₂O | CO₂ | O₂ | CH₄ | O₃ | | affine: H₂O | CO₂ | O₂ | CH₄ | O₃ |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | −0.24 | −1.87 | −0.15 | 0.32 | 0.27 | | 0.36 | 0.11 | 0.33 | 0.34 | **0.80** |
| 100 | 0.55 | −1.82 | 0.53 | 0.64 | 0.67 | | 0.61 | 0.22 | 0.69 | 0.68 | 0.87 |
| 200 | 0.64 | 0.39 | 0.71 | 0.74 | 0.76 | | 0.67 | 0.52 | 0.76 | 0.76 | 0.88 |
| 870 | 0.59 | 0.70 | 0.80 | 0.80 | 0.83 | | 0.70 | 0.70 | 0.79 | 0.82 | 0.89 |

- **CO₂ is the demanding gas: R ≥ 200, and affine CANNOT rescue it at R=22** (0.11) — the
  information is destroyed, not biased. Venus/Mars remain unusable; that is physics.
- Every other gas degrades gracefully and is **mostly recalibratable**: O₃ keeps 0.80 even at
  R=22 (broad features + continuum slope), H₂O/O₂/CH₄ are usable from R≈50 with recalibration.
- Practical rule: quantitative CO₂ needs R≥200; CH₄/O₃ surveys can run at R≈50.

### 7.4 Cross-architecture stacking + conformal (`stack_conformal.py`) — the ceiling falls

Per-gas linear stack (6 coefficients + intercept, log10 space) over all six frozen tracks,
fit on half the INARA test split, scored on the disjoint half:

| combiner | R²_covered |
|---|---|
| best single (original1dcnn) | +0.777 |
| plain 6-track mean | +0.637 (weak tracks drag it) |
| **per-gas linear stack** | **+0.849** (O₃ 0.91, CH₄ 0.89, O₂ 0.86, H₂O 0.81, CO₂ 0.78) |

The audit's "flat architecture ceiling 0.73–0.78" is **not a data ceiling**: the six models make
complementary errors, and +0.07 was recoverable with least-squares. Even the weak CFI tracks
contribute. This is the cheapest accuracy gain available anywhere in the project.

Split-conformal intervals (causal): valid in-distribution (cov68 = 0.67–0.71 vs target 0.68)
but **collapse under resolution shift** (cov68 at R=22: 0.16–0.41). The honest R=22 widths are
2–4× larger (e.g. CO₂ 0.15→0.58 dex). Resolution-conditioned conformal — recomputing quantiles
at the target's R on INARA — restores validity with no retraining, and is what any real-target
claim should quote.

## 8. What to do next

- **Fix the declouder's training distribution**: `GREY_F_RANGE = (0.0, 1.0)` so the identity is
  in-distribution, then re-check `f=0` recovery ≈ 0. Cheap and clearly correct. Until then, do
  not run the declouder on any real spectrum.
- **The O₂ prior is a dataset property, not a model bug.** INARA labels have mean O₂ = 0.300.
  Any biosignature claim from a model trained on this prior needs an airless-body control
  reported alongside it.
- **Adopt the airless bodies as a standing eval section.** They are free, real, and they are the
  only thing here that can falsify a biosignature detection.
- **Tighten the honesty threshold.** `suites/solar_system.py` uses O₂+O₃ < 0.1; no track meets it
  on real data. Either the threshold or the models must change.
- Report CH₄ AUROC = 1.00 as the headline real-data result. It is defensible, matched-resolution,
  and reproducible from `sagan_results.json`.
