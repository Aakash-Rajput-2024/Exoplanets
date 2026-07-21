# Evaluation report — `original1dcnn`

*generated 2026-07-09T22:23:16*  ·  seeds [0]  ·  device `mps`  ·  ran 10/10 epochs  ·  git `aad036b02b15`

> Reflected-light / direct-imaging retrieval (0.2–2.0 µm, LUVOIR-like). Each section states its epistemic status. The only LITERAL-ground-truth real tests are Sections E (solar-system) and F (real Earth); transiting-planet data (G, and 'far' rows of H) is a wrong-observable OOD probe, not an accuracy measurement.

## Contents

- **A** In-distribution (INARA held-out test) — 🟡 preliminary
- **B** Classical baselines (PriorMean / Ridge / RandomForest) — ✅ ok
- **C** Cross-generator (pRT / TauREx / MultiREx) — ✅ ok
- **D** PSG sanity anchor (eval-path control) — ✅ ok
- **E** Solar-System-as-exoplanet (known composition) — ✅ ok
- **F** Real disk-integrated Earth (VPL Robinson 2011) — ✅ ok
- **G** Transiting-planet OOD probe — ✅ ok
- **H** Published-retrieval comparison (benchmark exoplanets) — ✅ ok
- **I** Posterior calibration (SBC / TARP / PIT / ECE) — 🟡 preliminary
- **J** OOD honesty (δ/v, raw vs debiased R²) — ✅ ok

## Section A — In-distribution (INARA held-out test)  🟡
*Ground truth: exact (synthetic). The information ceiling of the observable.*

| metric | value |
|---|---|
| R2_covered @a=1 | 0.1241 |
| R2_all12 @a=1 | 0.0541 |
| R2_covered noiseless-ref (a=300) | 0.7787 |
| R2_all12 noiseless-ref | 0.5501 |
| RMSE_all12 (dex) @a=1 | 0.4458 |
| n_seeds | 1 |
| per_seed_R2_all12_mean±std | [0.0541, None] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | 0.0584 | [0.0409, 0.0770] | 0.4385 | 0.2991 |
| CO2 | 0.0252 | [0.0091, 0.0428] | 0.3624 | 0.2346 |
| O2 | 0.1094 | [0.0884, 0.1313] | 0.3652 | 0.2331 |
| N2 | -0.0604 | [-0.0690, -0.0520] | 0.3911 | 0.2490 |
| CH4 | 0.1046 | [0.0854, 0.1243] | 0.4216 | 0.2932 |
| N2O | -0.0432 | [-0.0534, -0.0333] | 0.4536 | 0.3133 |
| CO | -0.0592 | [-0.0683, -0.0508] | 0.4739 | 0.3259 |
| O3 | 0.3228 | [0.3029, 0.3429] | 0.4820 | 0.3462 |
| SO2 | -0.0492 | [-0.0575, -0.0409] | 0.4699 | 0.3257 |
| NH3 | 0.2402 | [0.2151, 0.2631] | 0.4067 | 0.2772 |
| C2H6 | 0.0481 | [0.0338, 0.0613] | 0.6132 | 0.4453 |
| NO2 | -0.0472 | [-0.0565, -0.0371] | 0.4717 | 0.3247 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0249 | -0.0376 |
| 1.0000 | 1.0000 | 2.9950 | 0.1241 | 0.0541 |
| 3.0000 | 9.0000 | 8.9849 | 0.3432 | 0.1916 |
| 10.0000 | 100.0000 | 29.9498 | 0.5513 | 0.3440 |
| 30.0000 | 900.0000 | 89.8493 | 0.6857 | 0.4595 |
| 100.0000 | 10000.0000 | 299.4975 | 0.7795 | 0.5425 |
| 300.0000 | 90000.0000 | 898.4926 | 0.8219 | 0.5770 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.
- PRELIMINARY: n<3 seeds — no across-seed variance; CIs are test-planet bootstrap only.

<sub>artifacts: `original1dcnn/in_distribution/r2_vs_snr.png`, `original1dcnn/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0541 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | True |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: original1dcnn (α=1) | 0.0541 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.

<sub>artifacts: `original1dcnn/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.7787 |
| prt_R2_covered_ref | -5.5619 |
| taurex_R2_covered_ref | -0.3737 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.7787 | - | 0.0000 |
| prt | 2000 | -5.5619 | -1.8355 | -6.3407 |
| taurex | 2000 | -0.3737 | -0.1069 | -1.1524 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `original1dcnn/cross_generator/r2_vs_snr_prt.png`, `original1dcnn/cross_generator/r2_vs_snr_taurex.png`, `original1dcnn/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  ✅
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.7787 |
| anchor_R2_covered_ref | 0.7571 |
| anchor/native | 0.9722 |
| PASS(≥0.9×) | True |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.6750 | [0.6316, 0.7105] |
| CO2 | 0.6059 | [0.5422, 0.6625] |
| O2 | 0.8171 | [0.7960, 0.8383] |
| N2 | 0.0156 | [-0.0144, 0.0473] |
| CH4 | 0.8014 | [0.7756, 0.8245] |
| N2O | 0.4656 | [0.4225, 0.5067] |
| CO | 0.0143 | [-0.0107, 0.0394] |
| O3 | 0.8860 | [0.8663, 0.9027] |
| SO2 | 0.1070 | [0.0636, 0.1486] |
| NH3 | 0.8414 | [0.8196, 0.8602] |
| C2H6 | 0.4435 | [0.3957, 0.4902] |
| NO2 | 0.6521 | [0.5885, 0.7072] |

<sub>artifacts: `original1dcnn/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 1/4 |
| mean dex-err covered (terrestrial) | 1.5578 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | O2 | ✗ | 2.2400 | 0.9100 |
| Mars | CO2 | O2 | ✗ | 1.1800 | 0.7900 |
| Venus | CO2 | CO2 | ✓ | 2.0200 | 0.6200 |
| Titan | N2 | O2 | ✗ | 0.7800 | 0.2700 |
| Jupiter (giant/honesty) | CH4 | O2 | ✗ | 1.6000 | -0.0800 |
| Saturn (giant/honesty) | CH4 | O2 | ✗ | 1.2700 | -0.0800 |
| Uranus (giant/honesty) | CH4 | O2 | ✗ | 0.4700 | 0.2600 |
| Neptune (giant/honesty) | CH4 | O2 | ✗ | 0.6600 | 0.2600 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.4506 | ⚠ fabricated O2/O3 |
| Saturn | 0.4583 | ⚠ fabricated O2/O3 |
| Uranus | 0.4623 | ⚠ fabricated O2/O3 |
| Neptune | 0.4621 | ⚠ fabricated O2/O3 |

**Caveats**
- Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this tests whether the model responds to which gases are present, not absolute fidelity. For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.
- Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, not an accuracy test.
- INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like ppb trace gases are out-of-distribution; read the MAJOR/covered gases.

<sub>artifacts: `original1dcnn/solar_system/truth_Earth.png`, `original1dcnn/solar_system/truth_Mars.png`, `original1dcnn/solar_system/truth_Venus.png`, `original1dcnn/solar_system/truth_Titan.png`, `original1dcnn/solar_system/truth_Jupiter.png`, `original1dcnn/solar_system/truth_Saturn.png`, `original1dcnn/solar_system/truth_Uranus.png`, `original1dcnn/solar_system/truth_Neptune.png`, `original1dcnn/solar_system/result.json`</sub>

## Section F — Real disk-integrated Earth (VPL Robinson 2011)  ✅
*Ground truth: LITERAL, on REAL photons. The strongest single real test.*

| metric | value |
|---|---|
| dominant true | N2 |
| dominant pred | O2 |
| dominant correct | False |
| mean dex-err (covered) | 2.0776 |
| ordering ρ | 0.8791 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 4.92e-02 | 3.47e-02 | yes |
| CO2 | 4.20e-04 | 1.68e-01 | 2.86e-01 | yes |
| O2 | 2.09e-01 | 4.54e-01 | 3.14e-01 | yes |
| N2 | 7.81e-01 | 3.01e-01 | 3.10e-01 | - |
| CH4 | 1.90e-06 | 6.35e-03 | 3.06e-02 | yes |
| N2O | 3.30e-07 | 6.96e-03 | 6.63e-03 | - |
| CO | 1.20e-07 | 6.85e-03 | 6.52e-03 | - |
| O3 | 7.00e-07 | 1.20e-03 | 1.10e-03 | yes |
| SO2 | 1.00e-12 | 6.23e-03 | 6.52e-03 | - |
| NH3 | 1.00e-12 | 9.02e-04 | 3.67e-03 | - |
| C2H6 | 1.00e-12 | 2.40e-05 | 1.18e-04 | - |
| NO2 | 1.00e-12 | 3.79e-06 | 6.60e-06 | - |

**Caveats**
- REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and median-matched to INARA before the INARA-fit norm — the observable/units are adapted, not identical to INARA.
- INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases (N2, O2, H2O, CO2).
- Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into data/real_earth/ with the same loader.

<sub>artifacts: `original1dcnn/real_earth/truth_bars_Earth-VPL.png`, `original1dcnn/real_earth/result.json`</sub>

## Section G — Transiting-planet OOD probe  ✅
*NO ground truth. Wrong observable — anomaly/stability only, never accuracy.*

| metric | value |
|---|---|
| INARA anomaly median | 0.8058 |
| INARA anomaly p95 | 1.8558 |
| real spectra found | 0 |

**OOD probe — anomaly vs INARA training distribution**

| input | anomaly (RMS z) | × INARA median | pred stability (std) |
|---|---|---|---|
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0047 |

**Caveats**
- ⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light direct-imaging model — any predicted composition is meaningless. The point is to SHOW the inputs are far off-distribution (large anomaly) and/or unstable.
- A high anomaly × INARA-median is the expected, desired result: the model correctly sees a transiting spectrum as out-of-distribution.
- To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).

<sub>artifacts: `original1dcnn/transiting_ood/result.json`</sub>

## Section H — Published-retrieval comparison (benchmark exoplanets)  ✅
*Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy.*

| metric | value |
|---|---|
| n_targets | 6 |
| n_near-domain (direct-imaging) | 2 |
| mean dex-err (near-domain) | 2.1113 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.7400 | 0.1000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.5800 | -0.8700 |
| WASP-96b | far | transmission | H2O,CO2 | 2.8000 | -1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.5400 | 0.5000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.5000 | -0.5000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.7200 | -0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `original1dcnn/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  🟡
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 20 |
| coverage_68 (active) | 0.2456 |
| coverage_95 (active) | 0.4134 |
| PIT-KS (active mean) | 0.3790 |
| reliability_ECE | 0.3374 |
| TARP_ECE | 0.3385 |
| posterior_spread(dex) | 0.0497 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.1780 | 0.3220 | 0.5760 | 2618.2000 |
| CO2 | 0.1800 | 0.3060 | 0.4360 | 2379.5000 |
| O2 | 0.2720 | 0.4320 | 0.2760 | 1277.5000 |
| N2 | 0.0880 | 0.1600 | 0.4140 | 2986.4000 |
| CH4 | 0.2900 | 0.4940 | 0.3800 | 1140.9000 |
| N2O | 0.2320 | 0.3840 | 0.3380 | 1615.0000 |
| CO | 0.0340 | 0.0660 | 0.4700 | 3829.5000 |
| O3 | 0.4160 | 0.6560 | 0.2080 | 377.3000 |
| SO2 | 0.1240 | 0.2640 | 0.4580 | 2450.8000 |
| NH3 | 0.3200 | 0.5480 | 0.3460 | 879.4000 |
| C2H6 | 0.1140 | 0.2020 | 0.4940 | 2876.8000 |
| NO2 | 0.3300 | 0.5260 | 0.2780 | 811.3000 |

**Caveats**
- Posterior = 20 samples (MC-dropout T=20 × 1 seed(s); dropout layers active=1); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).
- PRELIMINARY: n<3 seeds — limited deep-ensemble diversity.

<sub>artifacts: `original1dcnn/calibration/reliability.png`, `original1dcnn/calibration/tarp_coverage.png`, `original1dcnn/calibration/sbc_ranks.png`, `original1dcnn/calibration/result.json`</sub>

## Section J — OOD honesty (δ/v, raw vs debiased R²)  ✅
*Decomposes cross-gen R² into input shift + bias vs genuine information loss.*

| metric | value |
|---|---|
| prt_verdict | no transfer (genuine) |
| taurex_verdict | no transfer (genuine) |
| psg_verdict | biased extrapolation (fixable) |

**OOD decomposition (covered species)**

| engine | med|δ| | med v | frac λ in-family | R²_raw | R²_debiased | r²_pearson | verdict |
|---|---|---|---|---|---|---|---|
| prt | 1.6300 | 0.1600 | 0.0000 | -5.5620 | -1.1800 | 0.0020 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -0.3740 | -0.0900 | 0.0010 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.7570 | 0.7940 | 0.8040 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `original1dcnn/ood_honesty/ood_delta_prt.png`, `original1dcnn/ood_honesty/ood_delta_taurex.png`, `original1dcnn/ood_honesty/ood_delta_psg.png`, `original1dcnn/ood_honesty/result.json`</sub>

