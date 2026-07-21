# Evaluation report — `causal_xl_cfi`

*generated 2026-07-09T23:27:37*  ·  seeds [0, 1, 2]  ·  device `mps`  ·  ran 48/50 epochs  ·  git `unknown`

> Reflected-light / direct-imaging retrieval (0.2–2.0 µm, LUVOIR-like). Each section states its epistemic status. The only LITERAL-ground-truth real tests are Sections E (solar-system) and F (real Earth); transiting-planet data (G, and 'far' rows of H) is a wrong-observable OOD probe, not an accuracy measurement.

## Contents

- **A** In-distribution (INARA held-out test) — ✅ ok
- **B** Classical baselines (PriorMean / Ridge / RandomForest) — ✅ ok
- **C** Cross-generator (pRT / TauREx / MultiREx) — ✅ ok
- **D** PSG sanity anchor (eval-path control) — ✅ ok
- **E** Solar-System-as-exoplanet (known composition) — ✅ ok
- **F** Real disk-integrated Earth (VPL Robinson 2011) — ✅ ok
- **G** Transiting-planet OOD probe — ✅ ok
- **H** Published-retrieval comparison (benchmark exoplanets) — ✅ ok
- **I** Posterior calibration (SBC / TARP / PIT / ECE) — ✅ ok
- **J** OOD honesty (δ/v, raw vs debiased R²) — ✅ ok

## Section A — In-distribution (INARA held-out test)  ✅
*Ground truth: exact (synthetic). The information ceiling of the observable.*

| metric | value |
|---|---|
| R2_covered @a=1 | 0.0537 |
| R2_all12 @a=1 | 0.0113 |
| R2_covered noiseless-ref (a=300) | 0.1735 |
| R2_all12 noiseless-ref | 0.0840 |
| RMSE_all12 (dex) @a=1 | 0.4568 |
| n_seeds | 3 |
| per_seed_R2_all12_mean±std | [0.0111, 0.0010] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | 0.0102 | [1.681e-04, 0.0204] | 0.4496 | 0.3050 |
| CO2 | -0.0212 | [-0.0307, -0.0112] | 0.3710 | 0.2377 |
| O2 | 0.0251 | [0.0134, 0.0363] | 0.3821 | 0.2398 |
| N2 | -0.0813 | [-0.0898, -0.0732] | 0.3949 | 0.2486 |
| CH4 | 0.0521 | [0.0403, 0.0645] | 0.4338 | 0.3020 |
| N2O | -0.0476 | [-0.0557, -0.0400] | 0.4545 | 0.3138 |
| CO | -0.0668 | [-0.0756, -0.0584] | 0.4756 | 0.3259 |
| O3 | 0.2023 | [0.1890, 0.2158] | 0.5232 | 0.3787 |
| SO2 | -0.0549 | [-0.0629, -0.0471] | 0.4712 | 0.3255 |
| NH3 | 0.1548 | [0.1395, 0.1695] | 0.4290 | 0.2939 |
| C2H6 | 0.0170 | [0.0078, 0.0258] | 0.6232 | 0.4538 |
| NO2 | -0.0534 | [-0.0609, -0.0455] | 0.4731 | 0.3250 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0272 | -0.0394 |
| 1.0000 | 1.0000 | 2.9950 | 0.0537 | 0.0113 |
| 3.0000 | 9.0000 | 8.9849 | 0.1647 | 0.0788 |
| 10.0000 | 100.0000 | 29.9498 | 0.2580 | 0.1387 |
| 30.0000 | 900.0000 | 89.8493 | 0.3073 | 0.1764 |
| 100.0000 | 10000.0000 | 299.4975 | 0.3357 | 0.2020 |
| 300.0000 | 90000.0000 | 898.4926 | 0.3433 | 0.2087 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.

<sub>artifacts: `causal_xl_cfi/in_distribution/r2_vs_snr.png`, `causal_xl_cfi/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0113 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | False |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: causal_xl_cfi (α=1) | 0.0113 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.
- ⚠ This track's α=1 R² does NOT exceed the Ridge floor — at the photon floor that is expected; compare at the noiseless ref / α* before concluding.

<sub>artifacts: `causal_xl_cfi/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.1735 |
| prt_R2_covered_ref | -0.1010 |
| taurex_R2_covered_ref | -0.0590 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.1735 | - | 0.0000 |
| prt | 2000 | -0.1010 | -0.1132 | -0.2745 |
| taurex | 2000 | -0.0590 | -0.0775 | -0.2325 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `causal_xl_cfi/cross_generator/r2_vs_snr_prt.png`, `causal_xl_cfi/cross_generator/r2_vs_snr_taurex.png`, `causal_xl_cfi/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  ✅
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.1735 |
| anchor_R2_covered_ref | 0.1739 |
| anchor/native | 1.0023 |
| PASS(≥0.9×) | True |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.1086 | [0.0889, 0.1294] |
| CO2 | 0.0794 | [0.0571, 0.1009] |
| O2 | 0.1071 | [0.0864, 0.1283] |
| N2 | -0.0919 | [-0.1167, -0.0702] |
| CH4 | 0.1894 | [0.1695, 0.2112] |
| N2O | -0.0511 | [-0.0722, -0.0304] |
| CO | -0.0759 | [-0.0992, -0.0540] |
| O3 | 0.3848 | [0.3667, 0.4021] |
| SO2 | -0.0512 | [-0.0727, -0.0313] |
| NH3 | 0.1991 | [0.1762, 0.2197] |
| C2H6 | 0.0990 | [0.0798, 0.1176] |
| NO2 | 0.0691 | [0.0484, 0.0907] |

<sub>artifacts: `causal_xl_cfi/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 0/4 |
| mean dex-err covered (terrestrial) | 1.4994 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | O2 | ✗ | 2.1800 | 0.8500 |
| Mars | CO2 | O2 | ✗ | 1.1800 | 0.8200 |
| Venus | CO2 | O2 | ✗ | 1.9900 | 0.5500 |
| Titan | N2 | O2 | ✗ | 0.6400 | 0.2500 |
| Jupiter (giant/honesty) | CH4 | O2 | ✗ | 1.4700 | -0.1000 |
| Saturn (giant/honesty) | CH4 | O2 | ✗ | 1.1500 | -0.1000 |
| Uranus (giant/honesty) | CH4 | O2 | ✗ | 0.1100 | 0.2500 |
| Neptune (giant/honesty) | CH4 | O2 | ✗ | 0.3000 | 0.2500 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.3295 | ⚠ fabricated O2/O3 |
| Saturn | 0.3289 | ⚠ fabricated O2/O3 |
| Uranus | 0.3256 | ⚠ fabricated O2/O3 |
| Neptune | 0.3256 | ⚠ fabricated O2/O3 |

**Caveats**
- Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this tests whether the model responds to which gases are present, not absolute fidelity. For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.
- Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, not an accuracy test.
- INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like ppb trace gases are out-of-distribution; read the MAJOR/covered gases.

<sub>artifacts: `causal_xl_cfi/solar_system/truth_Earth.png`, `causal_xl_cfi/solar_system/truth_Mars.png`, `causal_xl_cfi/solar_system/truth_Venus.png`, `causal_xl_cfi/solar_system/truth_Titan.png`, `causal_xl_cfi/solar_system/truth_Jupiter.png`, `causal_xl_cfi/solar_system/truth_Saturn.png`, `causal_xl_cfi/solar_system/truth_Uranus.png`, `causal_xl_cfi/solar_system/truth_Neptune.png`, `causal_xl_cfi/solar_system/result.json`</sub>

## Section F — Real disk-integrated Earth (VPL Robinson 2011)  ✅
*Ground truth: LITERAL, on REAL photons. The strongest single real test.*

| metric | value |
|---|---|
| dominant true | N2 |
| dominant pred | O2 |
| dominant correct | False |
| mean dex-err (covered) | 2.2049 |
| ordering ρ | 0.8507 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 3.74e-02 | 3.28e-02 | yes |
| CO2 | 4.20e-04 | 2.60e-01 | 2.97e-01 | yes |
| O2 | 2.09e-01 | 3.41e-01 | 3.12e-01 | yes |
| N2 | 7.81e-01 | 3.07e-01 | 3.05e-01 | - |
| CH4 | 1.90e-06 | 3.04e-02 | 3.06e-02 | yes |
| N2O | 3.30e-07 | 6.44e-03 | 6.35e-03 | - |
| CO | 1.20e-07 | 6.61e-03 | 6.35e-03 | - |
| O3 | 7.00e-07 | 1.23e-03 | 1.16e-03 | yes |
| SO2 | 1.00e-12 | 6.60e-03 | 6.32e-03 | - |
| NH3 | 1.00e-12 | 3.48e-03 | 3.27e-03 | - |
| C2H6 | 1.00e-12 | 1.18e-04 | 1.20e-04 | - |
| NO2 | 1.00e-12 | 6.04e-06 | 6.31e-06 | - |

**Caveats**
- REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and median-matched to INARA before the INARA-fit norm — the observable/units are adapted, not identical to INARA.
- INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases (N2, O2, H2O, CO2).
- Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into data/real_earth/ with the same loader.

<sub>artifacts: `causal_xl_cfi/real_earth/truth_bars_Earth-VPL.png`, `causal_xl_cfi/real_earth/result.json`</sub>

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0015 |

**Caveats**
- ⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light direct-imaging model — any predicted composition is meaningless. The point is to SHOW the inputs are far off-distribution (large anomaly) and/or unstable.
- A high anomaly × INARA-median is the expected, desired result: the model correctly sees a transiting spectrum as out-of-distribution.
- To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).

<sub>artifacts: `causal_xl_cfi/transiting_ood/result.json`</sub>

## Section H — Published-retrieval comparison (benchmark exoplanets)  ✅
*Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy.*

| metric | value |
|---|---|
| n_targets | 6 |
| n_near-domain (direct-imaging) | 2 |
| mean dex-err (near-domain) | 2.0010 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.7100 | 0.1000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.4300 | 0.0000 |
| WASP-96b | far | transmission | H2O,CO2 | 2.7300 | -1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.4700 | 0.5000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.4000 | -1.0000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.6000 | 0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `causal_xl_cfi/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  ✅
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 60 |
| coverage_68 (active) | 0.0764 |
| coverage_95 (active) | 0.1092 |
| PIT-KS (active mean) | 0.4626 |
| reliability_ECE | 0.4499 |
| TARP_ECE | 0.4524 |
| posterior_spread(dex) | 0.0206 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.0920 | 0.1240 | 0.4580 | 3422.6000 |
| CO2 | 0.0700 | 0.1040 | 0.4860 | 3618.6000 |
| O2 | 0.1140 | 0.1700 | 0.4260 | 3074.4000 |
| N2 | 0.0360 | 0.0620 | 0.5000 | 4022.3000 |
| CH4 | 0.0520 | 0.0860 | 0.4540 | 3774.1000 |
| N2O | 0.0320 | 0.0540 | 0.4760 | 4060.1000 |
| CO | 0.0360 | 0.0480 | 0.5180 | 4131.9000 |
| O3 | 0.1080 | 0.1520 | 0.4180 | 3201.2000 |
| SO2 | 0.0360 | 0.0440 | 0.4940 | 4117.8000 |
| NH3 | 0.1120 | 0.1480 | 0.4540 | 3208.9000 |
| C2H6 | 0.0700 | 0.1060 | 0.4580 | 3577.6000 |
| NO2 | 0.0780 | 0.1040 | 0.5020 | 3645.3000 |

**Caveats**
- Posterior = 60 samples (MC-dropout T=20 × 3 seed(s); dropout layers active=13); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).

<sub>artifacts: `causal_xl_cfi/calibration/reliability.png`, `causal_xl_cfi/calibration/tarp_coverage.png`, `causal_xl_cfi/calibration/sbc_ranks.png`, `causal_xl_cfi/calibration/result.json`</sub>

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
| prt | 1.6300 | 0.1600 | 0.0000 | -0.1010 | -0.1120 | 0.0010 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -0.0590 | -0.0700 | 0.0000 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.1740 | 0.1830 | 0.5290 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `causal_xl_cfi/ood_honesty/ood_delta_prt.png`, `causal_xl_cfi/ood_honesty/ood_delta_taurex.png`, `causal_xl_cfi/ood_honesty/ood_delta_psg.png`, `causal_xl_cfi/ood_honesty/result.json`</sub>

