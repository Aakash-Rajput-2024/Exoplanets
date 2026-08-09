# Evaluation report — `original1dcnn`

*generated 2026-08-09T15:23:41*  ·  seeds [0, 1, 2]  ·  device `mps`  ·  ran 50/50 epochs  ·  git `3d7318bb26fb`

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
- **K** Bayesian reference retrieval (information ceiling) — ⏭️ skipped

## Section A — In-distribution (INARA held-out test)  ✅
*Ground truth: exact (synthetic). The information ceiling of the observable.*

| metric | value |
|---|---|
| R2_covered @a=1 | 0.1370 |
| R2_all12 @a=1 | 0.0626 |
| R2_covered noiseless-ref (a=300) | 0.8475 |
| R2_all12 noiseless-ref | 0.5919 |
| RMSE_all12 (dex) @a=1 | 0.4437 |
| n_seeds | 3 |
| per_seed_R2_all12_mean±std | [0.0615, 3.731e-04] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | 0.0700 | [0.0528, 0.0889] | 0.4358 | 0.2977 |
| CO2 | 0.0357 | [0.0192, 0.0528] | 0.3605 | 0.2331 |
| O2 | 0.1243 | [0.1028, 0.1471] | 0.3621 | 0.2313 |
| N2 | -0.0619 | [-0.0708, -0.0533] | 0.3913 | 0.2485 |
| CH4 | 0.1175 | [0.0976, 0.1367] | 0.4186 | 0.2908 |
| N2O | -0.0356 | [-0.0456, -0.0258] | 0.4519 | 0.3129 |
| CO | -0.0587 | [-0.0678, -0.0501] | 0.4738 | 0.3258 |
| O3 | 0.3375 | [0.3169, 0.3576] | 0.4768 | 0.3410 |
| SO2 | -0.0468 | [-0.0551, -0.0389] | 0.4694 | 0.3253 |
| NH3 | 0.2531 | [0.2289, 0.2757] | 0.4032 | 0.2749 |
| C2H6 | 0.0539 | [0.0398, 0.0672] | 0.6114 | 0.4439 |
| NO2 | -0.0373 | [-0.0465, -0.0269] | 0.4695 | 0.3236 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0164 | -0.0317 |
| 1.0000 | 1.0000 | 2.9950 | 0.1370 | 0.0626 |
| 3.0000 | 9.0000 | 8.9849 | 0.3573 | 0.2026 |
| 10.0000 | 100.0000 | 29.9498 | 0.5706 | 0.3603 |
| 30.0000 | 900.0000 | 89.8493 | 0.7099 | 0.4798 |
| 100.0000 | 10000.0000 | 299.4975 | 0.8059 | 0.5657 |
| 300.0000 | 90000.0000 | 898.4926 | 0.8502 | 0.6056 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.

<sub>artifacts: `original1dcnn/in_distribution/r2_vs_snr.png`, `original1dcnn/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0626 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | True |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: original1dcnn (α=1) | 0.0626 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.

<sub>artifacts: `original1dcnn/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.8475 |
| prt_R2_covered_ref | -6.8269 |
| taurex_R2_covered_ref | -0.6699 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.8475 | - | 0.0000 |
| prt | 2000 | -6.8269 | -2.0873 | -7.6745 |
| taurex | 2000 | -0.6699 | -0.1035 | -1.5174 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `original1dcnn/cross_generator/r2_vs_snr_prt.png`, `original1dcnn/cross_generator/r2_vs_snr_taurex.png`, `original1dcnn/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  ✅
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.8475 |
| anchor_R2_covered_ref | 0.8251 |
| anchor/native | 0.9735 |
| PASS(≥0.9×) | True |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.7927 | [0.7650, 0.8159] |
| CO2 | 0.7046 | [0.6422, 0.7623] |
| O2 | 0.8453 | [0.8227, 0.8656] |
| N2 | -0.0194 | [-0.0529, 0.0160] |
| CH4 | 0.8747 | [0.8590, 0.8892] |
| N2O | 0.5095 | [0.4675, 0.5520] |
| CO | -0.0131 | [-0.0429, 0.0162] |
| O3 | 0.9080 | [0.8942, 0.9210] |
| SO2 | 0.1225 | [0.0607, 0.1789] |
| NH3 | 0.8651 | [0.8446, 0.8827] |
| C2H6 | 0.4781 | [0.4343, 0.5258] |
| NO2 | 0.7612 | [0.7253, 0.7929] |

<sub>artifacts: `original1dcnn/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 0/4 |
| mean dex-err covered (terrestrial) | 1.6095 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | O2 | ✗ | 2.2400 | 0.9100 |
| Mars | CO2 | O2 | ✗ | 1.2500 | 0.7800 |
| Venus | CO2 | N2 | ✗ | 2.1600 | 0.6200 |
| Titan | N2 | O2 | ✗ | 0.7900 | 0.2700 |
| Jupiter (giant/honesty) | CH4 | O2 | ✗ | 1.6900 | -0.0800 |
| Saturn (giant/honesty) | CH4 | O2 | ✗ | 1.3700 | -0.0800 |
| Uranus (giant/honesty) | CH4 | O2 | ✗ | 0.4600 | 0.2600 |
| Neptune (giant/honesty) | CH4 | O2 | ✗ | 0.6500 | 0.2600 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.4319 | ⚠ fabricated O2/O3 |
| Saturn | 0.4398 | ⚠ fabricated O2/O3 |
| Uranus | 0.4572 | ⚠ fabricated O2/O3 |
| Neptune | 0.4573 | ⚠ fabricated O2/O3 |

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
| mean dex-err (covered) | 2.0604 |
| ordering ρ | 0.8435 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 5.39e-02 | 3.38e-02 | yes |
| CO2 | 4.20e-04 | 1.16e-01 | 2.98e-01 | yes |
| O2 | 2.09e-01 | 4.72e-01 | 3.11e-01 | yes |
| N2 | 7.81e-01 | 3.28e-01 | 3.02e-01 | - |
| CH4 | 1.90e-06 | 6.79e-03 | 3.18e-02 | yes |
| N2O | 3.30e-07 | 7.24e-03 | 6.35e-03 | - |
| CO | 1.20e-07 | 6.75e-03 | 6.28e-03 | - |
| O3 | 7.00e-07 | 1.17e-03 | 1.05e-03 | yes |
| SO2 | 1.00e-12 | 6.86e-03 | 6.25e-03 | - |
| NH3 | 1.00e-12 | 1.13e-03 | 3.49e-03 | - |
| C2H6 | 1.00e-12 | 2.71e-05 | 1.22e-04 | - |
| NO2 | 1.00e-12 | 4.74e-06 | 6.21e-06 | - |

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0032 |

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
| mean dex-err (near-domain) | 2.1077 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.8100 | -0.2000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.6100 | 0.0000 |
| WASP-96b | far | transmission | H2O,CO2 | 2.7500 | -1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.5400 | 0.5000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.4800 | -0.5000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.7300 | -0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `original1dcnn/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  ✅
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 90 |
| coverage_68 (active) | 0.3753 |
| coverage_95 (active) | 0.5872 |
| PIT-KS (active mean) | 0.2820 |
| reliability_ECE | 0.2504 |
| TARP_ECE | 0.3001 |
| posterior_spread(dex) | 0.0660 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.3320 | 0.5640 | 0.3330 | 2080.3000 |
| CO2 | 0.3780 | 0.5980 | 0.2040 | 1355.2000 |
| O2 | 0.4030 | 0.6460 | 0.2650 | 1184.7000 |
| N2 | 0.1970 | 0.3340 | 0.3880 | 4060.7000 |
| CH4 | 0.4060 | 0.6550 | 0.2600 | 1184.8000 |
| N2O | 0.3670 | 0.5610 | 0.3220 | 2113.4000 |
| CO | 0.1120 | 0.1760 | 0.3960 | 6008.6000 |
| O3 | 0.4630 | 0.7450 | 0.1670 | 497.8000 |
| SO2 | 0.3450 | 0.4540 | 0.3200 | 2625.3000 |
| NH3 | 0.4430 | 0.7160 | 0.2590 | 978.2000 |
| C2H6 | 0.1230 | 0.2270 | 0.4750 | 5782.2000 |
| NO2 | 0.4930 | 0.7060 | 0.2150 | 967.6000 |

**Caveats**
- Posterior = 90 samples (MC-dropout T=30 × 3 seed(s); dropout layers active=1); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).

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
| prt | 1.6300 | 0.1600 | 0.0000 | -6.8270 | -1.9810 | 0.0010 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -0.6700 | -0.1180 | 0.0000 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.8250 | 0.8340 | 0.8370 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `original1dcnn/ood_honesty/ood_delta_prt.png`, `original1dcnn/ood_honesty/ood_delta_taurex.png`, `original1dcnn/ood_honesty/ood_delta_psg.png`, `original1dcnn/ood_honesty/result.json`</sub>

## Section K — Bayesian reference retrieval (information ceiling)  ⏭️

> **Skipped.** no T0 reference — run `PYTHONPATH=src python -m evaluation.bayes.run_prior_is`

