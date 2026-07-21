# Evaluation report — `optimized1dcnn_grey`

*generated 2026-07-09T22:22:14*  ·  seeds [0]  ·  device `mps`  ·  ran 49/50 epochs  ·  git `741b2672c15b`

> Reflected-light / direct-imaging retrieval (0.2–2.0 µm, LUVOIR-like). Each section states its epistemic status. The only LITERAL-ground-truth real tests are Sections E (solar-system) and F (real Earth); transiting-planet data (G, and 'far' rows of H) is a wrong-observable OOD probe, not an accuracy measurement.

## Contents

- **A** In-distribution (INARA held-out test) — 🟡 preliminary
- **B** Classical baselines (PriorMean / Ridge / RandomForest) — ✅ ok
- **C** Cross-generator (pRT / TauREx / MultiREx) — ✅ ok
- **D** PSG sanity anchor (eval-path control) — 🟡 preliminary
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
| R2_covered @a=1 | 0.0370 |
| R2_all12 @a=1 | -0.0022 |
| R2_covered noiseless-ref (a=300) | 0.1750 |
| R2_all12 noiseless-ref | 0.1444 |
| RMSE_all12 (dex) @a=1 | 0.4599 |
| n_seeds | 1 |
| per_seed_R2_all12_mean±std | [-0.0022, None] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | -0.0422 | [-0.0527, -0.0315] | 0.4613 | 0.3112 |
| CO2 | -0.0390 | [-0.0487, -0.0296] | 0.3742 | 0.2415 |
| O2 | 0.0371 | [0.0169, 0.0587] | 0.3797 | 0.2441 |
| N2 | -0.0646 | [-0.0730, -0.0567] | 0.3918 | 0.2495 |
| CH4 | 0.0031 | [-0.0116, 0.0191] | 0.4449 | 0.3081 |
| N2O | -0.0667 | [-0.0758, -0.0584] | 0.4586 | 0.3152 |
| CO | -0.0695 | [-0.0785, -0.0611] | 0.4762 | 0.3264 |
| O3 | 0.2259 | [0.2069, 0.2459] | 0.5154 | 0.3704 |
| SO2 | -0.0543 | [-0.0621, -0.0465] | 0.4711 | 0.3254 |
| NH3 | 0.1022 | [0.0839, 0.1202] | 0.4421 | 0.2998 |
| C2H6 | -0.0094 | [-0.0201, 0.0017] | 0.6315 | 0.4585 |
| NO2 | -0.0494 | [-0.0588, -0.0398] | 0.4722 | 0.3244 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0460 | -0.0514 |
| 1.0000 | 1.0000 | 2.9950 | 0.0370 | -0.0022 |
| 3.0000 | 9.0000 | 8.9849 | 0.1465 | 0.0669 |
| 10.0000 | 100.0000 | 29.9498 | 0.2534 | 0.1450 |
| 30.0000 | 900.0000 | 89.8493 | 0.3231 | 0.2068 |
| 100.0000 | 10000.0000 | 299.4975 | 0.3668 | 0.2539 |
| 300.0000 | 90000.0000 | 898.4926 | 0.3744 | 0.2615 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.
- PRELIMINARY: n<3 seeds — no across-seed variance; CIs are test-planet bootstrap only.

<sub>artifacts: `optimized1dcnn_grey/in_distribution/r2_vs_snr.png`, `optimized1dcnn_grey/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | -0.0022 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | False |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: optimized1dcnn_grey (α=1) | -0.0022 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.
- ⚠ This track's α=1 R² does NOT exceed the Ridge floor — at the photon floor that is expected; compare at the noiseless ref / α* before concluding.

<sub>artifacts: `optimized1dcnn_grey/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.1750 |
| prt_R2_covered_ref | -4.5835 |
| taurex_R2_covered_ref | -1.5290 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.1750 | - | 0.0000 |
| prt | 2000 | -4.5835 | -2.4479 | -4.7585 |
| taurex | 2000 | -1.5290 | -0.1169 | -1.7040 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `optimized1dcnn_grey/cross_generator/r2_vs_snr_prt.png`, `optimized1dcnn_grey/cross_generator/r2_vs_snr_taurex.png`, `optimized1dcnn_grey/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  🟡
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.1750 |
| anchor_R2_covered_ref | 0.0970 |
| anchor/native | 0.5545 |
| PASS(≥0.9×) | False |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | -0.6509 | [-0.8479, -0.4890] |
| CO2 | -0.1337 | [-0.2232, -0.0612] |
| O2 | 0.2559 | [0.1896, 0.3129] |
| N2 | -0.1573 | [-0.1941, -0.1227] |
| CH4 | 0.3448 | [0.2813, 0.4053] |
| N2O | -0.1188 | [-0.1517, -0.0869] |
| CO | -0.1767 | [-0.2133, -0.1406] |
| O3 | 0.6689 | [0.6328, 0.7013] |
| SO2 | 0.0724 | [0.0301, 0.1126] |
| NH3 | 0.1007 | [-0.0061, 0.1911] |
| C2H6 | 0.1952 | [0.1382, 0.2461] |
| NO2 | 0.5360 | [0.4743, 0.5972] |

**Caveats**
- ⚠ ANCHOR BELOW 0.9× NATIVE: the eval path (median-match / noise bootstrap / symlink) is degrading the score, so any cross-generator gap in Section C is confounded and must NOT be attributed to generator physics until this passes.

<sub>artifacts: `optimized1dcnn_grey/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 0/4 |
| mean dex-err covered (terrestrial) | 1.5730 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | O2 | ✗ | 2.1400 | 0.9500 |
| Mars | CO2 | O2 | ✗ | 1.3100 | 0.7500 |
| Venus | CO2 | O2 | ✗ | 2.2000 | 0.6200 |
| Titan | N2 | O2 | ✗ | 0.6400 | 0.2700 |
| Jupiter (giant/honesty) | CH4 | O2 | ✗ | 1.4100 | -0.0800 |
| Saturn (giant/honesty) | CH4 | O2 | ✗ | 1.2300 | -0.0800 |
| Uranus (giant/honesty) | CH4 | O2 | ✗ | 0.1400 | 0.2600 |
| Neptune (giant/honesty) | CH4 | O2 | ✗ | 0.3200 | 0.2600 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.6073 | ⚠ fabricated O2/O3 |
| Saturn | 0.5956 | ⚠ fabricated O2/O3 |
| Uranus | 0.6340 | ⚠ fabricated O2/O3 |
| Neptune | 0.6367 | ⚠ fabricated O2/O3 |

**Caveats**
- Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this tests whether the model responds to which gases are present, not absolute fidelity. For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.
- Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, not an accuracy test.
- INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like ppb trace gases are out-of-distribution; read the MAJOR/covered gases.

<sub>artifacts: `optimized1dcnn_grey/solar_system/truth_Earth.png`, `optimized1dcnn_grey/solar_system/truth_Mars.png`, `optimized1dcnn_grey/solar_system/truth_Venus.png`, `optimized1dcnn_grey/solar_system/truth_Titan.png`, `optimized1dcnn_grey/solar_system/truth_Jupiter.png`, `optimized1dcnn_grey/solar_system/truth_Saturn.png`, `optimized1dcnn_grey/solar_system/truth_Uranus.png`, `optimized1dcnn_grey/solar_system/truth_Neptune.png`, `optimized1dcnn_grey/solar_system/result.json`</sub>

## Section F — Real disk-integrated Earth (VPL Robinson 2011)  ✅
*Ground truth: LITERAL, on REAL photons. The strongest single real test.*

| metric | value |
|---|---|
| dominant true | N2 |
| dominant pred | O2 |
| dominant correct | False |
| mean dex-err (covered) | 2.0517 |
| ordering ρ | 0.8720 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 2.33e-02 | 3.33e-02 | yes |
| CO2 | 4.20e-04 | 3.33e-02 | 3.00e-01 | yes |
| O2 | 2.09e-01 | 7.00e-01 | 3.17e-01 | yes |
| N2 | 7.81e-01 | 2.08e-01 | 2.93e-01 | - |
| CH4 | 1.90e-06 | 7.40e-03 | 3.26e-02 | yes |
| N2O | 3.30e-07 | 4.59e-03 | 6.55e-03 | - |
| CO | 1.20e-07 | 6.76e-03 | 6.37e-03 | - |
| O3 | 7.00e-07 | 5.28e-03 | 1.18e-03 | yes |
| SO2 | 1.00e-12 | 6.83e-03 | 6.30e-03 | - |
| NH3 | 1.00e-12 | 4.33e-03 | 3.63e-03 | - |
| C2H6 | 1.00e-12 | 2.72e-05 | 1.26e-04 | - |
| NO2 | 1.00e-12 | 5.85e-06 | 6.56e-06 | - |

**Caveats**
- REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and median-matched to INARA before the INARA-fit norm — the observable/units are adapted, not identical to INARA.
- INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases (N2, O2, H2O, CO2).
- Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into data/real_earth/ with the same loader.

<sub>artifacts: `optimized1dcnn_grey/real_earth/truth_bars_Earth-VPL.png`, `optimized1dcnn_grey/real_earth/result.json`</sub>

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0066 |

**Caveats**
- ⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light direct-imaging model — any predicted composition is meaningless. The point is to SHOW the inputs are far off-distribution (large anomaly) and/or unstable.
- A high anomaly × INARA-median is the expected, desired result: the model correctly sees a transiting spectrum as out-of-distribution.
- To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).

<sub>artifacts: `optimized1dcnn_grey/transiting_ood/result.json`</sub>

## Section H — Published-retrieval comparison (benchmark exoplanets)  ✅
*Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy.*

| metric | value |
|---|---|
| n_targets | 6 |
| n_near-domain (direct-imaging) | 2 |
| mean dex-err (near-domain) | 1.7492 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.4100 | -0.5000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.2800 | -0.8700 |
| WASP-96b | far | transmission | H2O,CO2 | 2.1700 | -1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.1100 | 0.5000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.1100 | -0.5000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.3900 | -0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `optimized1dcnn_grey/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  🟡
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 20 |
| coverage_68 (active) | 0.1708 |
| coverage_95 (active) | 0.2956 |
| PIT-KS (active mean) | 0.4236 |
| reliability_ECE | 0.3852 |
| TARP_ECE | 0.3762 |
| posterior_spread(dex) | 0.0613 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.1540 | 0.2720 | 0.4880 | 2441.7000 |
| CO2 | 0.1180 | 0.2160 | 0.4960 | 2735.0000 |
| O2 | 0.1240 | 0.2300 | 0.5020 | 2917.4000 |
| N2 | 0.0860 | 0.1360 | 0.5160 | 3383.9000 |
| CH4 | 0.2520 | 0.4200 | 0.3100 | 1229.6000 |
| N2O | 0.1060 | 0.1800 | 0.4120 | 2877.0000 |
| CO | 0.0680 | 0.1040 | 0.5320 | 3750.4000 |
| O3 | 0.1980 | 0.3520 | 0.4440 | 2324.9000 |
| SO2 | 0.1400 | 0.2660 | 0.5260 | 3065.7000 |
| NH3 | 0.2100 | 0.3660 | 0.3340 | 1559.4000 |
| C2H6 | 0.1620 | 0.2560 | 0.4040 | 2320.9000 |
| NO2 | 0.2440 | 0.3980 | 0.3200 | 1523.8000 |

**Caveats**
- Posterior = 20 samples (MC-dropout T=20 × 1 seed(s); dropout layers active=1); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).
- PRELIMINARY: n<3 seeds — limited deep-ensemble diversity.

<sub>artifacts: `optimized1dcnn_grey/calibration/reliability.png`, `optimized1dcnn_grey/calibration/tarp_coverage.png`, `optimized1dcnn_grey/calibration/sbc_ranks.png`, `optimized1dcnn_grey/calibration/result.json`</sub>

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
| prt | 1.6300 | 0.1600 | 0.0000 | -4.5840 | -3.2330 | 0.0010 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -1.5290 | -0.2450 | 0.0000 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.0970 | 0.1640 | 0.3340 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `optimized1dcnn_grey/ood_honesty/ood_delta_prt.png`, `optimized1dcnn_grey/ood_honesty/ood_delta_taurex.png`, `optimized1dcnn_grey/ood_honesty/ood_delta_psg.png`, `optimized1dcnn_grey/ood_honesty/result.json`</sub>

