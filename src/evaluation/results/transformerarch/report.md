# Evaluation report — `transformerarch`

*generated 2026-07-09T22:30:44*  ·  seeds [0]  ·  device `mps`  ·  ran 48/50 epochs  ·  git `741b2672c15b`

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
| R2_covered @a=1 | 0.1382 |
| R2_all12 @a=1 | 0.0647 |
| R2_covered noiseless-ref (a=300) | 0.5684 |
| R2_all12 noiseless-ref | 0.3674 |
| RMSE_all12 (dex) @a=1 | 0.4432 |
| n_seeds | 1 |
| per_seed_R2_all12_mean±std | [0.0647, None] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | 0.0704 | [0.0536, 0.0894] | 0.4357 | 0.2975 |
| CO2 | 0.0292 | [0.0126, 0.0469] | 0.3617 | 0.2330 |
| O2 | 0.1355 | [0.1134, 0.1591] | 0.3598 | 0.2310 |
| N2 | -0.0651 | [-0.0742, -0.0558] | 0.3919 | 0.2484 |
| CH4 | 0.1114 | [0.0917, 0.1320] | 0.4200 | 0.2917 |
| N2O | -0.0309 | [-0.0411, -0.0210] | 0.4509 | 0.3130 |
| CO | -0.0539 | [-0.0630, -0.0453] | 0.4727 | 0.3258 |
| O3 | 0.3447 | [0.3234, 0.3667] | 0.4742 | 0.3395 |
| SO2 | -0.0417 | [-0.0496, -0.0342] | 0.4683 | 0.3254 |
| NH3 | 0.2584 | [0.2338, 0.2817] | 0.4018 | 0.2744 |
| C2H6 | 0.0500 | [0.0360, 0.0634] | 0.6126 | 0.4449 |
| NO2 | -0.0318 | [-0.0420, -0.0208] | 0.4683 | 0.3236 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0128 | -0.0283 |
| 1.0000 | 1.0000 | 2.9950 | 0.1382 | 0.0647 |
| 3.0000 | 9.0000 | 8.9849 | 0.3577 | 0.2035 |
| 10.0000 | 100.0000 | 29.9498 | 0.5639 | 0.3569 |
| 30.0000 | 900.0000 | 89.8493 | 0.6999 | 0.4726 |
| 100.0000 | 10000.0000 | 299.4975 | 0.7955 | 0.5536 |
| 300.0000 | 90000.0000 | 898.4926 | 0.8396 | 0.5873 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.
- PRELIMINARY: n<3 seeds — no across-seed variance; CIs are test-planet bootstrap only.

<sub>artifacts: `transformerarch/in_distribution/r2_vs_snr.png`, `transformerarch/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0647 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | True |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: transformerarch (α=1) | 0.0647 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.

<sub>artifacts: `transformerarch/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.5684 |
| prt_R2_covered_ref | -1.0914 |
| taurex_R2_covered_ref | -1.3593 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.5684 | - | 0.0000 |
| prt | 2000 | -1.0914 | -0.4617 | -1.6598 |
| taurex | 2000 | -1.3593 | -0.0996 | -1.9278 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `transformerarch/cross_generator/r2_vs_snr_prt.png`, `transformerarch/cross_generator/r2_vs_snr_taurex.png`, `transformerarch/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  ✅
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.5684 |
| anchor_R2_covered_ref | 0.5789 |
| anchor/native | 1.0184 |
| PASS(≥0.9×) | True |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.5218 | [0.4506, 0.5813] |
| CO2 | 0.3702 | [0.2816, 0.4463] |
| O2 | 0.7732 | [0.7429, 0.8044] |
| N2 | -0.2308 | [-0.2766, -0.1880] |
| CH4 | 0.4204 | [0.3509, 0.4809] |
| N2O | 0.4351 | [0.3844, 0.4816] |
| CO | -0.1247 | [-0.1587, -0.0900] |
| O3 | 0.8089 | [0.7851, 0.8296] |
| SO2 | -0.0827 | [-0.1149, -0.0521] |
| NH3 | 0.7210 | [0.6860, 0.7502] |
| C2H6 | 0.2854 | [0.2264, 0.3420] |
| NO2 | 0.5233 | [0.4682, 0.5698] |

<sub>artifacts: `transformerarch/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 2/4 |
| mean dex-err covered (terrestrial) | 1.6502 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | N2 | ✓ | 2.0600 | 0.8400 |
| Mars | CO2 | N2 | ✗ | 1.5700 | 0.8300 |
| Venus | CO2 | N2 | ✗ | 2.3400 | 0.6600 |
| Titan | N2 | N2 | ✓ | 0.6400 | 0.3500 |
| Jupiter (giant/honesty) | CH4 | N2 | ✗ | 1.5300 | 0.0000 |
| Saturn (giant/honesty) | CH4 | N2 | ✗ | 1.0700 | 0.0000 |
| Uranus (giant/honesty) | CH4 | N2 | ✗ | 0.1600 | 0.3100 |
| Neptune (giant/honesty) | CH4 | N2 | ✗ | 0.0200 | 0.3100 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.3033 | ⚠ fabricated O2/O3 |
| Saturn | 0.3029 | ⚠ fabricated O2/O3 |
| Uranus | 0.3132 | ⚠ fabricated O2/O3 |
| Neptune | 0.3135 | ⚠ fabricated O2/O3 |

**Caveats**
- Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this tests whether the model responds to which gases are present, not absolute fidelity. For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.
- Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, not an accuracy test.
- INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like ppb trace gases are out-of-distribution; read the MAJOR/covered gases.

<sub>artifacts: `transformerarch/solar_system/truth_Earth.png`, `transformerarch/solar_system/truth_Mars.png`, `transformerarch/solar_system/truth_Venus.png`, `transformerarch/solar_system/truth_Titan.png`, `transformerarch/solar_system/truth_Jupiter.png`, `transformerarch/solar_system/truth_Saturn.png`, `transformerarch/solar_system/truth_Uranus.png`, `transformerarch/solar_system/truth_Neptune.png`, `transformerarch/solar_system/result.json`</sub>

## Section F — Real disk-integrated Earth (VPL Robinson 2011)  ✅
*Ground truth: LITERAL, on REAL photons. The strongest single real test.*

| metric | value |
|---|---|
| dominant true | N2 |
| dominant pred | O2 |
| dominant correct | False |
| mean dex-err (covered) | 1.8895 |
| ordering ρ | 0.7510 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 5.54e-02 | 3.32e-02 | yes |
| CO2 | 4.20e-04 | 7.19e-02 | 2.99e-01 | yes |
| O2 | 2.09e-01 | 5.39e-01 | 3.13e-01 | yes |
| N2 | 7.81e-01 | 3.15e-01 | 3.00e-01 | - |
| CH4 | 1.90e-06 | 7.10e-04 | 3.11e-02 | yes |
| N2O | 3.30e-07 | 1.66e-03 | 6.38e-03 | - |
| CO | 1.20e-07 | 6.68e-03 | 6.22e-03 | - |
| O3 | 7.00e-07 | 2.15e-03 | 1.20e-03 | yes |
| SO2 | 1.00e-12 | 7.22e-03 | 6.16e-03 | - |
| NH3 | 1.00e-12 | 7.48e-04 | 3.39e-03 | - |
| C2H6 | 1.00e-12 | 2.89e-06 | 1.20e-04 | - |
| NO2 | 1.00e-12 | 7.84e-06 | 6.27e-06 | - |

**Caveats**
- REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and median-matched to INARA before the INARA-fit norm — the observable/units are adapted, not identical to INARA.
- INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases (N2, O2, H2O, CO2).
- Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into data/real_earth/ with the same loader.

<sub>artifacts: `transformerarch/real_earth/truth_bars_Earth-VPL.png`, `transformerarch/real_earth/result.json`</sub>

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0036 |

**Caveats**
- ⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light direct-imaging model — any predicted composition is meaningless. The point is to SHOW the inputs are far off-distribution (large anomaly) and/or unstable.
- A high anomaly × INARA-median is the expected, desired result: the model correctly sees a transiting spectrum as out-of-distribution.
- To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).

<sub>artifacts: `transformerarch/transiting_ood/result.json`</sub>

## Section H — Published-retrieval comparison (benchmark exoplanets)  ✅
*Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy.*

| metric | value |
|---|---|
| n_targets | 6 |
| n_near-domain (direct-imaging) | 2 |
| mean dex-err (near-domain) | 1.9472 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.6500 | 0.7000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.4900 | 0.8700 |
| WASP-96b | far | transmission | H2O,CO2 | 2.6000 | 1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.2200 | 1.0000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.2400 | -1.0000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.6600 | 0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `transformerarch/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  🟡
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 20 |
| coverage_68 (active) | 0.1662 |
| coverage_95 (active) | 0.2970 |
| PIT-KS (active mean) | 0.5330 |
| reliability_ECE | 0.3823 |
| TARP_ECE | 0.3682 |
| posterior_spread(dex) | 0.0561 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.1600 | 0.2980 | 0.5540 | 2670.0000 |
| CO2 | 0.1100 | 0.2520 | 0.6840 | 4070.7000 |
| O2 | 0.2420 | 0.3940 | 0.3220 | 1575.2000 |
| N2 | 0.1100 | 0.1980 | 0.6180 | 4108.2000 |
| CH4 | 0.1340 | 0.2740 | 0.6760 | 3895.1000 |
| N2O | 0.2360 | 0.4040 | 0.4380 | 1619.9000 |
| CO | 0.0740 | 0.1340 | 0.5240 | 3493.0000 |
| O3 | 0.2780 | 0.4600 | 0.2860 | 1082.1000 |
| SO2 | 0.0800 | 0.1320 | 0.4920 | 3357.4000 |
| NH3 | 0.1700 | 0.3240 | 0.6080 | 3051.9000 |
| C2H6 | 0.1100 | 0.1700 | 0.6200 | 3768.3000 |
| NO2 | 0.1420 | 0.2620 | 0.6500 | 4482.6000 |

**Caveats**
- Posterior = 20 samples (MC-dropout T=20 × 1 seed(s); dropout layers active=7); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).
- PRELIMINARY: n<3 seeds — limited deep-ensemble diversity.

<sub>artifacts: `transformerarch/calibration/reliability.png`, `transformerarch/calibration/tarp_coverage.png`, `transformerarch/calibration/sbc_ranks.png`, `transformerarch/calibration/result.json`</sub>

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
| prt | 1.6300 | 0.1600 | 0.0000 | -1.0910 | -0.2090 | 0.0010 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -1.3590 | -0.1030 | 0.0000 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.5790 | 0.6870 | 0.7350 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `transformerarch/ood_honesty/ood_delta_prt.png`, `transformerarch/ood_honesty/ood_delta_taurex.png`, `transformerarch/ood_honesty/ood_delta_psg.png`, `transformerarch/ood_honesty/result.json`</sub>

