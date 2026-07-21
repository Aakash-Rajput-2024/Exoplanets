# Evaluation report — `causal_cfi_cfi`

*generated 2026-07-09T22:52:14*  ·  seeds [0, 1]  ·  device `mps`  ·  ran 34/50 epochs  ·  git `unknown`

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
| R2_covered @a=1 | 0.0618 |
| R2_all12 @a=1 | 0.0170 |
| R2_covered noiseless-ref (a=300) | 0.3553 |
| R2_all12 noiseless-ref | 0.2111 |
| RMSE_all12 (dex) @a=1 | 0.4552 |
| n_seeds | 2 |
| per_seed_R2_all12_mean±std | [0.0164, 0.0023] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | 0.0230 | [0.0129, 0.0332] | 0.4467 | 0.3041 |
| CO2 | -0.0402 | [-0.0507, -0.0298] | 0.3744 | 0.2380 |
| O2 | 0.0409 | [0.0295, 0.0516] | 0.3790 | 0.2397 |
| N2 | -0.0777 | [-0.0858, -0.0696] | 0.3942 | 0.2488 |
| CH4 | 0.0684 | [0.0570, 0.0797] | 0.4301 | 0.3023 |
| N2O | -0.0539 | [-0.0622, -0.0462] | 0.4559 | 0.3142 |
| CO | -0.0631 | [-0.0716, -0.0549] | 0.4748 | 0.3258 |
| O3 | 0.2169 | [0.2036, 0.2296] | 0.5184 | 0.3775 |
| SO2 | -0.0585 | [-0.0668, -0.0507] | 0.4720 | 0.3254 |
| NH3 | 0.1723 | [0.1572, 0.1867] | 0.4245 | 0.2923 |
| C2H6 | 0.0294 | [0.0207, 0.0377] | 0.6192 | 0.4544 |
| NO2 | -0.0535 | [-0.0612, -0.0456] | 0.4732 | 0.3250 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0214 | -0.0341 |
| 1.0000 | 1.0000 | 2.9950 | 0.0618 | 0.0170 |
| 3.0000 | 9.0000 | 8.9849 | 0.1756 | 0.0844 |
| 10.0000 | 100.0000 | 29.9498 | 0.2730 | 0.1451 |
| 30.0000 | 900.0000 | 89.8493 | 0.3260 | 0.1834 |
| 100.0000 | 10000.0000 | 299.4975 | 0.3530 | 0.2072 |
| 300.0000 | 90000.0000 | 898.4926 | 0.3586 | 0.2126 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.
- PRELIMINARY: n<3 seeds — no across-seed variance; CIs are test-planet bootstrap only.

<sub>artifacts: `causal_cfi_cfi/in_distribution/r2_vs_snr.png`, `causal_cfi_cfi/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0170 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | False |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: causal_cfi_cfi (α=1) | 0.0170 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.
- ⚠ This track's α=1 R² does NOT exceed the Ridge floor — at the photon floor that is expected; compare at the noiseless ref / α* before concluding.

<sub>artifacts: `causal_cfi_cfi/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.3553 |
| prt_R2_covered_ref | -0.3567 |
| taurex_R2_covered_ref | -0.1101 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.3553 | - | 0.0000 |
| prt | 2000 | -0.3567 | -0.1502 | -0.7120 |
| taurex | 2000 | -0.1101 | -0.0735 | -0.4654 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `causal_cfi_cfi/cross_generator/r2_vs_snr_prt.png`, `causal_cfi_cfi/cross_generator/r2_vs_snr_taurex.png`, `causal_cfi_cfi/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  ✅
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.3553 |
| anchor_R2_covered_ref | 0.3542 |
| anchor/native | 0.9967 |
| PASS(≥0.9×) | True |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.3050 | [0.2836, 0.3281] |
| CO2 | 0.1082 | [0.0777, 0.1401] |
| O2 | 0.3156 | [0.2951, 0.3373] |
| N2 | -0.0671 | [-0.0925, -0.0450] |
| CH4 | 0.4254 | [0.4037, 0.4468] |
| N2O | -0.0622 | [-0.0854, -0.0393] |
| CO | -0.0567 | [-0.0798, -0.0356] |
| O3 | 0.6167 | [0.6008, 0.6324] |
| SO2 | -0.0455 | [-0.0684, -0.0248] |
| NH3 | 0.5347 | [0.5103, 0.5576] |
| C2H6 | 0.2281 | [0.2035, 0.2534] |
| NO2 | 0.1725 | [0.1485, 0.1969] |

<sub>artifacts: `causal_cfi_cfi/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 0/4 |
| mean dex-err covered (terrestrial) | 1.4996 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | O2 | ✗ | 2.2200 | 0.8200 |
| Mars | CO2 | O2 | ✗ | 1.1600 | 0.7500 |
| Venus | CO2 | O2 | ✗ | 1.9900 | 0.5700 |
| Titan | N2 | O2 | ✗ | 0.6400 | 0.2700 |
| Jupiter (giant/honesty) | CH4 | O2 | ✗ | 1.4700 | -0.0800 |
| Saturn (giant/honesty) | CH4 | O2 | ✗ | 1.2000 | -0.0800 |
| Uranus (giant/honesty) | CH4 | O2 | ✗ | 0.2100 | 0.2600 |
| Neptune (giant/honesty) | CH4 | O2 | ✗ | 0.3900 | 0.2600 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.3461 | ⚠ fabricated O2/O3 |
| Saturn | 0.3453 | ⚠ fabricated O2/O3 |
| Uranus | 0.3394 | ⚠ fabricated O2/O3 |
| Neptune | 0.3394 | ⚠ fabricated O2/O3 |

**Caveats**
- Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this tests whether the model responds to which gases are present, not absolute fidelity. For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.
- Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, not an accuracy test.
- INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like ppb trace gases are out-of-distribution; read the MAJOR/covered gases.

<sub>artifacts: `causal_cfi_cfi/solar_system/truth_Earth.png`, `causal_cfi_cfi/solar_system/truth_Mars.png`, `causal_cfi_cfi/solar_system/truth_Venus.png`, `causal_cfi_cfi/solar_system/truth_Titan.png`, `causal_cfi_cfi/solar_system/truth_Jupiter.png`, `causal_cfi_cfi/solar_system/truth_Saturn.png`, `causal_cfi_cfi/solar_system/truth_Uranus.png`, `causal_cfi_cfi/solar_system/truth_Neptune.png`, `causal_cfi_cfi/solar_system/result.json`</sub>

## Section F — Real disk-integrated Earth (VPL Robinson 2011)  ✅
*Ground truth: LITERAL, on REAL photons. The strongest single real test.*

| metric | value |
|---|---|
| dominant true | N2 |
| dominant pred | O2 |
| dominant correct | False |
| mean dex-err (covered) | 2.2115 |
| ordering ρ | 0.8329 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 3.89e-02 | 3.24e-02 | yes |
| CO2 | 4.20e-04 | 2.62e-01 | 2.97e-01 | yes |
| O2 | 2.09e-01 | 3.47e-01 | 3.14e-01 | yes |
| N2 | 7.81e-01 | 3.01e-01 | 3.03e-01 | - |
| CH4 | 1.90e-06 | 2.67e-02 | 3.02e-02 | yes |
| N2O | 3.30e-07 | 6.33e-03 | 6.25e-03 | - |
| CO | 1.20e-07 | 6.53e-03 | 6.27e-03 | - |
| O3 | 7.00e-07 | 1.42e-03 | 1.16e-03 | yes |
| SO2 | 1.00e-12 | 6.65e-03 | 6.28e-03 | - |
| NH3 | 1.00e-12 | 3.38e-03 | 3.18e-03 | - |
| C2H6 | 1.00e-12 | 1.03e-04 | 1.17e-04 | - |
| NO2 | 1.00e-12 | 6.22e-06 | 6.23e-06 | - |

**Caveats**
- REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and median-matched to INARA before the INARA-fit norm — the observable/units are adapted, not identical to INARA.
- INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases (N2, O2, H2O, CO2).
- Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into data/real_earth/ with the same loader.

<sub>artifacts: `causal_cfi_cfi/real_earth/truth_bars_Earth-VPL.png`, `causal_cfi_cfi/real_earth/result.json`</sub>

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0020 |

**Caveats**
- ⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light direct-imaging model — any predicted composition is meaningless. The point is to SHOW the inputs are far off-distribution (large anomaly) and/or unstable.
- A high anomaly × INARA-median is the expected, desired result: the model correctly sees a transiting spectrum as out-of-distribution.
- To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).

<sub>artifacts: `causal_cfi_cfi/transiting_ood/result.json`</sub>

## Section H — Published-retrieval comparison (benchmark exoplanets)  ✅
*Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy.*

| metric | value |
|---|---|
| n_targets | 6 |
| n_near-domain (direct-imaging) | 2 |
| mean dex-err (near-domain) | 2.0057 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.7100 | -0.5000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.4300 | -0.8700 |
| WASP-96b | far | transmission | H2O,CO2 | 2.6900 | -1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.4700 | -0.5000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.4100 | -0.5000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.6000 | -0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `causal_cfi_cfi/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  🟡
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 40 |
| coverage_68 (active) | 0.0790 |
| coverage_95 (active) | 0.1172 |
| PIT-KS (active mean) | 0.4554 |
| reliability_ECE | 0.4415 |
| TARP_ECE | 0.4524 |
| posterior_spread(dex) | 0.0202 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.0700 | 0.1080 | 0.4560 | 3546.3000 |
| CO2 | 0.1200 | 0.1600 | 0.4380 | 3089.1000 |
| O2 | 0.0940 | 0.1400 | 0.4380 | 3305.8000 |
| N2 | 0.0400 | 0.0600 | 0.4900 | 3972.2000 |
| CH4 | 0.0700 | 0.1000 | 0.4580 | 3602.4000 |
| N2O | 0.0400 | 0.0500 | 0.4960 | 4043.8000 |
| CO | 0.0200 | 0.0340 | 0.5160 | 4194.6000 |
| O3 | 0.1440 | 0.1940 | 0.4040 | 2858.6000 |
| SO2 | 0.0260 | 0.0400 | 0.5000 | 4159.3000 |
| NH3 | 0.1160 | 0.1880 | 0.4260 | 2993.4000 |
| C2H6 | 0.0440 | 0.0720 | 0.4540 | 3868.7000 |
| NO2 | 0.0660 | 0.1200 | 0.4840 | 3502.8000 |

**Caveats**
- Posterior = 40 samples (MC-dropout T=20 × 2 seed(s); dropout layers active=7); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).
- PRELIMINARY: n<3 seeds — limited deep-ensemble diversity.

<sub>artifacts: `causal_cfi_cfi/calibration/reliability.png`, `causal_cfi_cfi/calibration/tarp_coverage.png`, `causal_cfi_cfi/calibration/sbc_ranks.png`, `causal_cfi_cfi/calibration/result.json`</sub>

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
| prt | 1.6300 | 0.1600 | 0.0000 | -0.3570 | -0.2330 | 0.0050 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -0.1100 | -0.0700 | 0.0010 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.3540 | 0.3650 | 0.6240 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `causal_cfi_cfi/ood_honesty/ood_delta_prt.png`, `causal_cfi_cfi/ood_honesty/ood_delta_taurex.png`, `causal_cfi_cfi/ood_honesty/ood_delta_psg.png`, `causal_cfi_cfi/ood_honesty/result.json`</sub>

