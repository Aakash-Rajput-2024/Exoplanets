# Evaluation report — `transformerarch`

*generated 2026-08-09T15:48:38*  ·  seeds [0, 1, 2]  ·  device `mps`  ·  ran 48/50 epochs  ·  git `741b2672c15b`

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
| R2_covered @a=1 | 0.1394 |
| R2_all12 @a=1 | 0.0651 |
| R2_covered noiseless-ref (a=300) | 0.7684 |
| R2_all12 noiseless-ref | 0.5118 |
| RMSE_all12 (dex) @a=1 | 0.4430 |
| n_seeds | 3 |
| per_seed_R2_all12_mean±std | [0.0630, 0.0015] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | 0.0716 | [0.0542, 0.0910] | 0.4354 | 0.2972 |
| CO2 | 0.0270 | [0.0111, 0.0434] | 0.3621 | 0.2330 |
| O2 | 0.1358 | [0.1143, 0.1591] | 0.3597 | 0.2310 |
| N2 | -0.0640 | [-0.0728, -0.0551] | 0.3917 | 0.2483 |
| CH4 | 0.1155 | [0.0962, 0.1355] | 0.4190 | 0.2908 |
| N2O | -0.0313 | [-0.0415, -0.0216] | 0.4510 | 0.3125 |
| CO | -0.0543 | [-0.0631, -0.0458] | 0.4728 | 0.3257 |
| O3 | 0.3469 | [0.3260, 0.3683] | 0.4734 | 0.3389 |
| SO2 | -0.0433 | [-0.0513, -0.0357] | 0.4686 | 0.3254 |
| NH3 | 0.2565 | [0.2325, 0.2790] | 0.4023 | 0.2741 |
| C2H6 | 0.0526 | [0.0389, 0.0661] | 0.6118 | 0.4447 |
| NO2 | -0.0324 | [-0.0425, -0.0217] | 0.4684 | 0.3234 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0125 | -0.0286 |
| 1.0000 | 1.0000 | 2.9950 | 0.1394 | 0.0651 |
| 3.0000 | 9.0000 | 8.9849 | 0.3601 | 0.2045 |
| 10.0000 | 100.0000 | 29.9498 | 0.5685 | 0.3594 |
| 30.0000 | 900.0000 | 89.8493 | 0.7043 | 0.4752 |
| 100.0000 | 10000.0000 | 299.4975 | 0.7990 | 0.5563 |
| 300.0000 | 90000.0000 | 898.4926 | 0.8416 | 0.5894 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.

<sub>artifacts: `transformerarch/in_distribution/r2_vs_snr.png`, `transformerarch/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0651 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | True |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: transformerarch (α=1) | 0.0651 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.

<sub>artifacts: `transformerarch/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.7684 |
| prt_R2_covered_ref | -0.8755 |
| taurex_R2_covered_ref | -0.7965 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.7684 | - | 0.0000 |
| prt | 2000 | -0.8755 | -0.5262 | -1.6439 |
| taurex | 2000 | -0.7965 | -0.1009 | -1.5649 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `transformerarch/cross_generator/r2_vs_snr_prt.png`, `transformerarch/cross_generator/r2_vs_snr_taurex.png`, `transformerarch/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  ✅
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.7684 |
| anchor_R2_covered_ref | 0.7550 |
| anchor/native | 0.9825 |
| PASS(≥0.9×) | True |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.7138 | [0.6755, 0.7448] |
| CO2 | 0.6015 | [0.5489, 0.6522] |
| O2 | 0.8209 | [0.7990, 0.8425] |
| N2 | -0.1377 | [-0.1775, -0.1005] |
| CH4 | 0.7374 | [0.7053, 0.7624] |
| N2O | 0.5714 | [0.5244, 0.6149] |
| CO | -0.0639 | [-0.0945, -0.0339] |
| O3 | 0.9012 | [0.8899, 0.9119] |
| SO2 | -0.0435 | [-0.0716, -0.0171] |
| NH3 | 0.7744 | [0.7462, 0.7983] |
| C2H6 | 0.4310 | [0.3843, 0.4752] |
| NO2 | 0.7249 | [0.6815, 0.7632] |

<sub>artifacts: `transformerarch/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 2/4 |
| mean dex-err covered (terrestrial) | 1.6885 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | N2 | ✓ | 2.1600 | 0.8600 |
| Mars | CO2 | N2 | ✗ | 1.5900 | 0.8000 |
| Venus | CO2 | N2 | ✗ | 2.3500 | 0.5800 |
| Titan | N2 | N2 | ✓ | 0.6400 | 0.3500 |
| Jupiter (giant/honesty) | CH4 | N2 | ✗ | 1.6300 | 0.0000 |
| Saturn (giant/honesty) | CH4 | N2 | ✗ | 1.2100 | 0.0000 |
| Uranus (giant/honesty) | CH4 | N2 | ✗ | 0.2400 | 0.3100 |
| Neptune (giant/honesty) | CH4 | N2 | ✗ | 0.4300 | 0.3100 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.3498 | ⚠ fabricated O2/O3 |
| Saturn | 0.3486 | ⚠ fabricated O2/O3 |
| Uranus | 0.3430 | ⚠ fabricated O2/O3 |
| Neptune | 0.3435 | ⚠ fabricated O2/O3 |

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
| mean dex-err (covered) | 2.0088 |
| ordering ρ | 0.7830 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 7.55e-02 | 3.36e-02 | yes |
| CO2 | 4.20e-04 | 1.10e-01 | 2.99e-01 | yes |
| O2 | 2.09e-01 | 4.72e-01 | 3.11e-01 | yes |
| N2 | 7.81e-01 | 3.19e-01 | 3.01e-01 | - |
| CH4 | 1.90e-06 | 1.63e-03 | 3.15e-02 | yes |
| N2O | 3.30e-07 | 3.25e-03 | 6.38e-03 | - |
| CO | 1.20e-07 | 6.90e-03 | 6.24e-03 | - |
| O3 | 7.00e-07 | 2.02e-03 | 1.20e-03 | yes |
| SO2 | 1.00e-12 | 7.13e-03 | 6.20e-03 | - |
| NH3 | 1.00e-12 | 1.44e-03 | 3.41e-03 | - |
| C2H6 | 1.00e-12 | 6.52e-06 | 1.21e-04 | - |
| NO2 | 1.00e-12 | 4.91e-06 | 6.30e-06 | - |

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0040 |

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
| mean dex-err (near-domain) | 2.0793 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.7300 | 0.2000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.6400 | 0.0000 |
| WASP-96b | far | transmission | H2O,CO2 | 2.6000 | 1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.3800 | 1.0000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.3600 | -0.5000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.8000 | 0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `transformerarch/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  ✅
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 90 |
| coverage_68 (active) | 0.4484 |
| coverage_95 (active) | 0.5999 |
| PIT-KS (active mean) | 0.2904 |
| reliability_ECE | 0.2037 |
| TARP_ECE | 0.3401 |
| posterior_spread(dex) | 0.0948 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.4120 | 0.6060 | 0.3090 | 1888.1000 |
| CO2 | 0.5010 | 0.6480 | 0.2770 | 1533.0000 |
| O2 | 0.5720 | 0.7460 | 0.1560 | 573.2000 |
| N2 | 0.2510 | 0.3770 | 0.4550 | 4511.9000 |
| CH4 | 0.4180 | 0.5940 | 0.3880 | 2681.1000 |
| N2O | 0.5200 | 0.7240 | 0.1450 | 551.5000 |
| CO | 0.1200 | 0.2020 | 0.4250 | 5706.2000 |
| O3 | 0.6450 | 0.8250 | 0.0840 | 217.8000 |
| SO2 | 0.1260 | 0.2000 | 0.3950 | 5630.8000 |
| NH3 | 0.3050 | 0.4870 | 0.4950 | 4330.5000 |
| C2H6 | 0.1790 | 0.2800 | 0.5030 | 5589.9000 |
| NO2 | 0.8060 | 0.8890 | 0.1530 | 852.9000 |

**Caveats**
- Posterior = 90 samples (MC-dropout T=30 × 3 seed(s); dropout layers active=7); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).

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
| prt | 1.6300 | 0.1600 | 0.0000 | -0.8760 | -0.2190 | 0.0020 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -0.7960 | -0.0730 | 0.0010 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.7550 | 0.7770 | 0.7840 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `transformerarch/ood_honesty/ood_delta_prt.png`, `transformerarch/ood_honesty/ood_delta_taurex.png`, `transformerarch/ood_honesty/ood_delta_psg.png`, `transformerarch/ood_honesty/result.json`</sub>

## Section K — Bayesian reference retrieval (information ceiling)  ⏭️

> **Skipped.** no T0 reference — run `PYTHONPATH=src python -m evaluation.bayes.run_prior_is`

