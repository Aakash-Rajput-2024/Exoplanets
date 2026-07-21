# Evaluation report — `optimized1dcnn`

*generated 2026-07-09T22:20:22*  ·  seeds [0, 1, 2]  ·  device `mps`  ·  ran 48/50 epochs  ·  git `unknown`

> Reflected-light / direct-imaging retrieval (0.2–2.0 µm, LUVOIR-like). Each section states its epistemic status. The only LITERAL-ground-truth real tests are Sections E (solar-system) and F (real Earth); transiting-planet data (G, and 'far' rows of H) is a wrong-observable OOD probe, not an accuracy measurement.

## Contents

- **A** In-distribution (INARA held-out test) — ✅ ok
- **B** Classical baselines (PriorMean / Ridge / RandomForest) — ✅ ok
- **C** Cross-generator (pRT / TauREx / MultiREx) — ✅ ok
- **D** PSG sanity anchor (eval-path control) — 🟡 preliminary
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
| R2_covered @a=1 | 0.1354 |
| R2_all12 @a=1 | 0.0668 |
| R2_covered noiseless-ref (a=300) | 0.7285 |
| R2_all12 noiseless-ref | 0.5301 |
| RMSE_all12 (dex) @a=1 | 0.4432 |
| n_seeds | 3 |
| per_seed_R2_all12_mean±std | [0.0628, 0.0032] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | 0.0728 | [0.0581, 0.0901] | 0.4351 | 0.2952 |
| CO2 | 0.0624 | [0.0462, 0.0787] | 0.3555 | 0.2269 |
| O2 | 0.1200 | [0.1010, 0.1387] | 0.3630 | 0.2304 |
| N2 | -0.0652 | [-0.0746, -0.0563] | 0.3919 | 0.2474 |
| CH4 | 0.1129 | [0.0955, 0.1307] | 0.4197 | 0.2903 |
| N2O | 0.0382 | [0.0274, 0.0503] | 0.4355 | 0.2970 |
| CO | -0.0571 | [-0.0660, -0.0486] | 0.4734 | 0.3254 |
| O3 | 0.3086 | [0.2908, 0.3272] | 0.4871 | 0.3492 |
| SO2 | -0.0453 | [-0.0534, -0.0375] | 0.4691 | 0.3249 |
| NH3 | 0.2439 | [0.2238, 0.2640] | 0.4057 | 0.2751 |
| C2H6 | 0.0521 | [0.0397, 0.0639] | 0.6120 | 0.4443 |
| NO2 | -0.0420 | [-0.0511, -0.0325] | 0.4706 | 0.3241 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0153 | -0.0287 |
| 1.0000 | 1.0000 | 2.9950 | 0.1354 | 0.0668 |
| 3.0000 | 9.0000 | 8.9849 | 0.3481 | 0.2069 |
| 10.0000 | 100.0000 | 29.9498 | 0.5346 | 0.3408 |
| 30.0000 | 900.0000 | 89.8493 | 0.6568 | 0.4422 |
| 100.0000 | 10000.0000 | 299.4975 | 0.7454 | 0.5189 |
| 300.0000 | 90000.0000 | 898.4926 | 0.7852 | 0.5522 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.

<sub>artifacts: `optimized1dcnn/in_distribution/r2_vs_snr.png`, `optimized1dcnn/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0668 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | True |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: optimized1dcnn (α=1) | 0.0668 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.

<sub>artifacts: `optimized1dcnn/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.7285 |
| prt_R2_covered_ref | -3.7359 |
| taurex_R2_covered_ref | -0.2735 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.7285 | - | 0.0000 |
| prt | 2000 | -3.7359 | -2.2188 | -4.4644 |
| taurex | 2000 | -0.2735 | -0.1012 | -1.0020 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `optimized1dcnn/cross_generator/r2_vs_snr_prt.png`, `optimized1dcnn/cross_generator/r2_vs_snr_taurex.png`, `optimized1dcnn/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  🟡
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.7285 |
| anchor_R2_covered_ref | 0.6394 |
| anchor/native | 0.8777 |
| PASS(≥0.9×) | False |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.5926 | [0.5609, 0.6221] |
| CO2 | 0.4596 | [0.3949, 0.5176] |
| O2 | 0.5862 | [0.5415, 0.6284] |
| N2 | -0.0451 | [-0.0802, -0.0129] |
| CH4 | 0.8024 | [0.7819, 0.8222] |
| N2O | 0.4292 | [0.3886, 0.4689] |
| CO | -0.0161 | [-0.0428, 0.0117] |
| O3 | 0.7560 | [0.7135, 0.7930] |
| SO2 | 0.1625 | [0.1239, 0.1954] |
| NH3 | 0.7671 | [0.7321, 0.7968] |
| C2H6 | 0.4357 | [0.3944, 0.4791] |
| NO2 | 0.3669 | [0.2746, 0.4518] |

**Caveats**
- ⚠ ANCHOR BELOW 0.9× NATIVE: the eval path (median-match / noise bootstrap / symlink) is degrading the score, so any cross-generator gap in Section C is confounded and must NOT be attributed to generator physics until this passes.

<sub>artifacts: `optimized1dcnn/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 0/4 |
| mean dex-err covered (terrestrial) | 1.5771 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | O2 | ✗ | 2.2200 | 0.8800 |
| Mars | CO2 | O2 | ✗ | 1.2900 | 0.8200 |
| Venus | CO2 | O2 | ✗ | 2.1400 | 0.5700 |
| Titan | N2 | O2 | ✗ | 0.6500 | 0.2700 |
| Jupiter (giant/honesty) | CH4 | O2 | ✗ | 1.5700 | -0.0200 |
| Saturn (giant/honesty) | CH4 | O2 | ✗ | 1.2600 | -0.0200 |
| Uranus (giant/honesty) | CH4 | O2 | ✗ | 0.3500 | 0.2600 |
| Neptune (giant/honesty) | CH4 | O2 | ✗ | 0.5300 | 0.2600 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.5408 | ⚠ fabricated O2/O3 |
| Saturn | 0.5376 | ⚠ fabricated O2/O3 |
| Uranus | 0.5033 | ⚠ fabricated O2/O3 |
| Neptune | 0.5041 | ⚠ fabricated O2/O3 |

**Caveats**
- Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this tests whether the model responds to which gases are present, not absolute fidelity. For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.
- Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, not an accuracy test.
- INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like ppb trace gases are out-of-distribution; read the MAJOR/covered gases.

<sub>artifacts: `optimized1dcnn/solar_system/truth_Earth.png`, `optimized1dcnn/solar_system/truth_Mars.png`, `optimized1dcnn/solar_system/truth_Venus.png`, `optimized1dcnn/solar_system/truth_Titan.png`, `optimized1dcnn/solar_system/truth_Jupiter.png`, `optimized1dcnn/solar_system/truth_Saturn.png`, `optimized1dcnn/solar_system/truth_Uranus.png`, `optimized1dcnn/solar_system/truth_Neptune.png`, `optimized1dcnn/solar_system/result.json`</sub>

## Section F — Real disk-integrated Earth (VPL Robinson 2011)  ✅
*Ground truth: LITERAL, on REAL photons. The strongest single real test.*

| metric | value |
|---|---|
| dominant true | N2 |
| dominant pred | O2 |
| dominant correct | False |
| mean dex-err (covered) | 2.1924 |
| ordering ρ | 0.8507 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 7.04e-02 | 3.47e-02 | yes |
| CO2 | 4.20e-04 | 1.32e-01 | 2.97e-01 | yes |
| O2 | 2.09e-01 | 4.69e-01 | 3.08e-01 | yes |
| N2 | 7.81e-01 | 2.94e-01 | 3.04e-01 | - |
| CH4 | 1.90e-06 | 1.47e-02 | 3.21e-02 | yes |
| N2O | 3.30e-07 | 4.49e-03 | 6.69e-03 | - |
| CO | 1.20e-07 | 6.85e-03 | 6.39e-03 | - |
| O3 | 7.00e-07 | 1.67e-03 | 1.08e-03 | yes |
| SO2 | 1.00e-12 | 4.55e-03 | 6.32e-03 | - |
| NH3 | 1.00e-12 | 2.61e-03 | 3.40e-03 | - |
| C2H6 | 1.00e-12 | 5.90e-05 | 1.24e-04 | - |
| NO2 | 1.00e-12 | 4.39e-06 | 6.55e-06 | - |

**Caveats**
- REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and median-matched to INARA before the INARA-fit norm — the observable/units are adapted, not identical to INARA.
- INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases (N2, O2, H2O, CO2).
- Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into data/real_earth/ with the same loader.

<sub>artifacts: `optimized1dcnn/real_earth/truth_bars_Earth-VPL.png`, `optimized1dcnn/real_earth/result.json`</sub>

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0042 |

**Caveats**
- ⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light direct-imaging model — any predicted composition is meaningless. The point is to SHOW the inputs are far off-distribution (large anomaly) and/or unstable.
- A high anomaly × INARA-median is the expected, desired result: the model correctly sees a transiting spectrum as out-of-distribution.
- To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).

<sub>artifacts: `optimized1dcnn/transiting_ood/result.json`</sub>

## Section H — Published-retrieval comparison (benchmark exoplanets)  ✅
*Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy.*

| metric | value |
|---|---|
| n_targets | 6 |
| n_near-domain (direct-imaging) | 2 |
| mean dex-err (near-domain) | 1.9819 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.6400 | 0.1000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.5000 | 0.0000 |
| WASP-96b | far | transmission | H2O,CO2 | 2.6700 | -1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.4300 | 0.5000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.3300 | -0.5000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.6300 | 0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `optimized1dcnn/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  ✅
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 60 |
| coverage_68 (active) | 0.3598 |
| coverage_95 (active) | 0.5418 |
| PIT-KS (active mean) | 0.3176 |
| reliability_ECE | 0.2668 |
| TARP_ECE | 0.3415 |
| posterior_spread(dex) | 0.0808 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.3140 | 0.4840 | 0.3040 | 1308.9000 |
| CO2 | 0.3160 | 0.4740 | 0.3520 | 1395.0000 |
| O2 | 0.3100 | 0.5000 | 0.4450 | 2027.6000 |
| N2 | 0.1160 | 0.2180 | 0.3750 | 2762.7000 |
| CH4 | 0.4800 | 0.6820 | 0.2100 | 413.8000 |
| N2O | 0.3880 | 0.5680 | 0.2870 | 879.9000 |
| CO | 0.0580 | 0.1180 | 0.4430 | 3529.8000 |
| O3 | 0.4140 | 0.7180 | 0.2620 | 473.3000 |
| SO2 | 0.2440 | 0.3900 | 0.3920 | 1808.2000 |
| NH3 | 0.4760 | 0.6600 | 0.2610 | 634.8000 |
| C2H6 | 0.2000 | 0.3000 | 0.3990 | 2210.9000 |
| NO2 | 0.4560 | 0.6420 | 0.2640 | 704.8000 |

**Caveats**
- Posterior = 60 samples (MC-dropout T=20 × 3 seed(s); dropout layers active=1); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).

<sub>artifacts: `optimized1dcnn/calibration/reliability.png`, `optimized1dcnn/calibration/tarp_coverage.png`, `optimized1dcnn/calibration/sbc_ranks.png`, `optimized1dcnn/calibration/result.json`</sub>

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
| prt | 1.6300 | 0.1600 | 0.0000 | -3.7360 | -1.3270 | 0.0010 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -0.2740 | -0.0960 | 0.0000 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.6390 | 0.6600 | 0.6720 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `optimized1dcnn/ood_honesty/ood_delta_prt.png`, `optimized1dcnn/ood_honesty/ood_delta_taurex.png`, `optimized1dcnn/ood_honesty/ood_delta_psg.png`, `optimized1dcnn/ood_honesty/result.json`</sub>

