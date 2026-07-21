# Evaluation report — `transformerarch_grey`

*generated 2026-07-09T13:42:37*  ·  seeds [0]  ·  device `mps`  ·  ran 10/10 epochs  ·  git `aad036b02b15`

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
| R2_covered @a=1 | 0.0767 |
| R2_all12 @a=1 | 0.0210 |
| R2_covered noiseless-ref (a=300) | 0.3943 |
| R2_all12 noiseless-ref | 0.2705 |
| RMSE_all12 (dex) @a=1 | 0.4540 |
| n_seeds | 1 |
| per_seed_R2_all12_mean±std | [0.0210, None] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | -0.0060 | [-0.0190, 0.0092] | 0.4532 | 0.3067 |
| CO2 | -0.0294 | [-0.0404, -0.0184] | 0.3725 | 0.2402 |
| O2 | 0.0877 | [0.0656, 0.1108] | 0.3696 | 0.2400 |
| N2 | -0.0702 | [-0.0791, -0.0614] | 0.3929 | 0.2492 |
| CH4 | 0.0449 | [0.0273, 0.0638] | 0.4354 | 0.3022 |
| N2O | -0.0600 | [-0.0689, -0.0516] | 0.4572 | 0.3147 |
| CO | -0.0665 | [-0.0753, -0.0581] | 0.4755 | 0.3265 |
| O3 | 0.2864 | [0.2657, 0.3083] | 0.4948 | 0.3570 |
| SO2 | -0.0539 | [-0.0621, -0.0461] | 0.4710 | 0.3254 |
| NH3 | 0.1543 | [0.1335, 0.1751] | 0.4291 | 0.2923 |
| C2H6 | 0.0151 | [0.0030, 0.0272] | 0.6238 | 0.4533 |
| NO2 | -0.0508 | [-0.0602, -0.0401] | 0.4726 | 0.3253 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0361 | -0.0469 |
| 1.0000 | 1.0000 | 2.9950 | 0.0767 | 0.0210 |
| 3.0000 | 9.0000 | 8.9849 | 0.2188 | 0.1088 |
| 10.0000 | 100.0000 | 29.9498 | 0.3522 | 0.2011 |
| 30.0000 | 900.0000 | 89.8493 | 0.4401 | 0.2747 |
| 100.0000 | 10000.0000 | 299.4975 | 0.5067 | 0.3344 |
| 300.0000 | 90000.0000 | 898.4926 | 0.5363 | 0.3601 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.
- PRELIMINARY: n<3 seeds — no across-seed variance; CIs are test-planet bootstrap only.

<sub>artifacts: `transformerarch_grey/in_distribution/r2_vs_snr.png`, `transformerarch_grey/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0210 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | False |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: transformerarch_grey (α=1) | 0.0210 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.
- ⚠ This track's α=1 R² does NOT exceed the Ridge floor — at the photon floor that is expected; compare at the noiseless ref / α* before concluding.

<sub>artifacts: `transformerarch_grey/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.3943 |
| prt_R2_covered_ref | -0.5139 |
| taurex_R2_covered_ref | -1.8907 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.3943 | - | 0.0000 |
| prt | 2000 | -0.5139 | -0.7698 | -0.9082 |
| taurex | 2000 | -1.8907 | -0.1207 | -2.2850 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `transformerarch_grey/cross_generator/r2_vs_snr_prt.png`, `transformerarch_grey/cross_generator/r2_vs_snr_taurex.png`, `transformerarch_grey/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  ✅
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.3943 |
| anchor_R2_covered_ref | 0.4079 |
| anchor/native | 1.0345 |
| PASS(≥0.9×) | True |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.2965 | [0.1963, 0.3866] |
| CO2 | -0.0410 | [-0.0901, 0.0082] |
| O2 | 0.4303 | [0.3846, 0.4702] |
| N2 | -0.0591 | [-0.0851, -0.0343] |
| CH4 | 0.6225 | [0.5779, 0.6588] |
| N2O | -0.0374 | [-0.0616, -0.0137] |
| CO | -0.0107 | [-0.0326, 0.0103] |
| O3 | 0.7312 | [0.7032, 0.7547] |
| SO2 | 0.0117 | [-0.0082, 0.0287] |
| NH3 | 0.6876 | [0.6512, 0.7197] |
| C2H6 | 0.3787 | [0.3265, 0.4269] |
| NO2 | 0.2050 | [0.0848, 0.3031] |

<sub>artifacts: `transformerarch_grey/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 2/4 |
| mean dex-err covered (terrestrial) | 1.6019 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | N2 | ✓ | 2.1800 | 0.6500 |
| Mars | CO2 | N2 | ✗ | 1.0900 | 0.5700 |
| Venus | CO2 | N2 | ✗ | 1.9800 | 0.4100 |
| Titan | N2 | N2 | ✓ | 1.1500 | 0.2300 |
| Jupiter (giant/honesty) | CH4 | N2 | ✗ | 1.3900 | -0.1500 |
| Saturn (giant/honesty) | CH4 | N2 | ✗ | 1.5800 | -0.1500 |
| Uranus (giant/honesty) | CH4 | N2 | ✗ | 0.8400 | 0.1000 |
| Neptune (giant/honesty) | CH4 | N2 | ✗ | 1.0200 | 0.1000 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.2350 | ⚠ fabricated O2/O3 |
| Saturn | 0.2375 | ⚠ fabricated O2/O3 |
| Uranus | 0.2476 | ⚠ fabricated O2/O3 |
| Neptune | 0.2473 | ⚠ fabricated O2/O3 |

**Caveats**
- Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this tests whether the model responds to which gases are present, not absolute fidelity. For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.
- Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, not an accuracy test.
- INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like ppb trace gases are out-of-distribution; read the MAJOR/covered gases.

<sub>artifacts: `transformerarch_grey/solar_system/truth_Earth.png`, `transformerarch_grey/solar_system/truth_Mars.png`, `transformerarch_grey/solar_system/truth_Venus.png`, `transformerarch_grey/solar_system/truth_Titan.png`, `transformerarch_grey/solar_system/truth_Jupiter.png`, `transformerarch_grey/solar_system/truth_Saturn.png`, `transformerarch_grey/solar_system/truth_Uranus.png`, `transformerarch_grey/solar_system/truth_Neptune.png`, `transformerarch_grey/solar_system/result.json`</sub>

## Section F — Real disk-integrated Earth (VPL Robinson 2011)  ✅
*Ground truth: LITERAL, on REAL photons. The strongest single real test.*

| metric | value |
|---|---|
| dominant true | N2 |
| dominant pred | O2 |
| dominant correct | False |
| mean dex-err (covered) | 2.0335 |
| ordering ρ | 0.7830 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 6.20e-02 | 3.54e-02 | yes |
| CO2 | 4.20e-04 | 2.30e-01 | 2.85e-01 | yes |
| O2 | 2.09e-01 | 4.07e-01 | 3.23e-01 | yes |
| N2 | 7.81e-01 | 2.81e-01 | 3.00e-01 | - |
| CH4 | 1.90e-06 | 1.46e-03 | 3.25e-02 | yes |
| N2O | 3.30e-07 | 4.75e-03 | 6.40e-03 | - |
| CO | 1.20e-07 | 5.01e-03 | 6.40e-03 | - |
| O3 | 7.00e-07 | 2.03e-03 | 1.33e-03 | yes |
| SO2 | 1.00e-12 | 5.70e-03 | 6.35e-03 | - |
| NH3 | 1.00e-12 | 1.31e-03 | 3.62e-03 | - |
| C2H6 | 1.00e-12 | 6.09e-06 | 1.27e-04 | - |
| NO2 | 1.00e-12 | 2.90e-06 | 6.56e-06 | - |

**Caveats**
- REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and median-matched to INARA before the INARA-fit norm — the observable/units are adapted, not identical to INARA.
- INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases (N2, O2, H2O, CO2).
- Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into data/real_earth/ with the same loader.

<sub>artifacts: `transformerarch_grey/real_earth/truth_bars_Earth-VPL.png`, `transformerarch_grey/real_earth/result.json`</sub>

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0063 |

**Caveats**
- ⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light direct-imaging model — any predicted composition is meaningless. The point is to SHOW the inputs are far off-distribution (large anomaly) and/or unstable.
- A high anomaly × INARA-median is the expected, desired result: the model correctly sees a transiting spectrum as out-of-distribution.
- To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).

<sub>artifacts: `transformerarch_grey/transiting_ood/result.json`</sub>

## Section H — Published-retrieval comparison (benchmark exoplanets)  ✅
*Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy.*

| metric | value |
|---|---|
| n_targets | 6 |
| n_near-domain (direct-imaging) | 2 |
| mean dex-err (near-domain) | 1.9252 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.6600 | -0.8000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.3800 | -0.8700 |
| WASP-96b | far | transmission | H2O,CO2 | 1.9400 | -1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.1600 | -1.0000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.1100 | 0.5000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.7400 | -1.0000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `transformerarch_grey/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  🟡
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 20 |
| coverage_68 (active) | 0.0860 |
| coverage_95 (active) | 0.1620 |
| PIT-KS (active mean) | 0.5802 |
| reliability_ECE | 0.4397 |
| TARP_ECE | 0.4376 |
| posterior_spread(dex) | 0.0275 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.1100 | 0.1930 | 0.5750 | 4008.7000 |
| CO2 | 0.0620 | 0.1100 | 0.6380 | 5161.8000 |
| O2 | 0.1050 | 0.1680 | 0.5570 | 4062.9000 |
| N2 | 0.0350 | 0.0700 | 0.4750 | 4620.7000 |
| CH4 | 0.1120 | 0.2400 | 0.5870 | 3951.4000 |
| N2O | 0.0420 | 0.0880 | 0.4580 | 4378.4000 |
| CO | 0.0400 | 0.0620 | 0.4870 | 4676.1000 |
| O3 | 0.1220 | 0.2170 | 0.6530 | 4714.3000 |
| SO2 | 0.0270 | 0.0600 | 0.5180 | 4778.1000 |
| NH3 | 0.1350 | 0.2570 | 0.4950 | 3169.1000 |
| C2H6 | 0.0600 | 0.1220 | 0.5850 | 4479.9000 |
| NO2 | 0.0870 | 0.1650 | 0.7350 | 6019.7000 |

**Caveats**
- Posterior = 20 samples (MC-dropout T=20 × 1 seed(s); dropout layers active=7); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).
- PRELIMINARY: n<3 seeds — limited deep-ensemble diversity.

<sub>artifacts: `transformerarch_grey/calibration/reliability.png`, `transformerarch_grey/calibration/tarp_coverage.png`, `transformerarch_grey/calibration/sbc_ranks.png`, `transformerarch_grey/calibration/result.json`</sub>

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
| prt | 1.6300 | 0.1600 | 0.0000 | -0.5140 | -0.4710 | 0.0110 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -1.8910 | -0.0900 | 0.0010 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.4080 | 0.5140 | 0.5750 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `transformerarch_grey/ood_honesty/ood_delta_prt.png`, `transformerarch_grey/ood_honesty/ood_delta_taurex.png`, `transformerarch_grey/ood_honesty/ood_delta_psg.png`, `transformerarch_grey/ood_honesty/result.json`</sub>

