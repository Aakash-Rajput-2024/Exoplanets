# Evaluation report — `causal`

*generated 2026-07-09T22:38:24*  ·  seeds [0]  ·  device `mps`  ·  ran 50/50 epochs  ·  git `aad036b02b15`

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
| R2_covered @a=1 | 0.1283 |
| R2_all12 @a=1 | 0.0578 |
| R2_covered noiseless-ref (a=300) | 0.7810 |
| R2_all12 noiseless-ref | 0.5371 |
| RMSE_all12 (dex) @a=1 | 0.4450 |
| n_seeds | 1 |
| per_seed_R2_all12_mean±std | [0.0578, None] |

**Per-species R²(log10) @ α=1 (95% bootstrap CI)**

| species | R² | 95% CI | RMSE(dex) | MAE(dex) |
|---|---|---|---|---|
| H2O | 0.0621 | [0.0456, 0.0805] | 0.4376 | 0.2989 |
| CO2 | 0.0286 | [0.0125, 0.0472] | 0.3618 | 0.2337 |
| O2 | 0.1175 | [0.0954, 0.1410] | 0.3635 | 0.2327 |
| N2 | -0.0619 | [-0.0711, -0.0533] | 0.3913 | 0.2488 |
| CH4 | 0.1063 | [0.0872, 0.1265] | 0.4212 | 0.2925 |
| N2O | -0.0357 | [-0.0464, -0.0259] | 0.4519 | 0.3127 |
| CO | -0.0568 | [-0.0658, -0.0483] | 0.4734 | 0.3259 |
| O3 | 0.3268 | [0.3049, 0.3481] | 0.4806 | 0.3440 |
| SO2 | -0.0460 | [-0.0543, -0.0380] | 0.4692 | 0.3256 |
| NH3 | 0.2472 | [0.2219, 0.2705] | 0.4048 | 0.2762 |
| C2H6 | 0.0477 | [0.0335, 0.0609] | 0.6134 | 0.4458 |
| NO2 | -0.0425 | [-0.0525, -0.0320] | 0.4707 | 0.3243 |

**SNR sweep (α = √(t/t_nom))**

| alpha | exposure× | SNR_planet(band) | R²_covered | R²_all12 |
|---|---|---|---|---|
| 0.3000 | 0.0900 | 0.8985 | -0.0214 | -0.0348 |
| 1.0000 | 1.0000 | 2.9950 | 0.1283 | 0.0578 |
| 3.0000 | 9.0000 | 8.9849 | 0.3507 | 0.1977 |
| 10.0000 | 100.0000 | 29.9498 | 0.5608 | 0.3525 |
| 30.0000 | 900.0000 | 89.8493 | 0.6954 | 0.4689 |
| 100.0000 | 10000.0000 | 299.4975 | 0.7904 | 0.5519 |
| 300.0000 | 90000.0000 | 898.4926 | 0.8345 | 0.5886 |

**Caveats**
- α=1 R²≈0 is EXPECTED BY PHYSICS (planet ~10³× below the LUVOIR per-bin noise); read the noiseless reference and the sweep, not the single α=1 number.
- PRELIMINARY: n<3 seeds — no across-seed variance; CIs are test-planet bootstrap only.

<sub>artifacts: `causal/in_distribution/r2_vs_snr.png`, `causal/in_distribution/result.json`</sub>

## Section B — Classical baselines (PriorMean / Ridge / RandomForest)  ✅
*Ground truth: exact. The linear/prior information floor the neural net must beat.*

| metric | value |
|---|---|
| neural_all12_R2@a1 | 0.0578 |
| Ridge_floor_R2 | 0.0310 |
| beats_linear_floor | True |

**Overall R²(log10), all-12, same observable & α**

| model | R²(log10) | note |
|---|---|---|
| PriorMean | -0.0592 | no-information reference (R²≈0 line) |
| Ridge | 0.0310 | linear information floor |
| RandomForest | 0.0130 | capacity-limited nonlinear ref |
| THIS: causal (α=1) | 0.0578 | neural, same observable/α |

**Caveats**
- Baselines are TRACK-INDEPENDENT (classical fits on the same cache); the neural row is this track at α=1. For the rigorous paired-bootstrap-significant comparison at α*, see VALIDATION_PLAN A3.

<sub>artifacts: `causal/baselines/result.json`</sub>

## Section C — Cross-generator (pRT / TauREx / MultiREx)  ✅
*Ground truth: synthetic labels, DIFFERENT generator. Measures overfit to PSG physics.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.7810 |
| prt_R2_covered_ref | -3.0475 |
| taurex_R2_covered_ref | -0.3251 |

**Covered-species R²(log10): native INARA vs each generator**

| generator | N | R²_covered noiseless-ref | R²_covered @α=1 | gap vs INARA (ref) |
|---|---|---|---|---|
| INARA (native) | - | 0.7810 | - | 0.0000 |
| prt | 2000 | -3.0475 | -0.9477 | -3.8285 |
| taurex | 2000 | -0.3251 | -0.1115 | -1.1061 |

**Caveats**
- A negative/low cross-gen R² mixes (a) real domain shift, (b) engine label/scale shift, and (c) forward-model approximation. Gate the gap on the Section-D PSG anchor and decompose it with Section-J honesty stats before quoting it.

<sub>artifacts: `causal/cross_generator/r2_vs_snr_prt.png`, `causal/cross_generator/r2_vs_snr_taurex.png`, `causal/cross_generator/result.json`</sub>

## Section D — PSG sanity anchor (eval-path control)  ✅
*Ground truth: exact (real held-out PSG). Validates the cross-gen eval path itself.*

| metric | value |
|---|---|
| native_R2_covered_ref | 0.7810 |
| anchor_R2_covered_ref | 0.7473 |
| anchor/native | 0.9568 |
| PASS(≥0.9×) | True |

**Per-species R²(log10) on the PSG anchor (α=1, 95% CI)**

| species | R² | 95% CI |
|---|---|---|
| H2O | 0.7090 | [0.6725, 0.7408] |
| CO2 | 0.6308 | [0.5711, 0.6912] |
| O2 | 0.8032 | [0.7768, 0.8276] |
| N2 | 0.0346 | [0.0024, 0.0635] |
| CH4 | 0.7278 | [0.6910, 0.7575] |
| N2O | 0.5575 | [0.5180, 0.5937] |
| CO | -0.0788 | [-0.1109, -0.0434] |
| O3 | 0.8656 | [0.8533, 0.8775] |
| SO2 | -0.0283 | [-0.0586, 5.048e-04] |
| NH3 | 0.7795 | [0.7485, 0.8050] |
| C2H6 | 0.4219 | [0.3754, 0.4681] |
| NO2 | 0.6926 | [0.6461, 0.7327] |

<sub>artifacts: `causal/psg_anchor/result.json`</sub>

## Section E — Solar-System-as-exoplanet (known composition)  ✅
*Ground truth: LITERAL (known VMRs). The gold-standard real-target accuracy test.*

| metric | value |
|---|---|
| dominant-gas correct (terrestrial) | 2/4 |
| mean dex-err covered (terrestrial) | 1.5028 |

**Per-target recovery (noiseless reference)**

| target | dominant true | dominant pred | dom✓ | mean dex-err (covered) | ordering ρ |
|---|---|---|---|---|---|
| Earth | N2 | CO2 | ✗ | 2.2100 | 0.8500 |
| Mars | CO2 | CO2 | ✓ | 1.1400 | 0.8500 |
| Venus | CO2 | CO2 | ✓ | 2.0200 | 0.6800 |
| Titan | N2 | CO2 | ✗ | 0.6400 | 0.2700 |
| Jupiter (giant/honesty) | CH4 | CO2 | ✗ | 1.4100 | -0.1000 |
| Saturn (giant/honesty) | CH4 | CO2 | ✗ | 1.0800 | -0.1000 |
| Uranus (giant/honesty) | CH4 | CO2 | ✗ | 0.1400 | 0.2600 |
| Neptune (giant/honesty) | CH4 | CO2 | ✗ | 0.3200 | 0.2600 |

**Honesty probe — giants must NOT show high O2/O3**

| giant | pred O2+O3 | verdict |
|---|---|---|
| Jupiter | 0.2395 | ⚠ fabricated O2/O3 |
| Saturn | 0.2375 | ⚠ fabricated O2/O3 |
| Uranus | 0.2367 | ⚠ fabricated O2/O3 |
| Neptune | 0.2367 | ⚠ fabricated O2/O3 |

**Caveats**
- Spectra are BAND-TEMPLATE PROXIES (reflected_engine), not line-by-line RT — this tests whether the model responds to which gases are present, not absolute fidelity. For RT fidelity, generate a pRT/TauREx solar-system cache and score it via Section C.
- Giants are outside the 12-simplex (H2/He dominate) — they are an honesty probe, not an accuracy test.
- INARA's training LABEL distribution samples thick exotic atmospheres, so Earth-like ppb trace gases are out-of-distribution; read the MAJOR/covered gases.

<sub>artifacts: `causal/solar_system/truth_Earth.png`, `causal/solar_system/truth_Mars.png`, `causal/solar_system/truth_Venus.png`, `causal/solar_system/truth_Titan.png`, `causal/solar_system/truth_Jupiter.png`, `causal/solar_system/truth_Saturn.png`, `causal/solar_system/truth_Uranus.png`, `causal/solar_system/truth_Neptune.png`, `causal/solar_system/result.json`</sub>

## Section F — Real disk-integrated Earth (VPL Robinson 2011)  ✅
*Ground truth: LITERAL, on REAL photons. The strongest single real test.*

| metric | value |
|---|---|
| dominant true | N2 |
| dominant pred | O2 |
| dominant correct | False |
| mean dex-err (covered) | 1.9903 |
| ordering ρ | 0.7546 |

**Earth composition: truth vs predicted**

| species | Earth VMR | pred (noiseless) | pred (α=1) | covered |
|---|---|---|---|---|
| H2O | 1.00e-02 | 4.67e-02 | 3.44e-02 | yes |
| CO2 | 4.20e-04 | 2.66e-01 | 2.89e-01 | yes |
| O2 | 2.09e-01 | 4.03e-01 | 3.26e-01 | yes |
| N2 | 7.81e-01 | 2.62e-01 | 2.95e-01 | - |
| CH4 | 1.90e-06 | 1.67e-03 | 3.20e-02 | yes |
| N2O | 3.30e-07 | 3.94e-03 | 6.51e-03 | - |
| CO | 1.20e-07 | 6.96e-03 | 6.33e-03 | - |
| O3 | 7.00e-07 | 1.25e-03 | 1.28e-03 | yes |
| SO2 | 1.00e-12 | 6.72e-03 | 6.28e-03 | - |
| NH3 | 1.00e-12 | 1.37e-03 | 3.58e-03 | - |
| C2H6 | 1.00e-12 | 6.71e-06 | 1.24e-04 | - |
| NO2 | 1.00e-12 | 5.85e-06 | 6.50e-06 | - |

**Caveats**
- REAL Robinson 2011 VPL spectrum (units W/m²/µm/sr) regridded onto the INARA grid and median-matched to INARA before the INARA-fit norm — the observable/units are adapted, not identical to INARA.
- INARA labels sample thick exotic atmospheres, so Earth's ppm/ppb trace gases (CH4, N2O, O3...) are out-of-distribution; both models over-predict them. Read the MAJOR gases (N2, O2, H2O, CO2).
- Single target → qualitative read, no R². Extensible to EPOXI/Galileo files dropped into data/real_earth/ with the same loader.

<sub>artifacts: `causal/real_earth/truth_bars_Earth-VPL.png`, `causal/real_earth/result.json`</sub>

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
| DEMO: WASP-39b-like transmission | 2.3500 | 2.9× | 0.0035 |

**Caveats**
- ⛔ NOT AN ACCURACY TEST. These are wrong-observable inputs for a reflected-light direct-imaging model — any predicted composition is meaningless. The point is to SHOW the inputs are far off-distribution (large anomaly) and/or unstable.
- A high anomaly × INARA-median is the expected, desired result: the model correctly sees a transiting spectrum as out-of-distribution.
- To probe real files, drop 2–3-column (wavelength_µm, depth/flux[, err]) spectra into data/real_spectra/ (see fetch_benchmarks.py for optional download helpers).

<sub>artifacts: `causal/transiting_ood/result.json`</sub>

## Section H — Published-retrieval comparison (benchmark exoplanets)  ✅
*Pseudo-truth: literature posteriors. Mostly OOD observable — ordering, not accuracy.*

| metric | value |
|---|---|
| n_targets | 6 |
| n_near-domain (direct-imaging) | 2 |
| mean dex-err (near-domain) | 2.0225 |

**Model vs published retrieval (dex error on retrieved species)**

| target | domain | observable | retrieved | mean dex-err | ρ |
|---|---|---|---|---|---|
| WASP-39b | far | transmission | H2O,CO2,CO,SO2,CH4 | 2.7500 | 0.1000 |
| HD 189733b | far | transmission+emission | H2O,CO,CH4 | 2.4300 | 0.0000 |
| WASP-96b | far | transmission | H2O,CO2 | 2.8400 | -1.0000 |
| WASP-43b | far | emission | H2O,CO2,CH4 | 3.4900 | 0.5000 |
| 51 Eridani b | near | direct_imaging | CH4,H2O,CO2 | 2.4500 | -1.0000 |
| HR 8799 e | near | direct_imaging | H2O,CO,CH4 | 1.5900 | 0.5000 |

**Caveats**
- Published abundances are themselves MODEL-DEPENDENT posterior estimates (often with large error bars / degeneracies) — pseudo-truth, not ground truth.
- domain_match=far (transmission/emission hot Jupiters) is a DIFFERENT observable than this model's reflected-light domain: those rows are context, NOT accuracy. Only domain_match=near (directly-imaged giants) is a meaningful, still-caveated comparison.
- Spectra are band-template proxies from published params; see reflected_engine docstring.

<sub>artifacts: `causal/published_retrieval/result.json`</sub>

## Section I — Posterior calibration (SBC / TARP / PIT / ECE)  🟡
*Ground truth: exact. Is the reported uncertainty trustworthy?*

| metric | value |
|---|---|
| n_posterior_samples | 20 |
| coverage_68 (active) | 0.1796 |
| coverage_95 (active) | 0.3190 |
| PIT-KS (active mean) | 0.4526 |
| reliability_ECE | 0.3778 |
| TARP_ECE | 0.3805 |
| posterior_spread(dex) | 0.0408 |

**Per-species calibration**

| species | cov68 | cov95 | PIT-KS | SBC χ² |
|---|---|---|---|---|
| H2O | 0.1660 | 0.3300 | 0.3980 | 1902.6000 |
| CO2 | 0.2060 | 0.3480 | 0.3840 | 1794.0000 |
| O2 | 0.1840 | 0.3460 | 0.4680 | 2499.0000 |
| N2 | 0.0860 | 0.1620 | 0.4080 | 2987.0000 |
| CH4 | 0.2260 | 0.3840 | 0.4220 | 1683.4000 |
| N2O | 0.1880 | 0.3160 | 0.5460 | 2661.6000 |
| CO | 0.0540 | 0.1020 | 0.5280 | 3707.5000 |
| O3 | 0.2500 | 0.3980 | 0.4800 | 2386.0000 |
| SO2 | 0.0340 | 0.0820 | 0.4880 | 3686.6000 |
| NH3 | 0.2020 | 0.3520 | 0.5420 | 2516.3000 |
| C2H6 | 0.0720 | 0.1460 | 0.5300 | 3365.8000 |
| NO2 | 0.2680 | 0.4880 | 0.2680 | 1065.4000 |

**Caveats**
- Posterior = 20 samples (MC-dropout T=20 × 1 seed(s); dropout layers active=7); computed at the noiseless reference (α=300).
- Well-calibrated ⇒ 68% coverage∈[0.64,0.72], 95%∈[0.92,0.97], flat SBC/PIT, TARP on the diagonal. Inactive species (N2, CO) are expected to be wide (the model should report ignorance, not fabricate).
- PRELIMINARY: n<3 seeds — limited deep-ensemble diversity.

<sub>artifacts: `causal/calibration/reliability.png`, `causal/calibration/tarp_coverage.png`, `causal/calibration/sbc_ranks.png`, `causal/calibration/result.json`</sub>

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
| prt | 1.6300 | 0.1600 | 0.0000 | -3.0480 | -0.9550 | 0.0020 | no transfer (genuine) |
| taurex | 0.8900 | 0.0000 | 0.0000 | -0.3250 | -0.1300 | 0.0010 | no transfer (genuine) |
| psg | 0.0100 | 0.9900 | 1.0000 | 0.7470 | 0.7660 | 0.7770 | biased extrapolation (fixable) |

**Caveats**
- δ,v are computed AFTER the INARA-fit norm, so δ≈0 / v≈1 means the generator input sits inside INARA's distribution at that wavelength.
- A raw negative R² is never published without its debiased companion (this section).

<sub>artifacts: `causal/ood_honesty/ood_delta_prt.png`, `causal/ood_honesty/ood_delta_taurex.png`, `causal/ood_honesty/ood_delta_psg.png`, `causal/ood_honesty/result.json`</sub>

