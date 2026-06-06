# Causality plan for the `trans_cnn` / transformer-CNN exoplanet model

Status: evidence-informed plan; no experiments run yet.

## Context observed in this repo

- The relevant model appears to be `src/transformerarch/model.py::NasaInaraTransformer`: 1D CNN downsampling over spectra, positional encoding, Transformer encoder, global average pooling, MLP regression head.
- Current training (`src/transformerarch/train.py`) uses random 80/20 split, MSE loss, standardized targets, and cached tensors.
- Labels are 12 atmospheric gas abundances: `H2O, CO2, O2, N2, CH4, N2O, CO, O3, SO2, NH3, C2H6, NO2`.
- `data/summary.csv` has useful environment/nuisance columns: `star_class`, `star_temperature`, `star_radius`, `distance_parsec`, `semimajor_axis`, `planet_radius`, `planet_density`, `surface_pressure`, `surface_temperature`, `albedo`, etc.

## What “inducing causality” should mean here

You cannot make a neural net causal just by adding an attention layer or a causal loss. A defensible goal is:

> Learn spectral features that remain predictive of atmospheric composition across nuisance shifts such as star type, distance, radius, temperature, pressure, and observation scaling, while avoiding shortcuts that only work in the random split.

For this dataset, the most practical causal route is **invariant prediction / domain generalization**, not full causal discovery. The causal intuition is: molecular abundances generate absorption/emission structure; stellar/planetary/system parameters and observation scaling can create spurious correlations. Train and test by environments so the model is rewarded for using stable mechanisms.

## Best papers to read first

### 1. Causal Representation Learning foundation

1. **“Towards Causal Representation Learning” — Schölkopf et al., 2021, arXiv:2102.11107**  
   Why: best conceptual overview of why causal mechanisms, invariance, interventions, and representation learning matter for neural models.  
   URL: https://arxiv.org/abs/2102.11107

2. **“Identifiable Causal Representation Learning” — von Kügelgen, 2024, arXiv:2406.13371**  
   Why: clarifies what is and is not identifiable from unsupervised/multi-environment data. Useful guardrail against overclaiming.  
   URL: https://arxiv.org/abs/2406.13371

### 2. Invariance / OOD methods most relevant to your model

3. **“Causal inference using invariant prediction” — Peters, Bühlmann, Meinshausen, 2015/2016, arXiv:1501.01332**  
   Why: foundational invariant causal prediction idea: causal predictors preserve their conditional relation to the target across environments.  
   URL: https://arxiv.org/abs/1501.01332

4. **“Invariant Risk Minimization” — Arjovsky et al., 2019, arXiv:1907.02893**  
   Why: neural-network-friendly version of invariant prediction. This is the core paper for adding an IRM-style penalty to `NasaInaraTransformer`.  
   URL: https://arxiv.org/abs/1907.02893

5. **“Out-of-Distribution Generalization via Risk Extrapolation (REx)” — Krueger et al., 2020/2021, arXiv:2003.00688**  
   Why: simpler than IRM in practice: penalize variance of risk across environments. Strong first implementation for your training loop.  
   URL: https://arxiv.org/abs/2003.00688

6. **“The Risks of Invariant Risk Minimization” — Rosenfeld, Ravikumar, Risteski, 2020/2021, arXiv:2010.05761**  
   Why: essential warning paper. IRM can fail, especially with insufficient environments or nonlinear models. Use it to avoid claiming “causality” too strongly.  
   URL: https://arxiv.org/abs/2010.05761

7. **“In Search of Lost Domain Generalization” — Gulrajani & Lopez-Paz, 2020/2021, arXiv:2007.01434**  
   Why: shows that many domain-generalization methods only look good under weak evaluation. Use DomainBed-style model selection and held-out environments.  
   URL: https://arxiv.org/abs/2007.01434

### 3. Counterfactual and architecture-adjacent papers

8. **“Deep Structural Causal Models for Tractable Counterfactual Inference” — Pawlowski, Coelho de Castro, Glocker, 2020, arXiv:2006.06485**  
   Why: useful if you later want explicit counterfactual generation, not just invariant training.  
   URL: https://arxiv.org/abs/2006.06485

9. **“Causal Discovery with Attention-Based Convolutional Neural Networks” — Nauta et al., 2019**  
   Why: not directly your task, but relevant because it combines temporal CNNs, attention, and causal discovery. Read for architectural inspiration, not as direct proof of causality in spectra.  
   URL: https://www.mdpi.com/2504-4990/1/1/19

## Plan of action

### Phase 0 — Define the causal question

Use this working SCM sketch:

```text
star/system parameters  ─┐
planet parameters       ─┼─> observed spectrum X
atmospheric abundance Y ─┘

star/system/planet parameters may also correlate with Y in the simulator/data distribution.
Goal: predict Y from spectral mechanisms, not from shortcut correlations tied to one environment.
```

Do **not** claim the model discovers the true physical causal graph unless you validate against interventions or simulator-controlled counterfactuals.

### Phase 1 — Replace random validation with environment validation

Create environment labels from `summary.csv`, then evaluate held-out environment generalization.

Recommended first environments:

1. `star_class` groups.
2. `star_temperature` quantile bins.
3. `distance_parsec` quantile bins.
4. `surface_temperature` quantile bins.
5. `surface_pressure` quantile bins.
6. combined stress environment, e.g. `(star_class, surface_temperature_bin)` if sample counts permit.

Deliverable: a split script that saves train/val/test indices by environment, not only random split.

Metrics:

- mean RMSE / MAE over 12 gases;
- per-gas RMSE / MAE;
- worst-environment RMSE;
- gap between random validation and held-out-environment validation.

### Phase 2 — Establish baselines before adding causality losses

Run these models under identical held-out-environment splits:

1. current `NasaInaraTransformer` with ERM/MSE;
2. original 1D CNN baseline;
3. `NasaInaraTransformer` + simple augmentations only;
4. `NasaInaraTransformer` + V-REx;
5. `NasaInaraTransformer` + IRMv1 penalty.

If ERM already wins on held-out environments, causal regularization is not helping yet.

### Phase 3 — Implement the lowest-risk causal regularizer first: V-REx

For each batch, include samples from multiple environments. Compute MSE per environment:

```python
env_losses = torch.stack([
    mse(pred[env_id == e], y[env_id == e])
    for e in env_id.unique()
    if (env_id == e).sum() >= min_env_batch
])
loss = env_losses.mean() + lambda_rex * env_losses.var(unbiased=False)
```

Why first: it is easy, stable for regression, and directly aligned with REx.

Start with:

- `lambda_rex`: 1, 10, 100 grid;
- warm-up: train ERM for 5–10 epochs, then turn on REx;
- batch sampler: ensure at least 3–4 environments per batch.

### Phase 4 — Add IRMv1 carefully

Modify model so it can return pooled features before the head:

```python
pred, features = model(x, return_features=True)
```

For regression IRM, a common practical penalty is gradient norm with respect to a dummy scale parameter:

```python
scale = torch.ones(1, device=device, requires_grad=True)
penalty = 0
for e in envs:
    loss_e = mse(pred[env == e] * scale, y[env == e])
    grad = torch.autograd.grad(loss_e, [scale], create_graph=True)[0]
    penalty = penalty + grad.pow(2).sum()
loss = erm_loss + lambda_irm * penalty
```

Use this only after V-REx is working, because IRM is easier to destabilize and easier to overinterpret.

### Phase 5 — Add causal counterfactual augmentations

Only use augmentations that should preserve the atmospheric abundance label.

Safe first augmentations:

- multiplicative flux scaling;
- additive observation noise calibrated to spectrum magnitude;
- smooth continuum tilt/baseline perturbation;
- small wavelength jitter/interpolation;
- masking/dropout of narrow wavelength regions, as long as not excessive.

Risky augmentations:

- changing absorption bands in molecule-specific regions while keeping labels fixed;
- mixing spectra with different labels;
- augmenting in ways that violate radiative-transfer physics.

Training objective:

```text
MSE(original, y)
+ MSE(counterfactual_augmented, y)
+ consistency_weight * ||f(original) - f(augmented)||^2
+ optional V-REx penalty
```

### Phase 6 — Add representation probes

After training, freeze the feature extractor and train simple probes from features to environment labels.

Desired pattern:

- abundance prediction stays strong;
- environment prediction from features drops for nuisance environments such as distance/star class;
- held-out-environment performance improves.

Caution: if an environment variable is genuinely needed to predict abundances, removing it can hurt. This is why probes are diagnostics, not the main objective.

### Phase 7 — Molecular-band interpretability check

For each gas target, compute attribution maps over wavelength using integrated gradients or occlusion.

Check whether high-attribution wavelengths align with known molecular absorption regions. This does not prove causality, but it is a sanity check that the model is not relying mostly on global scaling shortcuts.

## Concrete code changes to make

1. **Dataloader**
   - Return `(spectrum, target, env_id, planet_index)` instead of only `(spectrum, target)`.
   - Build `env_id` from `summary.csv` using selected environment column/bin.

2. **Split builder**
   - Add `make_env_splits.py` that creates held-out-environment split files.
   - Save split metadata so every experiment is reproducible.

3. **Model**
   - Add `return_features=False` argument to `forward`.
   - Return pooled normalized feature vector before `self.head`.

4. **Training**
   - Add `--objective {erm, vrex, irm}`.
   - Add `--env-column`, `--lambda-rex`, `--lambda-irm`, `--warmup-epochs`.
   - Log per-environment losses every epoch.

5. **Evaluation**
   - Report random-val and held-out-env test separately.
   - Save per-gas and per-environment CSVs.

## Minimal experiment matrix

| Experiment | Objective | Split | Purpose |
|---|---:|---|---|
| E0 | ERM | random | reproduce current baseline |
| E1 | ERM | held-out `star_class` | measure shortcut vulnerability |
| E2 | V-REx | held-out `star_class` | first causal-invariance attempt |
| E3 | V-REx | held-out `surface_temperature_bin` | harder environment check |
| E4 | IRMv1 | best held-out split | compare with V-REx |
| E5 | V-REx + counterfactual augmentation | best split | test robustness gains |

Success criterion: not lower random validation loss; success is lower **worst-environment** and **held-out-environment** error without large degradation in average error.

## Recommended immediate next step

Start with `star_class` and `surface_temperature` environment splits, then implement V-REx. It is the best first intervention because it is simple, stable for regression, and directly testable against your existing training code.

## Sources

- Schölkopf et al., “Towards Causal Representation Learning,” 2021, arXiv:2102.11107 — https://arxiv.org/abs/2102.11107
- von Kügelgen, “Identifiable Causal Representation Learning,” 2024, arXiv:2406.13371 — https://arxiv.org/abs/2406.13371
- Peters et al., “Causal inference using invariant prediction,” 2015/2016, arXiv:1501.01332 — https://arxiv.org/abs/1501.01332
- Arjovsky et al., “Invariant Risk Minimization,” 2019, arXiv:1907.02893 — https://arxiv.org/abs/1907.02893
- Krueger et al., “Out-of-Distribution Generalization via Risk Extrapolation,” 2020/2021, arXiv:2003.00688 — https://arxiv.org/abs/2003.00688
- Rosenfeld et al., “The Risks of Invariant Risk Minimization,” 2020/2021, arXiv:2010.05761 — https://arxiv.org/abs/2010.05761
- Gulrajani & Lopez-Paz, “In Search of Lost Domain Generalization,” 2020/2021, arXiv:2007.01434 — https://arxiv.org/abs/2007.01434
- Pawlowski et al., “Deep Structural Causal Models for Tractable Counterfactual Inference,” 2020, arXiv:2006.06485 — https://arxiv.org/abs/2006.06485
- Nauta et al., “Causal Discovery with Attention-Based Convolutional Neural Networks,” 2019 — https://www.mdpi.com/2504-4990/1/1/19
