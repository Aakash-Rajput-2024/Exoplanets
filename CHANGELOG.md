# CHANGELOG

## 2026-06-05 — Feynman autoresearch causality loop started

- Created `feynmanresearch/` workspace with `autoresearch.md`, `autoresearch.sh`, and `autoresearch.jsonl`.
- Selected local execution with `/Users/aakashrajput/MachineLearning/venvNLP`.
- Baseline evaluation initially failed due missing `torchinfo`; installed `torchinfo==1.8.0` in the selected venv.
- Baseline evaluation then failed on MPS/CPU mismatch caused by `torchinfo.summary(..., device="cpu")`; added CPU-only wrapper in `feynmanresearch/run_transformer_test_cpu.py` without editing source code.
- Baseline evaluation succeeded: Average R2 `0.7179`, Average RMSE `0.0229`, Average MAE `0.0173` from `src/transformerarch/details.txt`.
- Ran `feynmanresearch/causal_mask_probe.py` on 512 validation samples. Full-sequence RMSE `0.024216`; keeping 75%, 50%, and 25% of the wavelength sequence gave RMSE `0.043381`, `0.053951`, and `0.066217` respectively.
- Next step: implement no-source-edit environment split evaluation for worst-environment RMSE using `data/summary.csv` nuisance variables.

## 2026-06-05 — Environment RMSE probe

- Added and ran `feynmanresearch/environment_rmse_probe.py` without editing source code.
- On 1024 reconstructed validation samples: overall RMSE `0.023069`; worst tested environment was `distance_parsec` `(11.532, 15.0]` with RMSE `0.025660`, gap `+0.002591`.
- Caveat: validation membership was reconstructed from current file order and seed; next rigorous step is saving explicit environment split indices.
