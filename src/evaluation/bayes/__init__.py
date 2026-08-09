"""Bayesian retrieval baselines — the reference the neural tracks are measured against.

The audit (junk/AUDIT_REPORT.md:95) records "no classical/Bayesian baseline" as a High
finding: without one, a given R² is unattributable between *information content* (the
spectra do not carry the answer) and *model capacity* (the net fails to extract it).

The tiers, cheapest and most defensible first:

  T0  prior_is    likelihood-weighted importance sampling over the INARA training
                  library. The 88k training spectra ARE real PSG outputs drawn from
                  the INARA prior, so this is an asymptotically EXACT posterior — the
                  correct prior, the exact forward model, no approximation. Valid only
                  while the effective sample size stays ≫1 (measured: α ≲ 3).
  T1  ../models/npe   amortized neural posterior estimation (approximate, must pass SBC).
  T2  emulator + nested   a learned PSG surrogate sampled over arbitrary θ — the only
                  tier that reaches the ceiling at high α.
  T3  real radiative transfer (petitRADTRANS) for cross-code agreement.

Everything here works in CONTRAST space (``common.observable.contrast``), upstream of
the asinh normalizer and the CLR label pipeline, so the inner loop is pure tensor math
and needs neither a trained model nor the encoding statistics.
"""
