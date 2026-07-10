"""Empirical calibration of synthetic spectra to INARA's observable scale.

The engine produces spectrum *shapes* (arbitrary units). INARA's signals have a
specific absolute scale (erg/s/cm2). We calibrate each channel with a single
multiplicative factor so the synthetic batch's robust level (median of positive
values) matches INARA's, then the model's standard INARA standardization places
the spectra in-distribution. Molecular band structure is preserved (a global
scale doesn't change relative feature depths).

INARA reference medians below are MEASURED from data/inara_1by3 (read-only):
  planet_signal       p50 ~ 1.59e-22 erg/s/cm2
  star_planet_signal  p50 ~ 1.60e-19 erg/s/cm2
We never modify INARA data/stats — these are reference constants only.
"""
import numpy as np

INARA_PLANET_MEDIAN = 1.59e-22
INARA_STARPLANET_MEDIAN = 1.60e-19


def calibration_factor(shape_batch, ref_median):
    """Scalar k so that median(|k * shape|>0) == ref_median for this batch."""
    arr = np.abs(np.asarray(shape_batch))
    pos = arr[arr > 0]
    if pos.size == 0:
        return 1.0
    return float(ref_median / np.median(pos))
