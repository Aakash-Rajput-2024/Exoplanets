"""INARA wavelength grid utilities.

Every trained model in this repo expects spectra sampled on INARA's exact
wavelength axis, filtered to <= 2.0 um (length 4379). Cross-generator spectra
must be interpolated onto this same axis before they can be fed to a model.
"""
import os
import glob
import numpy as np
import pandas as pd

REPO_ROOT = "/Users/aakashrajput/MachineLearning/Exoplanets"
INARA_SPECTRA_DIR = os.path.join(REPO_ROOT, "data", "inara_1by3")
WL_MAX_UM = 2.0


def load_inara_grid(spectra_dir=INARA_SPECTRA_DIR, wl_max=WL_MAX_UM):
    """Return INARA's wavelength axis in microns, filtered to <= wl_max.

    Read from a sample spectrum CSV so the grid is byte-identical to what the
    training dataloader used (see dataloader.py: spectrum_df[... <= 2.0]).
    """
    files = sorted(glob.glob(os.path.join(spectra_dir, "*.csv")))
    if not files:
        raise FileNotFoundError(f"No INARA spectra found in {spectra_dir}")
    df = pd.read_csv(files[0])
    wl = df.loc[df["wavelength_(um)"] <= wl_max, "wavelength_(um)"].to_numpy(dtype=float)
    return wl
