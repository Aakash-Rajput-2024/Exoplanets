"""Probe / provision petitRADTRANS correlated-k opacities for our species.

Pass 1 (default): list which line lists are available per species (answers the
O2/O3 availability question) by capturing pRT's printed file list before its
interactive selection prompt. Pass 2 (--download): set a default file per species
and instantiate Radtrans to trigger the download.

Run in .venv_prt.
"""
import sys
import io
import re
import os
import contextlib
import numpy as np

from petitRADTRANS.config import petitradtrans_config_parser as cfg
from petitRADTRANS.radtrans import Radtrans

SPECIES = ["H2O", "CO2", "CH4", "O2", "O3"]
IDP = cfg.get_input_data_path()


def probe(sp):
    """Return (status, isodir, files[]) for a species without committing a download
    when multiple files exist (the prompt EOFs first)."""
    buf = io.StringIO()
    status = "built"
    try:
        with contextlib.redirect_stdout(buf):
            Radtrans(pressures=np.logspace(-3, 1, 10),
                     wavelength_boundaries=[0.3, 2.0],
                     line_species=[sp], line_opacity_mode='c-k',
                     scattering_in_emission=False)
    except EOFError:
        status = "multi(prompt)"
    except Exception as e:
        status = f"{type(e).__name__}: {str(e)[:120]}"
    out = buf.getvalue()
    files = re.findall(r'^\s*\d+:\s+(\S+\.h5)', out, re.M)
    m = re.search(r"correlated_k/%s/([^/'\s]+)" % re.escape(sp), out)
    return status, (m.group(1) if m else None), files


if __name__ == "__main__":
    for sp in SPECIES:
        status, isodir, files = probe(sp)
        print(f"\n=== {sp} === status={status} isodir={isodir}")
        for f in files:
            print(f"    {f}")
