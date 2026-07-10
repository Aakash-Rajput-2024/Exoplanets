"""OPTIONAL, ONLINE benchmark fetcher — NOT imported by the core pipeline.

The evaluation pipeline runs fully offline on what is already on disk. This script is a
convenience for a user WITH network access who wants to populate the optional real-spectrum
directories the OOD probe (Section G) and real-Earth extensions (Section F) can pick up. It
performs NO work on import and is never a runtime dependency of run_eval.

Real spectra are hosted by the sources below and are best downloaded manually (they change
URLs, need registration, or have licenses to acknowledge). This script only prints where to
get them and, if a direct URL is given, saves it to the right place.

Sources (credible, cited in the benchmark JSONs):
  * NASA Exoplanet Archive (TAP) — target/stellar/planetary params:
      https://exoplanetarchive.ipac.caltech.edu/TAP
  * MAST (JWST/HST reduced spectra) — e.g. WASP-39b ERS products:
      https://mast.stsci.edu   (transiting → Section G OOD probe only)
  * VPL spectral library (Earth/solar-system reflected spectra):
      https://depts.washington.edu/naivpl/content/models
  * EPOXI disk-integrated Earth (Livengood et al. 2011):  PDS Small Bodies Node
  * Galileo disk-integrated Earth (Sagan et al. 1993):    NASA PDS
  * Ariel ABC database (synthetic transmission, 7-param): https://www.ariel-datachallenge.space

Target directories (create + drop files in; the suites auto-detect):
  data/real_spectra/*.{txt,csv,dat}   — (wavelength_µm, depth/flux[, err]) for Section G
  data/real_earth/*.dat               — additional disk-integrated Earth spectra for Section F
"""
from __future__ import annotations

import argparse
import os
import sys

REPO = "/Users/aakashrajput/MachineLearning/Exoplanets"
REAL_SPECTRA = os.path.join(REPO, "data", "real_spectra")
REAL_EARTH = os.path.join(REPO, "data", "real_earth")

SOURCES = {
    "nasa_exoplanet_archive_tap": "https://exoplanetarchive.ipac.caltech.edu/TAP",
    "mast_jwst": "https://mast.stsci.edu",
    "vpl_models": "https://depts.washington.edu/naivpl/content/models",
    "ariel_abc": "https://www.ariel-datachallenge.space",
}


def _download(url, dest):
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print("done")


def main():
    ap = argparse.ArgumentParser(description="Optional online benchmark fetcher (offline core "
                                             "does not need this).")
    ap.add_argument("--url", help="direct URL to a spectrum file")
    ap.add_argument("--dest", help="destination path (e.g. data/real_spectra/wasp39b.txt)")
    ap.add_argument("--list", action="store_true", help="print sources and exit")
    a = ap.parse_args()
    os.makedirs(REAL_SPECTRA, exist_ok=True)
    if a.list or not a.url:
        print(__doc__)
        for k, v in SOURCES.items():
            print(f"  {k:32s} {v}")
        print(f"\nDrop transiting spectra into {REAL_SPECTRA} (Section G) and disk-integrated "
              f"Earth into {REAL_EARTH} (Section F).")
        return
    if not a.dest:
        sys.exit("--dest required with --url")
    _download(a.url, os.path.join(REPO, a.dest) if not os.path.isabs(a.dest) else a.dest)


if __name__ == "__main__":
    main()
