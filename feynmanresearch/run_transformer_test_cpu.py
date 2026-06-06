#!/usr/bin/env python
"""Run src/transformerarch/test.py on CPU without editing project code.

Reason: test.py calls torchinfo.summary(..., device="cpu") after moving the model
to MPS, which can leave model weights on CPU while inputs remain on MPS. This
wrapper disables MPS for the test run only.
"""
import os
import runpy
import sys

import torch

# Disable MPS device selection for this process only.
torch.backends.mps.is_available = lambda: False

repo = "/Users/aakashrajput/MachineLearning/Exoplanets"
os.chdir(repo)
sys.path.insert(0, os.path.join(repo, "src", "transformerarch"))

runpy.run_path(os.path.join(repo, "src", "transformerarch", "test.py"), run_name="__main__")
