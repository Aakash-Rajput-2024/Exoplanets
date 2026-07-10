"""Unified evaluation pipeline for the INARA reflected-light retrieval models.

Sections A–J (in-distribution, baselines, cross-generator, PSG anchor, solar-system,
real Earth, transiting OOD probe, published-retrieval, calibration, OOD honesty), each
scored through the shared v2 core (``evaluation.core``) so every number is consistent with
training. Run everything for a checkpoint with::

    python -m evaluation.run_eval <track> --suites all

See ``evaluation/README.md`` for the section map and the offline/credible-source design.
"""
