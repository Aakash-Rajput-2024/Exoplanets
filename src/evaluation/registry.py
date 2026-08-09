"""Suite registry — the ordered A…K sections and selector resolution.

``run_eval`` iterates SUITES in order; each is gated by its own ``applicable(ctx)`` so a
missing input (no cross-gen cache, no PSG anchor, …) is recorded as *skipped*, never a
crash. Select by section letter (``A``), slug (``in_distribution``), or ``all``.
"""
from __future__ import annotations

from evaluation.suites import (
    in_distribution, baselines, cross_generator, psg_anchor, solar_system,
    real_earth, transiting_ood, published_retrieval, calibration, ood_honesty,
    bayes_reference,
)

# Report/run order = the narrative order of the sections.
SUITES = [
    in_distribution, baselines, cross_generator, psg_anchor, solar_system,
    real_earth, transiting_ood, published_retrieval, calibration, ood_honesty,
    bayes_reference,
]

BY_SECTION = {m.SECTION: m for m in SUITES}
BY_SLUG = {m.SUITE: m for m in SUITES}


def resolve(selectors):
    """selectors: iterable of section letters / slugs / 'all' (case-insensitive).
    Returns suite modules in registry order, de-duplicated."""
    if not selectors or any(str(s).lower() == "all" for s in selectors):
        return list(SUITES)
    want = set()
    for s in selectors:
        key = str(s).strip()
        if key.upper() in BY_SECTION:
            want.add(BY_SECTION[key.upper()])
        elif key in BY_SLUG:
            want.add(BY_SLUG[key])
        else:
            raise SystemExit(f"unknown suite selector {key!r}; "
                             f"choose section letters {sorted(BY_SECTION)} or slugs {sorted(BY_SLUG)}")
    return [m for m in SUITES if m in want]
