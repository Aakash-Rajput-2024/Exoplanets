"""Evaluation suites — one module per report section (A…J).

Each suite exposes:
  SECTION, SUITE, TITLE, EPISTEMIC   — identity/metadata for the report
  applicable(ctx) -> (bool, reason)  — offline gate (inputs present?)
  run(ctx, **kw)  -> core.SuiteResult

Suites never train or mutate INARA data; they read checkpoints/caches read-only and
write only under src/evaluation/results/<track>/<suite>/.
"""
