"""Checkpoint provenance stamping (fixes C6).

A trained checkpoint should answer, months later and without guesswork, the
question a referee always asks: *exactly what code and hyperparameters produced
these weights?* We therefore record, inside every checkpoint:

  - ``config``       : the fully resolved hyperparameter/data dict,
  - ``config_hash``  : a short deterministic digest of that dict (equality
                       checks, run naming, spotting silent config drift),
  - ``git_commit``   : the code version (``-dirty`` if the tree had
                       uncommitted changes),
  - ``saved_at``     : a UTC timestamp.

Pure standard library so it is safe to import from any training environment.
None of these helpers raise on the caller's behalf: a missing git binary or an
unserializable config degrades to a sentinel rather than crashing a 50-epoch
run at checkpoint time.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import subprocess


def git_commit(repo_dir: str | None = None) -> str:
    """Return the current commit hash (with a ``-dirty`` suffix if the working
    tree has uncommitted changes), or ``"unknown"`` if git is unavailable."""
    cwd = repo_dir or os.getcwd()
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        if head.returncode != 0:
            return "unknown"
        commit = head.stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd, capture_output=True, text=True, timeout=5,
        )
        dirty = bool(status.stdout.strip())
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def config_hash(config: dict) -> str:
    """Deterministic 12-hex-char digest of a config dict.

    Sorted keys make the hash order-invariant; ``default=str`` tolerates values
    (e.g. tuples, paths) that are not natively JSON-serializable.
    """
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def stamp_checkpoint(state: dict, config: dict, repo_dir: str | None = None) -> dict:
    """Return a copy of ``state`` augmented with provenance metadata.

    The original dict is not mutated. Intended to wrap the state passed to
    ``torch.save`` at every checkpoint write.
    """
    stamped = dict(state)
    stamped["config"] = config
    stamped["config_hash"] = config_hash(config)
    stamped["git_commit"] = git_commit(repo_dir)
    stamped["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    return stamped
