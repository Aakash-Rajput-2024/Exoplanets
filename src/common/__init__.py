"""Shared data/eval layer for the INARA retrieval pipeline.

Centralizes the pieces that were previously copy-pasted across every track
(dataloaders, label/input transforms, metrics, provenance), so a fix lands once
instead of drifting between five near-identical files. Introduced while
resolving the ULTRAPLAN review; grows one module per critical issue.
"""
