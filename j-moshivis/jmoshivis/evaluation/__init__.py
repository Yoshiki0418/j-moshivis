"""Reproducible image-understanding evaluation for J-MoshiVis.

The package deliberately separates benchmark preparation, cached TTS,
GPU inference, and CPU-only scoring.  This keeps expensive model outputs
reusable when answer normalisation or benchmark subsets change.
"""

from .schema import EvalExample, ManifestError, read_manifest, write_manifest

__all__ = ["EvalExample", "ManifestError", "read_manifest", "write_manifest"]
