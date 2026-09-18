"""Detection: bounding boxes in the one coordinate frame (DESIGN.md §5.4).

Two things live behind this package boundary, and keeping them apart is the point:

- :class:`~animal_classifier.detect.base.Detector` — the protocol every backend
  satisfies, so the pipeline never names a concrete detector.
- :class:`~animal_classifier.detect.base.Box` — the frozen result type, in the
  **EXIF-transposed** frame :func:`animal_classifier.images.decode` established
  (invariant I9), carrying the ``area_frac`` that §5.7's dominance rule compares.

Import from this package, not from its modules.
"""

from __future__ import annotations

from .base import (
    ANIMAL_CLASS,
    Box,
    BoxClass,
    Detector,
    animal_boxes,
    box_from_pixels,
)
from .scripted import SIDECAR_SUFFIX, ScriptedDetector

__all__ = [
    "ANIMAL_CLASS",
    "SIDECAR_SUFFIX",
    "Box",
    "BoxClass",
    "Detector",
    "ScriptedDetector",
    "animal_boxes",
    "box_from_pixels",
]
