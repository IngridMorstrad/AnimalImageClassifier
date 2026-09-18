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

# MegaDetector is intentionally *not* imported eagerly: it pulls in torch and
# yolov5, a multi-second import, and `classify --detector scripted`, the GUI and
# the tests must not pay that cost. `pipeline.build_detector` imports it lazily
# only when `--detector megadetector` is actually selected.
__all__ = [
    "ANIMAL_CLASS",
    "SIDECAR_SUFFIX",
    "Box",
    "BoxClass",
    "Detector",
    "MegaDetector",
    "ScriptedDetector",
    "animal_boxes",
    "box_from_pixels",
]


def __getattr__(name: str) -> object:
    """Lazily expose :class:`MegaDetector` without importing torch at package load."""
    if name == "MegaDetector":
        from .megadetector import MegaDetector  # noqa: PLC0415

        return MegaDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
