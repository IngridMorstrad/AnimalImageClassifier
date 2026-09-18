"""Training manifests and the one true split (DESIGN.md §7.2).

A manifest is JSONL, one sample per line: ``path`` (absolute or manifest-relative),
``label``, optional ``box`` (crops on load), optional ``split``. Every line is
validated and the first bad one is fatal with its line number and offending field —
a training run must never quietly train on a truncated manifest, and a missing image
is fatal, not skipped.

:func:`split_for` is the single source of truth for the train/val split, keyed on
the **basename** so the split is identical wherever ``data/raw`` was unpacked and so
every crop of one photograph lands on the same side (that is what prevents leakage,
not the hash itself). An explicit ``split`` field always wins.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..errors import ConfigError


@dataclass(frozen=True, slots=True)
class Sample:
    """One training sample: an image, its label, and an optional crop box."""

    path: Path
    label: str
    box: tuple[float, float, float, float] | None
    split: str


def split_for(path: str) -> str:
    """'val' for one image in five, deterministically and machine-independently."""
    key = os.path.basename(path)
    return "val" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def load_manifest(
    manifest_path: Path, *, split: str | None = None, require_images: bool = True
) -> list[Sample]:
    """Parse and validate a manifest, optionally filtered to one split (§7.2).

    ``require_images=False`` is for callers that only need the label space (e.g.
    determining the class list) without touching the pixels.
    """
    if not manifest_path.is_file():
        raise ConfigError(f"manifest not found: {manifest_path}")
    base = manifest_path.parent
    samples: list[Sample] = []
    with manifest_path.open(encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            samples.append(_parse_line(line, lineno, base, require_images))
    if not samples:
        raise ConfigError(f"manifest {manifest_path} has no samples")
    if split is not None:
        samples = [s for s in samples if s.split == split]
        if not samples:
            raise ConfigError(
                f"manifest {manifest_path} has no samples in split {split!r}"
            )
    return samples


def _parse_line(line: str, lineno: int, base: Path, require_images: bool) -> Sample:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as error:
        raise ConfigError(f"manifest line {lineno} is not valid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ConfigError(f"manifest line {lineno} is not a JSON object")

    for field in ("path", "label"):
        if field not in record or not str(record[field]).strip():
            raise ConfigError(f"manifest line {lineno} is missing required field {field!r}")

    raw_path = Path(str(record["path"]))
    path = raw_path if raw_path.is_absolute() else (base / raw_path)
    if require_images and not path.is_file():
        raise ConfigError(f"manifest line {lineno}: image not found: {path}")

    box = _parse_box(record.get("box"), lineno)
    split = str(record["split"]) if record.get("split") else split_for(str(record["path"]))
    if split not in ("train", "val"):
        raise ConfigError(
            f"manifest line {lineno}: split must be 'train' or 'val', got {split!r}"
        )
    return Sample(path=path, label=str(record["label"]), box=box, split=split)


def _parse_box(raw: object, lineno: int) -> tuple[float, float, float, float] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        raise ConfigError(
            f"manifest line {lineno}: box must be [x0, y0, x1, y1], got {raw!r}"
        )
    try:
        x0, y0, x1, y1 = (float(v) for v in raw)
    except (TypeError, ValueError) as error:
        raise ConfigError(f"manifest line {lineno}: box has non-numeric values: {raw!r}") from error
    if x1 < x0 or y1 < y0:
        raise ConfigError(f"manifest line {lineno}: box is inverted: {raw!r}")
    return (x0, y0, x1, y1)


def class_list(samples: list[Sample]) -> list[str]:
    """The sorted unique labels in ``samples`` — the model's class order (§7.2)."""
    return sorted({s.label for s in samples})
