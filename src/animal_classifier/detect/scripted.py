"""The scripted detector: boxes from a ``<image>.boxes.json`` sidecar (§5.4).

A **shipped** implementation, not test-only scaffolding, selected by the real flag
``--detector scripted``. The reason it ships is that the dominance rule (§5.7) is
the one decision the whole design turns on, and asserting it through a real
detector means asserting it through whatever geometry MegaDetector happens to
produce today. With a sidecar the e2e suite can state "these two boxes are in
exactly a 1.59 ratio" and pin the outcome, so E5's boundary cases and E6's
no-area-floor cases test the *rule* rather than the model.

Sidecar format — a JSON array beside the image, named ``<image name>.boxes.json``
(so ``DCIM/lion.jpg`` → ``DCIM/lion.jpg.boxes.json``):

.. code-block:: json

    [
      {"cls": "animal", "conf": 0.91, "x0": 100, "y0": 100, "x1": 500, "y1": 400},
      {"cls": "person", "conf": 0.77, "x0": 10,  "y0": 10,  "x1": 60,  "y1": 120}
    ]

Coordinates are pixels in the **EXIF-transposed** frame (invariant I9), the same
frame MegaDetector reports in, so a fixture written for this detector describes
the same geometry the real one would. ``area_frac`` is never written in the
sidecar: it is derived from the decoded frame, because a hand-written value could
disagree with the coordinates and then the catalog would contain two
contradictory accounts of one box.

**An absent sidecar means no detections**, which is how the animal-free
``landscape``/``junk`` fixtures are expressed without a file full of nothing. A
sidecar that exists but cannot be read or does not match the schema is **fatal**
(``ConfigError``, exit 3): a malformed fixture must not quietly become "no animals
here" and let a dominance test pass for the wrong reason. That asymmetry is the
whole validation policy of this module.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from ..errors import ConfigError
from ..images import DecodedImage
from .base import Box, BoxClass, box_from_pixels

log = logging.getLogger(__name__)

#: Appended to the image's **full** file name, not swapped for its suffix, so
#: ``a.jpg`` and ``a.png`` in one directory cannot share a sidecar.
SIDECAR_SUFFIX: Final = ".boxes.json"

#: Exactly the keys one sidecar box may carry. A closed set, so a typo like
#: ``"class"`` or ``"confidence"`` fails loudly instead of being ignored and
#: silently defaulted — the failure mode that would make a fixture assert nothing.
_REQUIRED_KEYS: Final = frozenset({"cls", "conf", "x0", "y0", "x1", "y1"})


def sidecar_for(image: Path) -> Path:
    """The sidecar path for an image (``a.jpg`` → ``a.jpg.boxes.json``)."""
    return image.with_name(image.name + SIDECAR_SUFFIX)


class ScriptedDetector:
    """Returns exactly the boxes a sidecar names — satisfies ``Detector`` (§5.4).

    Stateless apart from its ``model_id``, and it holds no model, so it is cheap to
    construct and safe to reuse across a whole run.
    """

    #: ``images.model_id`` for every image this detector labels. Names itself
    #: explicitly so a catalog written with ``--detector scripted`` can never be
    #: mistaken for one written by the real MegaDetector weights.
    MODEL_ID: Final = "scripted:boxes.json"

    def __init__(self, *, confidence: float = 0.0) -> None:
        """``confidence`` mirrors ``--detector-confidence``.

        Applied as a genuine threshold so the flag means the same thing for both
        backends, and defaulted to ``0.0`` — keep every box a fixture names —
        because a fixture stating a box exists is an instruction, and silently
        dropping it would make a dominance test assert the wrong geometry.
        """
        if not 0.0 <= confidence <= 1.0:
            raise ConfigError(
                f"detector_confidence must be in [0.0, 1.0], got {confidence!r}"
            )
        self._confidence = float(confidence)

    @property
    def model_id(self) -> str:
        return self.MODEL_ID

    def detect(self, decoded: DecodedImage) -> Sequence[Box]:
        """Boxes from ``<image>.boxes.json``, or none if there is no sidecar."""
        sidecar = sidecar_for(decoded.path)
        if not sidecar.is_file():
            log.debug(
                "%s: no %s sidecar; scripted detector reports no detections",
                decoded.path,
                SIDECAR_SUFFIX,
            )
            return ()

        entries = _load_sidecar(sidecar)
        boxes: list[Box] = []
        for position, entry in enumerate(entries):
            box = _box_from_entry(
                entry,
                sidecar=sidecar,
                position=position,
                frame_width=decoded.width,
                frame_height=decoded.height,
            )
            if box.conf < self._confidence:
                log.debug(
                    "%s box %d: conf %.3f below --detector-confidence %.3f; dropped",
                    sidecar,
                    position,
                    box.conf,
                    self._confidence,
                )
                continue
            boxes.append(box)
        log.debug("%s: %d box(es) from %s", decoded.path, len(boxes), sidecar.name)
        return tuple(boxes)


def _load_sidecar(sidecar: Path) -> list[Any]:
    """Read and shape-check one sidecar, fatally on any problem."""
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(
            f"cannot read scripted-detector sidecar {sidecar}: "
            f"{type(error).__name__}: {error.strerror or error}"
        ) from error
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise ConfigError(
            f"scripted-detector sidecar {sidecar} is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, list):
        raise ConfigError(
            f"scripted-detector sidecar {sidecar} must be a JSON array of boxes, "
            f"got {type(payload).__name__}"
        )
    return payload


def _box_from_entry(
    entry: object,
    *,
    sidecar: Path,
    position: int,
    frame_width: int,
    frame_height: int,
) -> Box:
    """One sidecar entry → :class:`Box`, naming the sidecar and index on failure.

    Every rejection carries ``sidecar`` and ``position`` because the alternative —
    "invalid box" with no locator — is useless when a fixture card has thirty
    sidecars.
    """
    where = f"{sidecar} box {position}"
    if not isinstance(entry, dict):
        raise ConfigError(f"{where} must be a JSON object, got {type(entry).__name__}")

    keys = set(entry)
    missing = _REQUIRED_KEYS - keys
    unexpected = keys - _REQUIRED_KEYS
    if missing or unexpected:
        problems = []
        if missing:
            problems.append(f"missing {sorted(missing)}")
        if unexpected:
            problems.append(f"unexpected {sorted(unexpected)}")
        raise ConfigError(
            f"{where} has the wrong keys ({'; '.join(problems)}); "
            f"each box needs exactly {sorted(_REQUIRED_KEYS)}"
        )

    numbers: dict[str, float] = {}
    for key in ("conf", "x0", "y0", "x1", "y1"):
        value = entry[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(
                f"{where} field {key!r} must be a number, got {value!r}"
            )
        numbers[key] = float(value)

    try:
        return box_from_pixels(
            box_class=BoxClass(str(entry["cls"])),
            conf=numbers["conf"],
            x0=numbers["x0"],
            y0=numbers["y0"],
            x1=numbers["x1"],
            y1=numbers["y1"],
            frame_width=frame_width,
            frame_height=frame_height,
        )
    except ValueError as error:
        # box_from_pixels' own validation, re-raised as the fatal config class: a
        # bad sidecar is a bad input file, not a detector malfunction.
        raise ConfigError(f"{where} is invalid: {error}") from error
