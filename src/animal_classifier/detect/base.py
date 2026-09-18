"""The ``Detector`` protocol and the frozen ``Box`` (DESIGN.md §5.4).

**One frame, and this type is where it is enforced.** Every box is in the
EXIF-transposed full-resolution pixel frame that
:func:`animal_classifier.images.decode` established (invariant I9). The raw stored
raster's dimensions exist nowhere in this system, so a detector that reports in
them is a defect, not a variant — :func:`box_from_pixels` takes the frame size it
is normalising against and refuses anything that cannot be a box in that frame.

**Coordinates must be finite, and that is checked here.** A NaN coordinate is not
a merely odd box: NaN loses every comparison, so clipping it silently yields the
whole frame, and the resulting ``area_frac`` is meaningless while looking
perfectly ordinary in the catalog. :func:`animal_classifier.images.crop` learned
this the hard way (follow-up F22) and now raises; this module states the same
contract one layer earlier, where a malfunctioning backend is caught before its
numbers reach the dominance rule.

**Only ``animal`` boxes label an image.** ``person`` and ``vehicle`` boxes are
real detections — stored, counted, drawn in the GUI — but they never create a
label, so an image containing only people is ``landscape`` (or ``junk`` if
blurry). §5.4 requires that surprise to be documented user-facing, which
``classify --help`` and ``README.md`` both do. :func:`animal_boxes` is the one
filter the pipeline and §5.7 share, so the rule cannot drift between them.

**No area floor lives here.** A box's size is never a reason to discard it
(invariant I2). :func:`box_from_pixels` computes ``area_frac`` for the GUI's
explainability and for §5.7's *relative* comparison, and nothing in this module
compares it against a constant.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, runtime_checkable

from ..images import DecodedImage


class BoxClass(enum.StrEnum):
    """MegaDetector v5a's three output classes (§5.4).

    A closed set: the checkpoint emits exactly these, and a backend that invents a
    fourth is misconfigured rather than interesting, so
    :func:`box_from_pixels` rejects anything else by name.
    """

    ANIMAL = "animal"
    PERSON = "person"
    VEHICLE = "vehicle"


#: The only class that produces a label (§5.4). Exported so no caller spells the
#: string, because a typo would silently stop every animal from being labelled.
ANIMAL_CLASS: Final = BoxClass.ANIMAL


@dataclass(frozen=True, slots=True)
class Box:
    """One detection, in the EXIF-transposed frame, with its share of that frame.

    ``cls`` keeps §5.4's field name. ``area_frac`` is
    ``((x1-x0)*(y1-y0)) / (W*H)`` against the *same* transposed ``W``/``H`` the
    coordinates are in — it is stored for the GUI's audit trail and it is the only
    quantity §5.7 compares, always against another box's, never against a
    threshold.

    Frozen, because a box is evidence: once the detector has reported it, the crop
    step, the decision and the catalog row must all be describing the same
    rectangle.
    """

    cls: BoxClass
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float
    area_frac: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """The box as :func:`animal_classifier.images.crop` wants it."""
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def is_animal(self) -> bool:
        """Whether this box can produce a label (§5.4)."""
        return self.cls is BoxClass.ANIMAL


def box_from_pixels(
    *,
    box_class: BoxClass | str,
    conf: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    frame_width: int,
    frame_height: int,
) -> Box:
    """Build a :class:`Box`, normalising ``area_frac`` against the given frame.

    The frame size is a required argument rather than a default or a global,
    because ``area_frac`` is only meaningful relative to the exact raster the
    coordinates came from. Passing the pre-transpose size here is the one mistake
    that would corrupt every dominance decision while leaving the catalog looking
    plausible, so the caller is made to name it.

    Total validation, all of it fatal rather than clamped (I7): the class must be
    one of §5.4's three, the confidence must be a finite ``[0, 1]``, coordinates
    must be finite and ordered (``x0 <= x1``), and the frame must have positive
    area. Out-of-frame coordinates are *not* rejected — clipping is
    :func:`animal_classifier.images.crop`'s job and a detector reporting slightly
    outside the frame is normal — but they are still normalised against the true
    frame area, so ``area_frac`` can exceed 1.0 only if the box genuinely exceeds
    the frame, which the GUI shows rather than hides.
    """
    try:
        kind = BoxClass(box_class)
    except ValueError as error:
        raise ValueError(
            f"detection class {box_class!r} is not one of "
            f"{[str(member) for member in BoxClass]}"
        ) from error

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError(
            f"frame is {frame_width}x{frame_height}; area_frac is undefined "
            "without a positive frame area"
        )

    coordinates = (float(x0), float(y0), float(x1), float(y1))
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(
            f"detection box {coordinates} has a non-finite coordinate; boxes must "
            "be finite pixel coordinates in the EXIF-transposed frame"
        )
    if not math.isfinite(float(conf)):
        raise ValueError(f"detection confidence {conf!r} is not finite")
    if not 0.0 <= float(conf) <= 1.0:
        raise ValueError(f"detection confidence {conf!r} is outside [0.0, 1.0]")

    left, top, right, bottom = coordinates
    if right < left or bottom < top:
        raise ValueError(
            f"detection box {coordinates} is inverted; expected x0 <= x1 and "
            "y0 <= y1 in (x0, y0, x1, y1) order"
        )

    area_frac = ((right - left) * (bottom - top)) / float(
        frame_width * frame_height
    )
    return Box(
        cls=kind,
        conf=float(conf),
        x0=left,
        y0=top,
        x1=right,
        y1=bottom,
        area_frac=area_frac,
    )


def animal_boxes(boxes: Iterable[Box]) -> tuple[Box, ...]:
    """The subset of ``boxes`` that can produce a label (§5.4).

    Order is preserved, so a caller that needs the detector's own box indices for
    ``boxes.idx`` can still recover them by identity. The filter exists once,
    here, because §5.4 (what is stored) and §5.7 (what is labelled) disagreeing
    about which boxes are animals would be invisible in both.
    """
    return tuple(box for box in boxes if box.is_animal)


@runtime_checkable
class Detector(Protocol):
    """What the pipeline requires of a detection backend (§5.4).

    Deliberately tiny. ``detect`` receives the whole
    :class:`~animal_classifier.images.DecodedImage` rather than a bare array
    because a backend needs the frame size to normalise ``area_frac`` and
    :class:`~animal_classifier.detect.scripted.ScriptedDetector` needs the source
    path to find its sidecar — and because passing the decoded object keeps the
    single-frame guarantee intact instead of re-deriving dimensions downstream.

    ``model_id`` is recorded in ``images.model_id`` for every image, so a result
    can always be attributed to the exact detector that produced it. It is a
    required part of the protocol, not an optional nicety: a catalog row whose
    provenance is unknown cannot be re-examined later.
    """

    @property
    def model_id(self) -> str:
        """Stable identifier for this backend and its weights (``images.model_id``)."""
        ...

    def detect(self, decoded: DecodedImage) -> Sequence[Box]:
        """Boxes for one image, in the EXIF-transposed frame.

        Returns an empty sequence for an image with no detections — that is the
        ``landscape``/``junk`` case (§5.7), not an error.
        """
        ...
