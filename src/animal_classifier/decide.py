"""The dominance rule: one image, one label (DESIGN.md §5.7).

This module is §5.7 transcribed, and it is deliberately the smallest interesting
file in the tree. It answers exactly one question — given the animal boxes and the
blur score, which single label does this image get — and it is the **only** place
in the system that compares one box's size with another's.

**Invariant I2, stated as a property of this file.** There is no ``min_box_area``,
no ``min_box_area_frac``, no ``min_animal_area`` and no absolute box-area floor
here or anywhere else. The only size comparison in the entire codebase is
:data:`dominance_ratio`'s, and it is purely *relative*: it asks whether one animal
is much bigger than the next, never whether an animal is big enough to bother
with. Two consequences the design wants, spelled out because they are the reason
the floor is excluded:

- A distant impala in the corner of a lion portrait does not make the image
  ``multiple``. The lion's box area is far more than 1.6x the impala's, so the
  photo is filed as ``lion``.
- A bird filling 0.1% of the frame with nothing else detected is filed as that
  bird's species. An area floor would have thrown it away, and E6 exists to keep
  that from being re-introduced.

Read the module and you can verify I2 by eye: ``dominance_ratio`` appears once,
``area_frac`` is never compared with a constant, and nothing is discarded.

**Three deliberate details that look like details and are not.**

1. The comparison is a **multiplication**, never a division. A degenerate
   zero-area second box makes the first dominant instead of raising
   ``ZeroDivisionError`` — E5 passes exactly that case.
2. The comparison is ``>=``, so an exact-ratio tie resolves to *dominant*, while
   two equal-area animals (ratio 1.0, below 1.6) resolve to ``multiple``.
3. A dominant box whose species is below the confidence gate is ``unknown``, **not**
   ``multiple``. The dominance question was answered; only the identity is
   uncertain, and conflating the two would file a confidently-single animal as if
   the frame were crowded.

**Blur can only produce ``junk`` in the zero-animal branch.** A blurry photo with
a detected animal is still classified: the detector firing is stronger evidence
than a hand-tuned sharpness threshold, so ``blur_threshold`` never overrides a
detection.

**Total, and never raises on data.** :func:`species_or_unknown` returns a member
of the closed set on every path — ``None`` is never compared with a float, and a
degenerate winner is ``unknown`` rather than a crash or an invented species (E26).
It cannot raise on a bad label either: ``species.slug`` was produced and validated
by :func:`animal_classifier.taxonomy.slug` when the model artifact was loaded
(§7.1), so a label-space problem is always a startup failure, never a mid-run one.
This function does not slugify at decision time.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from .detect import Box
from .errors import ConfigError
from .images import SPECIES_STATUS_DEGENERATE
from .taxonomy import RESERVED_LABELS

log = logging.getLogger(__name__)

#: The four non-species outcomes, named so no call site spells a bare string.
LABEL_MULTIPLE: Final = "multiple"
LABEL_LANDSCAPE: Final = "landscape"
LABEL_JUNK: Final = "junk"
LABEL_UNKNOWN: Final = "unknown"

#: Every label this module can return that is not a species slug.
NON_SPECIES_LABELS: Final = frozenset(
    {LABEL_MULTIPLE, LABEL_LANDSCAPE, LABEL_JUNK, LABEL_UNKNOWN}
)

# Checked at import: these four names are exactly the taxonomy's reserved set. If
# the two ever drift, a model class could shadow a pipeline outcome and `unknown/`
# would become ambiguous — a failure that is invisible until a real card produces
# the colliding label, which is why it is asserted here rather than trusted.
assert NON_SPECIES_LABELS == RESERVED_LABELS, (
    "decide.py's non-species labels must equal taxonomy.RESERVED_LABELS; "
    f"got {sorted(NON_SPECIES_LABELS)} vs {sorted(RESERVED_LABELS)}"
)


@dataclass(frozen=True, slots=True)
class SpeciesPrediction:
    """What the species classifier concluded about one crop.

    ``slug`` is already a legal label directory: it was produced by
    :func:`animal_classifier.taxonomy.slug` at artifact-load time (§7.1), which is
    why :func:`species_or_unknown` can return it directly without validating.
    """

    slug: str
    common: str
    scientific: str | None
    rank: str | None
    conf: float


@dataclass(frozen=True, slots=True)
class ScoredBox:
    """One detection plus whatever the classifier managed to say about it.

    ``species is None`` and ``species_status`` are the two ways a box can be
    unclassified, and they mean different things: ``None`` is *not classified*
    (no model loaded yet, or inference declined), while
    ``species_status == "degenerate"`` is *unclassifiable* because the crop was
    under 2 px a side (§5.3). Both land on ``unknown``, and **neither removes the
    box from the dominance count** — excluding a degenerate box would be an area
    floor by the back door.
    """

    box: Box
    species: SpeciesPrediction | None = None
    species_status: str | None = None

    @property
    def area_frac(self) -> float:
        """This box's share of the EXIF-transposed frame (§5.4)."""
        return self.box.area_frac

    @property
    def is_degenerate(self) -> bool:
        """Whether the crop was too small to classify (§5.3), not too small to count."""
        return self.species_status == SPECIES_STATUS_DEGENERATE

    @property
    def species_conf(self) -> float | None:
        return None if self.species is None else self.species.conf


@dataclass(frozen=True, slots=True)
class Decision:
    """The single label for one image, plus what justified it.

    ``dominant`` is the box that won dominance, or ``None`` when no box did —
    ``multiple``, ``landscape`` and ``junk`` have no winner. It is set even when
    the label is ``unknown``, because a low-confidence winner still *won*, and the
    catalog records that in ``boxes.is_dominant`` so the GUI can show which animal
    the decision was about.

    ``confidence`` is the dominant box's species confidence, and it is ``None``
    for every non-species label rather than a zero — a fabricated 0.0 would sort
    and display as a real measurement.
    """

    label: str
    confidence: float | None
    dominant: ScoredBox | None
    species_common: str | None = None
    species_scientific: str | None = None
    species_rank: str | None = None

    @property
    def is_species(self) -> bool:
        """Whether this label names a species rather than a pipeline outcome."""
        return self.label not in NON_SPECIES_LABELS


def species_or_unknown(box: ScoredBox, *, min_species_confidence: float) -> str:
    """This box's species slug, or ``unknown`` — total, and never raises (§5.7).

    The three ways to be ``unknown``, in the order §5.7 tests them: the crop was
    degenerate, the box was never classified, or it was classified below the
    confidence gate. Keeping the ``None`` test ahead of the comparison is what
    makes the function total — ``None < 0.45`` is a ``TypeError``, and it would
    fire mid-run on the first unclassified box.
    """
    if box.is_degenerate or box.species_conf is None:
        return LABEL_UNKNOWN
    if box.species_conf < min_species_confidence:
        return LABEL_UNKNOWN
    assert box.species is not None  # implied by species_conf being non-None
    return box.species.slug


def decide(
    boxes: Sequence[Box] | Sequence[ScoredBox],
    *,
    blur_score: float,
    dominance_ratio: float,
    min_species_confidence: float,
    blur_threshold: float,
) -> Decision:
    """One label for one image, from §5.7's rule and nothing else.

    ``boxes`` may be raw :class:`~animal_classifier.detect.Box` objects (no
    classifier yet — every animal is then ``unknown``) or :class:`ScoredBox`
    objects carrying predictions. Only ``cls == "animal"`` boxes participate:
    ``person`` and ``vehicle`` detections are stored and drawn but never label an
    image, so a frame containing only people is ``landscape``, or ``junk`` if it
    is also blurry (§5.4).
    """
    if not math.isfinite(blur_score):
        raise ValueError(
            f"blur_score {blur_score!r} is not finite; measure_blur always returns "
            "a finite variance, so a non-finite score means the caller invented one"
        )
    if dominance_ratio < 1.0:
        # Below 1.0 the rule inverts: the largest animal would be "dominant" even
        # when it is smaller than the runner-up, so every crowded frame would be
        # filed as a species. config.py already rejects this; re-checked here
        # because this function is the invariant's owner.
        raise ConfigError(
            f"dominance_ratio must be >= 1.0, got {dominance_ratio!r}; below 1.0 a "
            "smaller animal would out-rank a larger one"
        )

    scored = [
        entry if isinstance(entry, ScoredBox) else ScoredBox(box=entry)
        for entry in boxes
    ]
    animals = sorted(
        (entry for entry in scored if entry.box.is_animal),
        key=lambda entry: entry.area_frac,
        reverse=True,
    )

    if not animals:
        # The only branch blur can reach. Everything else has a detection, and a
        # detection outranks a sharpness heuristic.
        if blur_score < blur_threshold:
            log.debug(
                "no animal boxes and blur %.2f < %.2f: junk",
                blur_score,
                blur_threshold,
            )
            return Decision(label=LABEL_JUNK, confidence=None, dominant=None)
        log.debug(
            "no animal boxes and blur %.2f >= %.2f: landscape",
            blur_score,
            blur_threshold,
        )
        return Decision(label=LABEL_LANDSCAPE, confidence=None, dominant=None)

    if len(animals) == 1:
        return _species_decision(
            animals[0], min_species_confidence=min_species_confidence
        )

    # >= 2 animals: the ONLY size gate in the system. A multiplication, never a
    # division, so a zero-area runner-up makes the leader dominant instead of
    # raising; `>=` so an exact-ratio tie resolves to dominant.
    if animals[0].area_frac >= dominance_ratio * animals[1].area_frac:
        log.debug(
            "dominant: %.6g >= %.3f * %.6g",
            animals[0].area_frac,
            dominance_ratio,
            animals[1].area_frac,
        )
        return _species_decision(
            animals[0], min_species_confidence=min_species_confidence
        )
    log.debug(
        "no dominant animal: %.6g < %.3f * %.6g -> multiple",
        animals[0].area_frac,
        dominance_ratio,
        animals[1].area_frac,
    )
    return Decision(label=LABEL_MULTIPLE, confidence=None, dominant=None)


def _species_decision(
    winner: ScoredBox, *, min_species_confidence: float
) -> Decision:
    """Wrap :func:`species_or_unknown`'s verdict for the winning box.

    ``unknown`` keeps ``dominant=winner`` — the box did win — and carries **no
    species name**, because there is none the tool will stand behind.

    **But a *scored* ``unknown`` keeps its confidence** (§5.9). The distinction is
    load-bearing: NULL means "this label has no confidence by construction"
    (``landscape``, ``junk``, ``multiple``, or an ``unknown`` whose boxes were all
    degenerate), while a sub-threshold score is a *real measurement* that happens to
    be below the gate. Nulling it would throw away the one number that ranks
    "probably a zebra, 0.41" ahead of "no idea at all" — which is exactly what the
    GUI's review queue sorts on, and what makes the confidence slider able to see
    these rows at all.
    """
    label = species_or_unknown(winner, min_species_confidence=min_species_confidence)
    if label == LABEL_UNKNOWN:
        # winner.species_conf is None only when the box was never scored (no model,
        # or a degenerate crop); then NULL is correct by construction.
        return Decision(label=label, confidence=winner.species_conf, dominant=winner)
    assert winner.species is not None  # species_or_unknown returned a slug
    return Decision(
        label=label,
        confidence=winner.species.conf,
        dominant=winner,
        species_common=winner.species.common,
        species_scientific=winner.species.scientific,
        species_rank=winner.species.rank,
    )
