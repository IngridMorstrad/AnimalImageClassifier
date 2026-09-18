#!/usr/bin/env python
"""Build the fixture SD-card tree the e2e suite drives ``classify`` over.

Synthetic by construction, and deliberately so. §11's fixture card needs exact
control over two things — box geometry (for §5.7's boundaries) and sharpness (for
the ``landscape``/``junk`` split) — and a real photograph gives neither: whatever
MegaDetector happens to report today would become the expectation, so a test would
be asserting the model rather than the rule. Generated images plus
``.boxes.json`` sidecars make every dominance case exactly stateable, which is the
entire reason ``--detector scripted`` ships (§5.4).

Every image here is also a real file in a real format, decoded by the same
``images.decode`` path a card goes through: PNG, JPEG, TIFF and HEIC are written by
Pillow (HEIC through ``pillow-heif``'s encoder, which is why registering the opener
at import matters), so format handling is exercised rather than mocked.

Run standalone to inspect the tree:

    uv run --frozen python scripts/make_e2e_fixtures.py /tmp/card
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pillow_heif
from PIL import Image, ImageFilter

pillow_heif.register_heif_opener()

#: Frame size for every generated photo. Big enough that a 0.05 %-of-frame box is
#: still several pixels a side, so E6's "lone tiny animal" is a genuine detection
#: rather than a degenerate crop — those are different cases and E26 owns the other.
FRAME = (800, 600)
FRAME_AREA = FRAME[0] * FRAME[1]


@dataclass(frozen=True, slots=True)
class Case:
    """One fixture image and the label the pipeline must give it."""

    name: str
    expect: str
    why: str


def noisy(size: tuple[int, int], seed: int) -> Image.Image:
    """Uniform noise — a high variance-of-Laplacian, i.e. reliably *sharp*."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), mode="RGB"
    )


def blurred(size: tuple[int, int], seed: int) -> Image.Image:
    """The same noise under a heavy Gaussian: variance collapses toward zero."""
    return noisy(size, seed).filter(ImageFilter.GaussianBlur(radius=12))


def square_box(area_frac: float, *, left: float, top: float = 10.0) -> dict[str, float]:
    """A square animal box covering ``area_frac`` of the frame, at ``left``."""
    side = (area_frac * FRAME_AREA) ** 0.5
    return {"x0": left, "y0": top, "x1": left + side, "y1": top + side}


def animal(area_frac: float, *, left: float, conf: float = 0.91) -> dict[str, object]:
    return {"cls": "animal", "conf": conf, **square_box(area_frac, left=left)}


def sidecar(path: Path, boxes: list[dict[str, object]]) -> None:
    path.with_name(path.name + ".boxes.json").write_text(
        json.dumps(boxes, indent=2), encoding="utf-8"
    )


def build_card(root: Path) -> list[Case]:
    """Write the fixture card under ``root``. Returns the expected outcomes.

    The expectations below are what the pipeline produces *today*: with no species
    model wired, a dominant animal is ``unknown`` rather than a species name, so
    dominance is observed as ``unknown`` vs ``multiple``. That distinction is the
    rule itself, and these same fixtures gain real species names when the
    classifier lands — the geometry never changes.
    """
    dcim = root / "DCIM" / "100CANON"
    dcim.mkdir(parents=True, exist_ok=True)
    cases: list[Case] = []

    # --- E5: the dominance boundary, stated exactly -------------------------
    # 0.40 vs 0.25 is precisely 1.6, and `>=` resolves an exact tie to dominant.
    exact = dcim / "e5_exact_ratio.jpg"
    noisy(FRAME, 1).save(exact, quality=95)
    sidecar(exact, [animal(0.40, left=10), animal(0.25, left=500)])
    cases.append(Case(exact.name, "unknown", "ratio exactly 1.6 -> dominant"))

    # Just under the ratio: nothing dominates, so the frame is crowded.
    under = dcim / "e5_just_under.jpg"
    noisy(FRAME, 2).save(under, quality=95)
    sidecar(under, [animal(0.40, left=10), animal(0.40 / 1.59, left=500)])
    cases.append(Case(under.name, "multiple", "ratio 1.59 -> multiple"))

    # Equal areas: ratio 1.0, far below 1.6.
    equal = dcim / "e5_equal.png"
    noisy(FRAME, 3).save(equal)
    sidecar(equal, [animal(0.20, left=10), animal(0.20, left=450)])
    cases.append(Case(equal.name, "multiple", "equal areas -> multiple"))

    # A zero-area runner-up must make the leader dominant, not divide by zero.
    zero = dcim / "e5_zero_area_runner_up.jpg"
    noisy(FRAME, 4).save(zero, quality=95)
    sidecar(
        zero,
        [
            animal(0.30, left=10),
            {"cls": "animal", "conf": 0.5, "x0": 700, "y0": 500, "x1": 700, "y1": 500},
        ],
    )
    cases.append(Case(zero.name, "unknown", "zero-area runner-up -> dominant"))

    # --- E6: no area floor, at any size ------------------------------------
    # 0.05 % of the frame, alone in the image. An absolute floor would bin it.
    tiny = dcim / "e6_lone_tiny_animal.jpg"
    noisy(FRAME, 5).save(tiny, quality=95)
    sidecar(tiny, [animal(0.0005, left=200)])
    cases.append(Case(tiny.name, "unknown", "lone 0.05%-of-frame animal is kept"))

    # 0.5 % vs 0.1 %: ratio 5.0, resolved by the ratio alone.
    ratio_pair = dcim / "e6_small_pair_resolved_by_ratio.tif"
    noisy(FRAME, 6).save(ratio_pair)
    sidecar(ratio_pair, [animal(0.005, left=100), animal(0.001, left=600)])
    cases.append(Case(ratio_pair.name, "unknown", "0.5% vs 0.1% -> dominant"))

    # --- E26: a degenerate box wins dominance ------------------------------
    # Sub-2px after the margin: unclassifiable, still counted, so `unknown`.
    degenerate = dcim / "e26_degenerate_dominant.jpg"
    noisy(FRAME, 7).save(degenerate, quality=95)
    sidecar(
        degenerate,
        [{"cls": "animal", "conf": 0.8, "x0": 400, "y0": 300, "x1": 401, "y1": 301}],
    )
    cases.append(Case(degenerate.name, "unknown", "degenerate dominant box"))

    # --- E2: the animal-free split, and person/vehicle ---------------------
    scenery = dcim / "e2_landscape_sharp.jpg"
    noisy(FRAME, 8).save(scenery, quality=95)
    cases.append(Case(scenery.name, "landscape", "no boxes, sharp"))

    junk = dcim / "e2_junk_blurry.jpg"
    blurred(FRAME, 9).save(junk, quality=95)
    cases.append(Case(junk.name, "junk", "no boxes, blurry"))

    # Only people: stored and drawn, but never a label (§5.4).
    people = dcim / "e2_people_only.heic"
    noisy(FRAME, 10).save(people, quality=90)
    sidecar(
        people,
        [{"cls": "person", "conf": 0.95, "x0": 100, "y0": 50, "x1": 300, "y1": 550}],
    )
    cases.append(Case(people.name, "landscape", "person-only -> landscape"))

    # --- skips: one benign, one abnormal ------------------------------------
    (dcim / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42not-a-real-video")
    truncated = dcim / "truncated.jpg"
    truncated.write_bytes(noisy((64, 64), 11).tobytes()[:200])

    return cases


def main(argv: list[str]) -> int:
    root = Path(argv[1]) if len(argv) > 1 else Path("tests/e2e/_fixtures/card")
    root.mkdir(parents=True, exist_ok=True)
    cases = build_card(root)
    print(f"fixture card at {root.resolve()}")
    for case in cases:
        print(f"  {case.name:<42} -> {case.expect:<10} ({case.why})")
    print("  clip.mp4                                   -> skipped video (benign)")
    print("  truncated.jpg                              -> decode_error (abnormal)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
