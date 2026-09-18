"""E5: the dominance rule's exact boundaries, through the CLI (DESIGN.md §5.7).

Every case here is a *pair* of animal boxes whose area ratio is stated exactly in
the fixture's sidecar, which is the whole reason ``--detector scripted`` ships: with
a real detector these assertions would be about whatever geometry MegaDetector
produced, not about the rule.

The three properties being pinned are the ones easiest to break by "tidying" the
comparison later:

- ``>=``, so an **exact** 1.6 ratio resolves to dominant. Rewriting it as ``>``
  would silently flip this case.
- a **multiplication**, so a zero-area runner-up makes the leader dominant instead
  of raising ``ZeroDivisionError``. Rewriting it as ``a[0]/a[1] >= ratio`` would
  crash on that fixture.
- ratio **1.0 is not dominance**: two equal animals are ``multiple``.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    ("filename", "expected", "why"),
    [
        ("e5_exact_ratio.jpg", "unknown", "0.40 vs 0.25 is exactly 1.6; >= is dominant"),
        ("e5_just_under.jpg", "multiple", "ratio 1.59 is below 1.6"),
        ("e5_equal.png", "multiple", "equal areas: ratio 1.0"),
        (
            "e5_zero_area_runner_up.jpg",
            "unknown",
            "a zero-area second box must not divide by zero",
        ),
    ],
)
def test_dominance_boundary(classified, filename: str, expected: str, why: str) -> None:
    assert classified.label_of(filename) == expected, why


def test_exact_ratio_really_is_exact(classified) -> None:
    """Guard the fixture itself: a drifting sidecar would make the test vacuous."""
    boxes = classified.boxes_of("e5_exact_ratio.jpg")
    areas = sorted((box["area_frac"] for box in boxes), reverse=True)
    assert areas[0] == pytest.approx(1.6 * areas[1], rel=1e-9)


def test_the_dominant_box_is_the_largest_one(classified) -> None:
    """§5.9 records *which* animal the decision was about."""
    boxes = classified.boxes_of("e5_exact_ratio.jpg")
    dominant = [box for box in boxes if box["is_dominant"]]
    assert len(dominant) == 1
    assert dominant[0]["area_frac"] == max(box["area_frac"] for box in boxes)


def test_multiple_has_no_dominant_box(classified) -> None:
    """Nothing won, so nothing is flagged — ``multiple`` is not a winner."""
    for filename in ("e5_just_under.jpg", "e5_equal.png"):
        boxes = classified.boxes_of(filename)
        assert boxes, filename
        assert not any(box["is_dominant"] for box in boxes), filename


def test_zero_area_runner_up_is_kept_as_an_animal(classified) -> None:
    """It loses dominance but still counts — dropping it would be an area floor."""
    boxes = classified.boxes_of("e5_zero_area_runner_up.jpg")
    assert len(boxes) == 2, "both boxes are recorded"
    assert min(box["area_frac"] for box in boxes) == 0.0
    assert sum(box["is_dominant"] for box in boxes) == 1
