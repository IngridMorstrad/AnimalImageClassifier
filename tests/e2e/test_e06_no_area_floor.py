"""E6: there is no absolute box-area floor, at any size (invariant I2).

The user's standing requirement, and the one this project most wants an executable
guard on. ``tests/e2e/test_cli_surface.py`` already proves no *option* resembling a
floor exists; this file proves no floor exists in the *behaviour*, which is the part
a well-meaning refactor could introduce without adding a flag.

The decisive fixture is a lone animal covering **0.05 % of the frame**. Any absolute
floor — on pixels, on area fraction, on either side's length — files it as
``landscape`` and the photo is effectively lost. The rule that keeps it is
``dominance_ratio``, which is purely relative and has nothing to compare against
when only one animal is present.
"""

from __future__ import annotations


def test_a_lone_tiny_animal_is_not_discarded(classified) -> None:
    """0.05 % of the frame, alone: filed as the animal, never ``landscape``."""
    label = classified.label_of("e6_lone_tiny_animal.jpg")
    assert label == "unknown", (
        "a lone animal covering 0.05% of the frame must be filed as an animal "
        "(`unknown` until a species model is wired), never as landscape — an "
        "absolute area floor is exactly what this asserts the absence of"
    )
    assert label != "landscape"


def test_the_tiny_animal_really_is_tiny(classified) -> None:
    """Guard the fixture: if the box grew, the test would stop meaning anything."""
    boxes = classified.boxes_of("e6_lone_tiny_animal.jpg")
    assert len(boxes) == 1
    assert boxes[0]["area_frac"] < 0.001, boxes[0]["area_frac"]
    assert boxes[0]["is_dominant"] == 1


def test_a_small_pair_is_resolved_by_ratio_not_by_size(classified) -> None:
    """0.5 % vs 0.1 %: both tiny in absolute terms, ratio 5.0 decides."""
    assert classified.label_of("e6_small_pair_resolved_by_ratio.tif") == "unknown"
    boxes = classified.boxes_of("e6_small_pair_resolved_by_ratio.tif")
    areas = sorted((box["area_frac"] for box in boxes), reverse=True)
    assert areas[0] < 0.01, "the winner is still a tiny box in absolute terms"
    assert areas[0] >= 1.6 * areas[1]


def test_no_filed_image_was_dropped_for_being_small(classified) -> None:
    """Every animal box on the card survives into the catalog, whatever its size."""
    tiny = classified.rows("SELECT COUNT(*) AS n FROM boxes WHERE area_frac < 0.001")
    assert tiny[0]["n"] >= 3, (
        "the card contains sub-0.1% boxes on purpose; if they stopped being "
        "recorded, something is filtering boxes by size"
    )
