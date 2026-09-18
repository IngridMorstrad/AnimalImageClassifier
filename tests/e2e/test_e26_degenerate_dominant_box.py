"""E26: a degenerate box that wins dominance yields ``unknown`` (§5.3, §5.7).

The awkward case the design is explicit about, and the one where the two tempting
shortcuts are both wrong. A box under 2 px a side cannot be resized to a model's
input meaningfully, so it cannot be *classified* — but:

- **Dropping it is wrong.** It would be an area floor by the back door, and E6 exists
  because that is the one thing this project will not do. It still counts as an
  animal for dominance.
- **Guessing a species for it is wrong.** There are no pixels to guess from.

So it stays in the count, wins dominance on area if it is the largest, and the image
is labelled ``unknown`` — recorded honestly, filed under ``unknown/``, and the run
still exits successfully as far as this image is concerned. Nothing is discarded.
"""

from __future__ import annotations


def test_the_image_is_filed_as_unknown(classified) -> None:
    assert classified.label_of("e26_degenerate_dominant.jpg") == "unknown"


def test_the_box_is_recorded_as_degenerate_and_dominant(classified) -> None:
    """§5.9: ``species_status='degenerate'`` and it still won."""
    boxes = classified.boxes_of("e26_degenerate_dominant.jpg")
    assert len(boxes) == 1, "the box was kept, not dropped"
    box = boxes[0]
    assert box["species_status"] == "degenerate"
    assert box["is_dominant"] == 1, "a degenerate box can still win dominance"
    assert box["cls"] == "animal"


def test_no_species_is_invented_for_it(classified) -> None:
    """``unknown`` carries no name and no confidence — there is nothing to report."""
    row = classified.image_row("e26_degenerate_dominant.jpg")
    assert row["label"] == "unknown"
    assert row["confidence"] is None
    assert row["species_common"] is None
    assert row["species_scientific"] is None

    box = classified.boxes_of("e26_degenerate_dominant.jpg")[0]
    assert box["species_common"] is None
    assert box["species_conf"] is None


def test_the_box_is_genuinely_sub_two_pixels(classified) -> None:
    """Guard the fixture: it must still be small enough to be degenerate (§5.3)."""
    box = classified.boxes_of("e26_degenerate_dominant.jpg")[0]
    assert (box["x1"] - box["x0"]) < 2.0
    assert (box["y1"] - box["y0"]) < 2.0


def test_the_image_is_still_present_on_disk(classified) -> None:
    """Nothing is discarded: the photo is filed, not dropped (§5.7)."""
    filed = classified.output / "unknown" / "e26_degenerate_dominant.jpg"
    assert filed.is_file()
    assert filed.stat().st_size > 0
