"""E2: a real ``classify`` run files the whole card and records it (DESIGN.md §11).

The happy path, asserted as an *exact* output tree rather than a spot check: a rule
that files one image correctly while also filing a second one somewhere unexpected
is not working, and only set equality catches that.

Exit code 4 is the expectation, not 0, and that is the point of F15's resolution.
The card deliberately contains a truncated JPEG — an **abnormal** skip — alongside
an ``.mp4`` and the sidecars, which are **benign**. §10.1's table marks
``video``/``unsupported_extension`` as "not an error", so a card whose only skips
were those would exit 0; one unreadable photo is what makes the run partial.
"""

from __future__ import annotations

EXPECTED_TREE = {
    # dominance resolved -> the winning animal's label (`unknown` until a species
    # model is wired; the dominance question is the one being asserted here)
    "unknown/e5_exact_ratio.jpg",
    "unknown/e5_zero_area_runner_up.jpg",
    "unknown/e6_lone_tiny_animal.jpg",
    "unknown/e6_small_pair_resolved_by_ratio.tif",
    "unknown/e26_degenerate_dominant.jpg",
    # no animal dominates
    "multiple/e5_just_under.jpg",
    "multiple/e5_equal.png",
    # no animal at all
    "landscape/e2_landscape_sharp.jpg",
    "landscape/e2_people_only.heic",
    "junk/e2_junk_blurry.jpg",
}


def test_exit_code_is_partial_because_one_photo_is_unreadable(classified) -> None:
    assert classified.returncode == 4, classified.stderr


def test_output_tree_is_exactly_as_expected(classified) -> None:
    assert classified.tree() == EXPECTED_TREE


def test_every_format_on_the_card_was_decoded(classified) -> None:
    """JPEG, PNG, TIFF and HEIC all reach a label (§3.1)."""
    filed = {entry.rsplit(".", 1)[1] for entry in classified.tree()}
    assert filed == {"jpg", "png", "tif", "heic"}


def test_source_card_is_byte_identical_afterwards(
    classified, card_snapshot, fixture_card
) -> None:
    """Invariant I1: the card is opened read-only and never written to."""
    from conftest import _hash_tree

    assert _hash_tree(fixture_card) == card_snapshot


def test_images_rows_carry_the_decoded_frame_and_blur(classified) -> None:
    """§5.9: every filed image records its post-transpose size and blur score."""
    for entry in sorted(EXPECTED_TREE):
        row = classified.image_row(entry.split("/", 1)[1])
        assert row["status"] == "done"
        assert (row["width"], row["height"]) == (800, 600)
        assert row["blur_score"] is not None
        assert row["blur_ref_edge"] == 512, "800px long edge downscales to the 512 ref"
        assert row["dest_path"].endswith(entry)
        assert row["mode"] == "copy"
        assert row["model_id"] == "scripted:boxes.json"


def test_blur_separates_landscape_from_junk(classified) -> None:
    """The threshold is only consulted when no animal was detected (§5.7)."""
    sharp = classified.image_row("e2_landscape_sharp.jpg")["blur_score"]
    blurry = classified.image_row("e2_junk_blurry.jpg")["blur_score"]
    assert blurry < 100.0 <= sharp


def test_person_boxes_are_stored_but_never_label(classified) -> None:
    """§5.4: a people-only frame is ``landscape``, and the box is still recorded."""
    assert classified.label_of("e2_people_only.heic") == "landscape"
    boxes = classified.boxes_of("e2_people_only.heic")
    assert [box["cls"] for box in boxes] == ["person"]
    assert [box["is_dominant"] for box in boxes] == [0]


def test_skips_are_recorded_with_their_reasons(classified) -> None:
    """§5.1/§10.1: one abnormal skip, and benign skips for the non-images."""
    reasons = {
        row["reason"]: row["n"]
        for row in classified.rows(
            "SELECT reason, COUNT(*) AS n FROM skipped GROUP BY reason"
        )
    }
    assert reasons["decode_error"] == 1, "the truncated JPEG"
    assert reasons["video"] == 1, "the .mp4"
    assert reasons["unsupported_extension"] == 8, "the .boxes.json sidecars"


def test_the_run_row_records_provenance_and_counts(classified, fixture_card) -> None:
    """§5.9: the run is explainable after the fact."""
    runs = classified.rows("SELECT * FROM runs")
    assert len(runs) == 1
    run = runs[0]
    assert run["state"] == "completed"
    assert run["source_root"] == str(fixture_card)
    assert run["n_done"] == len(EXPECTED_TREE)
    assert run["n_failed"] == 0
    assert run["config_json"] is not None and run["argv"] is not None
