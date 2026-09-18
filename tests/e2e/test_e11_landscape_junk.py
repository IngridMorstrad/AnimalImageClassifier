"""E11: the animal-free split into ``landscape`` vs ``junk`` (§5.2, §5.7).

When no animal is detected, sharpness decides: a sharp scene is ``landscape``, a
blurry mess is ``junk``. The blur metric is variance-of-Laplacian, and the one
subtle requirement is that it is measured **downscale-only** — never upscaled — so a
small sharp photo is not pushed below the threshold and misfiled as ``junk`` by
interpolation bias. This test includes a ≤512 px sharp photo precisely to pin that.

``blur_score`` and ``blur_ref_edge`` are stored for every image, because two scores
are only comparable with the edge they were measured at (a large photo is scored at
the 512 px reference; a small one natively).
"""

from __future__ import annotations

import sqlite3

import pytest


def _catalog(output):
    c = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _classify(run_cli, card, output):
    return run_cli(
        ["classify", str(card), "--output", str(output), "--detector", "scripted"]
    )


@pytest.fixture
def blur_card(tmp_path_factory):
    """Three animal-free photos: large sharp, large blurry, small sharp."""
    from PIL import ImageFilter  # noqa: PLC0415
    from conftest import make_e2e_fixtures  # noqa: PLC0415

    card = tmp_path_factory.mktemp("blur") / "DCIM"
    card.mkdir()
    make_e2e_fixtures.noisy((1600, 1200), 1).save(card / "large_sharp.jpg", quality=95)
    make_e2e_fixtures.noisy((1600, 1200), 2).filter(
        ImageFilter.GaussianBlur(radius=12)
    ).save(card / "large_blurry.jpg", quality=95)
    make_e2e_fixtures.noisy((400, 300), 3).save(card / "small_sharp.jpg", quality=95)
    return card.parent


def test_sharp_is_landscape_blurry_is_junk(run_cli, blur_card, tmp_path):
    output = tmp_path / "pics"
    _classify(run_cli, blur_card, output)
    filed = {p.name: p.parent.name for p in output.rglob("*.jpg")}
    assert filed["large_sharp.jpg"] == "landscape"
    assert filed["large_blurry.jpg"] == "junk"


def test_small_sharp_image_is_landscape_not_junk(run_cli, blur_card, tmp_path):
    """The no-upscaling requirement: a small sharp photo must not read as blurry."""
    output = tmp_path / "pics"
    _classify(run_cli, blur_card, output)
    filed = {p.name: p.parent.name for p in output.rglob("*.jpg")}
    assert filed["small_sharp.jpg"] == "landscape", (
        "a small sharp image scored natively must stay above the blur threshold; "
        "if it reads as junk, the blur step is upscaling and depressing the variance"
    )


def test_blur_score_and_ref_edge_are_stored_for_all(run_cli, blur_card, tmp_path):
    output = tmp_path / "pics"
    _classify(run_cli, blur_card, output)
    with _catalog(output) as c:
        rows = {
            r["path"].rsplit("/", 1)[1]: (r["blur_score"], r["blur_ref_edge"])
            for r in c.execute(
                "SELECT s.path, i.blur_score, i.blur_ref_edge "
                "FROM images i JOIN sources s USING(sha256)"
            )
        }
    for name, (score, ref) in rows.items():
        assert score is not None, name
        assert ref is not None, name
    # Large photos scored at the 512 reference; the small one at its own long edge.
    assert rows["large_sharp.jpg"][1] == 512
    assert rows["small_sharp.jpg"][1] == 400
