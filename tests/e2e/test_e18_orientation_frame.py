"""E18, coordinate-frame leg: one frame, the EXIF-transposed one (invariant I9).

The single most corruptible thing in this pipeline is which raster the coordinates
refer to. An orientation-6 photo has a **landscape stored raster** and a **portrait
displayed image**; §5.2 fixes the displayed one as *the* frame, and everything — the
catalog's `width`/`height`, every box coordinate, `area_frac`, the GUI overlay — must
be in it. The raw stored dimensions must appear nowhere.

This fixture is the one that catches a regression: if any layer reverted to the stored
raster, `images.width` would be 400 (not 300) and the box would be judged against the
wrong axis, silently skewing `area_frac` and therefore the dominance rule.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import build_orientation6_card, classify, make_client

STORED = (400, 300)      # what is on disk
DISPLAYED = (300, 400)   # what exif_transpose produces — the one frame


@pytest.fixture
def oriented(cli_path, tmp_path):
    card = build_orientation6_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    return card, output


def test_stored_raster_really_is_landscape(oriented):
    """Guard the fixture: without this the test could pass vacuously."""
    from PIL import Image

    card, _ = oriented
    photo = card / "DCIM" / "portrait.jpg"
    with Image.open(photo) as img:
        assert img.size == STORED, "the stored raster must be landscape for this test"
        assert img.getexif().get(0x0112) == 6, "orientation tag 6 must survive the save"


def test_catalog_records_the_transposed_frame(oriented):
    """I9: width/height are post-transpose, so width < height here."""
    _, output = oriented
    con = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT width, height FROM images").fetchone()
    finally:
        con.close()
    assert (row["width"], row["height"]) == DISPLAYED, (
        f"expected the transposed frame {DISPLAYED}, got "
        f"{(row['width'], row['height'])} — a layer is using the stored raster"
    )
    assert row["width"] < row["height"], "the displayed image is portrait"


def test_boxes_lie_inside_the_transposed_frame(oriented):
    """Every stored coordinate is inside width/height — the frame it was measured in."""
    _, output = oriented
    con = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        image = con.execute("SELECT sha256, width, height FROM images").fetchone()
        boxes = con.execute(
            "SELECT * FROM boxes WHERE sha256=?", (image["sha256"],)
        ).fetchall()
    finally:
        con.close()

    assert boxes, "the sidecar box was ingested"
    for box in boxes:
        assert 0 <= box["x0"] <= image["width"], box["x0"]
        assert 0 <= box["x1"] <= image["width"], box["x1"]
        assert 0 <= box["y0"] <= image["height"], box["y0"]
        assert 0 <= box["y1"] <= image["height"], box["y1"]
        # The box is taller than wide, which is only true in the transposed frame.
        assert (box["y1"] - box["y0"]) > (box["x1"] - box["x0"])
        assert 0.0 < box["area_frac"] <= 1.0


def test_served_thumb_matches_the_stored_dimensions(oriented):
    """§6: the thumb is exif_transposed, so its aspect matches width/height."""
    import io

    from PIL import Image

    _, output = oriented
    client = make_client(output)
    item = client.get("/api/images").json()["items"][0]
    resp = client.get(f"/api/images/{item['sha256']}/thumb")
    assert resp.status_code == 200

    with Image.open(io.BytesIO(resp.content)) as thumb:
        # Thumbnails preserve aspect ratio; the orientation must match the frame.
        assert thumb.width < thumb.height, (
            "the served thumb is landscape, so exif_transpose was not applied — the "
            "GUI would draw boxes against the wrong axis"
        )
        stored_aspect = item["width"] / item["height"]
        thumb_aspect = thumb.width / thumb.height
        assert abs(stored_aspect - thumb_aspect) < 0.05, (stored_aspect, thumb_aspect)
