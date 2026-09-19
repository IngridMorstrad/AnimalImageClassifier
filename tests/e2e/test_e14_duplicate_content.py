"""E14: duplicate content is filed once; content-changed-in-place is a new image (§5.9).

Content, not path, is the identity in this tool. Two consequences:

- **The same photo copied twice on the card** is one sha256, so it is classified
  once, filed once, and both source paths are recorded in ``sources``. That is what
  makes a card with redundant copies cheap.
- **A path whose bytes changed in place** (same name, different content) is an
  *upsert* of the ``sources`` row and a *new* ``images`` row — the previous image
  and its destination are retained (NIT 17), because the old photo really was filed
  and the new one is a different image.
"""

from __future__ import annotations

import json
import shutil
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
def duplicate_card(tmp_path_factory):
    """A card with the same photo under two names (byte-identical copies)."""

    card = tmp_path_factory.mktemp("dupcard") / "DCIM"
    card.mkdir()
    from conftest import make_e2e_fixtures

    original = card / "a.jpg"
    make_e2e_fixtures.noisy(make_e2e_fixtures.FRAME, 42).save(original, quality=95)
    boxes = json.dumps([{"cls": "animal", "conf": 0.9, "x0": 50, "y0": 50, "x1": 450, "y1": 400}])
    (card / "a.jpg.boxes.json").write_text(boxes)
    shutil.copy2(original, card / "b.jpg")  # byte-identical duplicate
    (card / "b.jpg.boxes.json").write_text(boxes)
    return card.parent


def test_duplicate_content_is_one_image_two_sources(run_cli, duplicate_card, tmp_path):
    output = tmp_path / "pics"
    _classify(run_cli, duplicate_card, output)
    with _catalog(output) as c:
        assert c.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == 1
        sources = [r["path"] for r in c.execute("SELECT path FROM sources ORDER BY path")]
        assert len(sources) == 2, "both card copies are recorded"
        assert sources[0].endswith("a.jpg") and sources[1].endswith("b.jpg")

    # One destination file — the duplicate was recognised as already_present.
    filed = [
        p for p in output.rglob("*")
        if p.is_file() and not p.name.startswith(".catalog")
    ]
    assert len(filed) == 1, "filed once, whichever name won the race"


def test_content_changed_in_place_is_a_new_image(run_cli, duplicate_card, tmp_path):
    """Same path, new bytes → upsert the source, keep the old image (NIT 17)."""
    output = tmp_path / "pics"
    _classify(run_cli, duplicate_card, output)
    with _catalog(output) as c:
        before = c.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]

    # Rewrite a.jpg with different content (and a matching sidecar) in place.
    from conftest import make_e2e_fixtures

    a = duplicate_card / "DCIM" / "a.jpg"
    make_e2e_fixtures.noisy(make_e2e_fixtures.FRAME, 999).save(a, quality=95)

    _classify(run_cli, duplicate_card, output)
    with _catalog(output) as c:
        after = c.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"]
        assert after == before + 1, "the changed file is a new image row"
        a_source = c.execute(
            "SELECT COUNT(*) AS n FROM sources WHERE path LIKE '%/a.jpg'"
        ).fetchone()["n"]
        assert a_source == 1, "the source path was upserted, not duplicated"
