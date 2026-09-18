"""E19-E21: GUI re-tag — rename, validation, and the override record (§5.8, §6).

Re-tag is a pure output-tree rename that never opens the source. The tests assert:

- the file moves to the new label directory, the old entry is gone, and the source
  card is byte-identical afterwards (I1);
- an ``overrides`` row is appended and ``label_source`` becomes ``human``;
- validation: a reserved label is 422, and an unknown label is 422 unless the GUI
  was started with ``--allow-new-labels``;
- a later ``classify`` keeps the human label (``--reclassify`` does not clobber it).
"""

from __future__ import annotations

import hashlib
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import build_card, classify, make_client  # noqa: E402


def _snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _catalog(output):
    c = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


@pytest.fixture
def setup(cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    return card, output


def test_retag_moves_the_file_and_records_the_override(setup):
    card, output = setup
    before = _snapshot(card)
    client = make_client(output, allow_new_labels=True)
    sha = [i for i in client.get("/api/images").json()["items"] if i["label"] == "unknown"][0]["sha256"]

    resp = client.post(f"/api/images/{sha}/label", json={"label": "lion", "note": "it is a lion"})
    assert resp.status_code == 200, resp.text
    assert (output / "lion").is_dir()
    assert not (output / "unknown" / "animal.jpg").exists(), "moved out of the old label dir"

    with _catalog(output) as c:
        row = c.execute("SELECT label, label_source FROM images WHERE sha256=?", (sha,)).fetchone()
        assert row["label"] == "lion"
        assert row["label_source"] == "human"
        override = c.execute("SELECT old_label, new_label, note FROM overrides WHERE sha256=?", (sha,)).fetchone()
        assert override["old_label"] == "unknown"
        assert override["new_label"] == "lion"
        assert override["note"] == "it is a lion"

    assert _snapshot(card) == before, "re-tag never touches the source card (I1)"


def test_reserved_label_is_rejected(setup):
    _, output = setup
    client = make_client(output, allow_new_labels=True)
    sha = client.get("/api/images").json()["items"][0]["sha256"]
    assert client.post(f"/api/images/{sha}/label", json={"label": "junk"}).status_code == 422
    assert client.post(f"/api/images/{sha}/label", json={"label": "BAD LABEL"}).status_code == 422


def test_unknown_label_needs_allow_new_labels(setup):
    _, output = setup
    strict = make_client(output, allow_new_labels=False)
    sha = strict.get("/api/images").json()["items"][0]["sha256"]
    assert strict.post(f"/api/images/{sha}/label", json={"label": "pangolin"}).status_code == 422

    lenient = make_client(output, allow_new_labels=True)
    assert lenient.post(f"/api/images/{sha}/label", json={"label": "pangolin"}).status_code == 200


def test_human_label_survives_reclassify(setup, cli_path):
    card, output = setup
    client = make_client(output, allow_new_labels=True)
    sha = [i for i in client.get("/api/images").json()["items"] if i["label"] == "unknown"][0]["sha256"]
    client.post(f"/api/images/{sha}/label", json={"label": "lion"})

    # A --reclassify run keeps label_source='human' (§5.9).
    classify(cli_path, card, output, "--reclassify")
    with _catalog(output) as c:
        row = c.execute("SELECT label, label_source FROM images WHERE sha256=?", (sha,)).fetchone()
        assert row["label_source"] == "human", "reclassify must not clobber a human label"
        assert row["label"] == "lion"
