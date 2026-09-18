"""E17: ``--dry-run`` plans without touching the output tree (§5.8).

A dry run does the whole pipeline — scan, decode, detect, decide — but writes
nothing: no label directories, no image files, not even a temp. It records the
destination each image *would* get, with ``status='planned'``, so a plan can be
inspected before a single byte moves. Exit 0, because planning is a success.

(The GUI 409 legs for ``planned`` images — a ``/full`` request with no materialized
file — arrive with the GUI in chunk 20.)
"""

from __future__ import annotations

import sqlite3


def _catalog(output):
    c = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _classify(run_cli, card, output, *extra):
    return run_cli(
        ["classify", str(card), "--output", str(output), "--detector", "scripted", *extra]
    )


def test_dry_run_writes_no_image_files(run_cli, five_image_card, tmp_path):
    output = tmp_path / "pics"
    result = _classify(run_cli, five_image_card, output, "--dry-run")
    assert result.returncode == 0, result.stderr

    image_files = [
        p for p in output.rglob("*")
        if p.is_file() and not p.name.startswith(".catalog")
    ]
    assert image_files == [], f"dry run wrote files: {image_files}"
    # No label directories either.
    assert not (output / "unknown").exists()


def test_dry_run_records_planned_rows_with_destinations(run_cli, five_image_card, tmp_path):
    output = tmp_path / "pics"
    _classify(run_cli, five_image_card, output, "--dry-run")
    with _catalog(output) as c:
        rows = list(c.execute("SELECT status, dest_path FROM images"))
    assert rows, "the run still recorded what it would do"
    for row in rows:
        assert row["status"] == "planned"
        assert row["dest_path"] is not None, "the planned destination is recorded"


def test_dry_run_then_real_run_files_everything(run_cli, five_image_card, tmp_path):
    """A plan is not a commitment: a following real run does the writing."""
    output = tmp_path / "pics"
    _classify(run_cli, five_image_card, output, "--dry-run")
    _classify(run_cli, five_image_card, output)
    with _catalog(output) as c:
        done = c.execute("SELECT COUNT(*) AS n FROM images WHERE status='done'").fetchone()["n"]
    assert done == 5
    filed = [p for p in output.rglob("*") if p.is_file() and not p.name.startswith(".catalog")]
    assert len(filed) == 5
