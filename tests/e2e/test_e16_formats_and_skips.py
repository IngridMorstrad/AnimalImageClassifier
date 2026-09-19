"""E16: formats, skip reasons, the byte cap and symlink policy (§5.1, §3.1, §10.1).

One card exercising the whole scan-time decision table, driven three ways
(default, ``--formats jpeg``, ``--follow-source-symlinks``) plus the fatal
``--formats raw``. The point is that every skip reason is reachable and lands in
``skipped`` with the right enumerated value, and that the benign/abnormal split
sets the exit code: this card exits **4** because it holds a 0-byte file and a
truncated JPEG (abnormal), even though most of its skips are benign.

The over-cap file is asserted by its **observable** consequences (NIT 15): a
``too_large`` skip, no ``images``/``sources`` row, and no ``DecodeError`` in
stderr. "Its bytes were not read" is a structural property of ``scan.py`` (it only
``stat``s), not something a test can see, so the test asserts what a user can see.
"""

from __future__ import annotations

import os
import sqlite3

import pytest


def _catalog(output):
    c = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


def _classify(run_cli, card, output, *extra):
    return run_cli(
        ["classify", str(card), "--output", str(output), "--detector", "scripted", *extra]
    )


@pytest.fixture
def formats_card(tmp_path_factory):
    """A card that reaches most of the scan-time skip reasons (§5.1)."""
    from conftest import make_e2e_fixtures

    root = tmp_path_factory.mktemp("fmtcard")
    dcim = root / "DCIM"
    dcim.mkdir()

    def img(name, fmt_seed):
        make_e2e_fixtures.noisy((320, 240), fmt_seed).save(dcim / name)

    img("photo.jpg", 1)
    img("photo.png", 2)
    make_e2e_fixtures.noisy((320, 240), 3).save(dcim / "photo.tif")
    make_e2e_fixtures.noisy((320, 240), 4).save(dcim / "photo.heic", quality=80)

    (dcim / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")   # video (benign)
    (dcim / "raw.cr2").write_bytes(b"raw-bytes")                    # raw_not_enabled (benign)
    (dcim / "notes.txt").write_bytes(b"hello")                      # unsupported_extension (benign)
    (dcim / "empty.jpg").write_bytes(b"")                           # zero_bytes (abnormal)
    (dcim / "truncated.jpg").write_bytes(b"\xff\xd8\xff\xe0broken") # decode_error (abnormal)

    # An over-cap sparse file: 2 MiB apparent size, near-zero real bytes.
    big = dcim / "huge.jpg"
    with open(big, "wb") as fh:
        fh.truncate(2 * 1024 * 1024)

    # A symlink to a JPEG inside the card, and one to a JPEG outside it.
    outside = tmp_path_factory.mktemp("outside")
    make_e2e_fixtures.noisy((320, 240), 5).save(outside / "external.jpg")
    os.symlink(dcim / "photo.jpg", dcim / "link_inside.jpg")
    os.symlink(outside / "external.jpg", dcim / "link_outside.jpg")

    return root


def test_default_run_files_the_four_real_photos(run_cli, formats_card, tmp_path):
    output = tmp_path / "pics"
    result = _classify(run_cli, formats_card, output, "--max-file-bytes", "1048576")
    # Abnormal skips present (empty + truncated), so exit 4.
    assert result.returncode == 4, result.stderr

    with _catalog(output) as c:
        images = c.execute("SELECT COUNT(*) AS n FROM images WHERE status='done'").fetchone()["n"]
        assert images == 4, "jpeg/png/tiff/heic decoded and filed"
        reasons = {
            r["reason"]: r["n"]
            for r in c.execute("SELECT reason, COUNT(*) AS n FROM skipped GROUP BY reason")
        }
    # Every benign reason on the card is represented.
    assert reasons.get("video") == 1
    assert reasons.get("raw_not_enabled") == 1
    assert reasons.get("unsupported_extension") == 1
    assert reasons.get("too_large") == 1
    assert reasons.get("symlink") == 2, "both symlinks skipped by default"
    assert reasons.get("zero_bytes") == 1
    assert reasons.get("decode_error") == 1


def test_over_cap_file_is_skipped_not_decoded(run_cli, formats_card, tmp_path):
    """NIT 15: observable consequences only — skip row, no image row, no DecodeError."""
    output = tmp_path / "pics"
    result = _classify(run_cli, formats_card, output, "--max-file-bytes", "1048576")
    with _catalog(output) as c:
        huge = c.execute("SELECT * FROM skipped WHERE path LIKE '%huge.jpg'").fetchall()
        assert len(huge) == 1 and huge[0]["reason"] == "too_large"
        rows = c.execute("SELECT COUNT(*) AS n FROM sources WHERE path LIKE '%huge.jpg'").fetchone()
        assert rows["n"] == 0, "no sources row for a file whose bytes were never read"
    assert "DecodeError" not in result.stderr
    assert "huge.jpg" not in result.stderr or "too_large" in result.stderr.lower()


def test_formats_jpeg_disables_the_other_families(run_cli, formats_card, tmp_path):
    """§3.1: ``--formats jpeg`` replaces the list; png/tiff/heic become format_disabled."""
    output = tmp_path / "pics"
    _classify(run_cli, formats_card, output, "--formats", "jpeg", "--max-file-bytes", "1048576")
    with _catalog(output) as c:
        reasons = {
            r["reason"]: r["n"]
            for r in c.execute("SELECT reason, COUNT(*) AS n FROM skipped GROUP BY reason")
        }
        done = c.execute("SELECT COUNT(*) AS n FROM images WHERE status='done'").fetchone()["n"]
    assert reasons.get("format_disabled") == 3, "png, tiff, heic disabled"
    assert done == 1, "only the one real jpeg"


def test_follow_source_symlinks_ingests_the_inside_link(run_cli, formats_card, tmp_path):
    """§5.1: with the flag, an in-card symlink is ingested; an escaping one is refused."""
    output = tmp_path / "pics"
    _classify(
        run_cli, formats_card, output,
        "--follow-source-symlinks", "--max-file-bytes", "1048576",
    )
    with _catalog(output) as c:
        reasons = {
            r["reason"]: r["n"]
            for r in c.execute("SELECT reason, COUNT(*) AS n FROM skipped GROUP BY reason")
        }
    assert "symlink" not in reasons, "the flag stops the default symlink skip"
    assert reasons.get("symlink_escape") == 1, "the link pointing outside the card is refused"


def test_formats_raw_is_a_fatal_usage_error(run_cli, formats_card, tmp_path):
    """§3.1: ``raw`` is not a family; it is fatal, listing the valid set."""
    output = tmp_path / "pics"
    result = _classify(run_cli, formats_card, output, "--formats", "raw")
    assert result.returncode in {2, 3}, result.stderr
    assert not output.exists() or not any(
        p.is_dir() and p.name in {"landscape", "junk", "unknown", "multiple"}
        for p in output.iterdir()
    ), "no label directory created on a config failure"
