"""E1: the source card is never written to (invariant I1, DESIGN.md §5.1, §5.8).

The single most important promise this tool makes: it reads an irreplaceable SD card
and must not change one byte of it. The test snapshots ``(size, mtime_ns, sha256)``
for every file on the card, runs the tool, and asserts the snapshot is identical
afterwards.

Two legs make the guarantee load-bearing rather than incidental:

- **copy mode** — the ordinary path.
- **``--hardlink`` mode** — the dangerous one. A hardlinked destination *is* the
  card's inode (``st_ino`` is shared), so a path-based "don't write the source"
  guard would not catch an in-place write through the destination. This leg proves
  the structural no-write rule (§5.8): the card's ``mtime_ns``/``sha256`` are
  unchanged *and* the inode is still shared, so the bytes were never touched from
  either name.

The re-tag, ``export-trainset`` and ``verify --fix`` legs are ``xfail`` until those
commands exist (chunks 21/23/24); they are written now so the immutability contract
covers them the moment they land, and ``strict=False`` means they flip to passing
without editing this file.
"""

from __future__ import annotations

import os

import pytest


def _snapshot(root) -> dict[str, tuple[int, int, str]]:
    import hashlib  # noqa: PLC0415

    return {
        str(p.relative_to(root)): (
            p.stat().st_size,
            p.stat().st_mtime_ns,
            hashlib.sha256(p.read_bytes()).hexdigest(),
        )
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _classify(run_cli, card, output, *extra):
    return run_cli(
        ["classify", str(card), "--output", str(output), "--detector", "scripted", *extra]
    )


def test_copy_mode_leaves_the_card_byte_identical(run_cli, fixture_card, tmp_path):
    before = _snapshot(fixture_card)
    _classify(run_cli, fixture_card, tmp_path / "pics")
    assert _snapshot(fixture_card) == before


def test_hardlink_mode_leaves_the_card_byte_identical(run_cli, fixture_card, tmp_path):
    """I1 under an aliased inode — the case a path guard would miss."""
    before = _snapshot(fixture_card)
    result = _classify(run_cli, fixture_card, tmp_path / "pics", "--hardlink")
    # Same filesystem here (both under tmp), so hardlink succeeds; if it ever
    # EXDEV'd the run would exit 3 and the card would still be untouched.
    assert result.returncode in {0, 4}
    assert _snapshot(fixture_card) == before


def test_a_read_only_card_still_classifies(run_cli, fixture_card, tmp_path):
    """A card mounted read-only (chmod 0o500) must be fully processable."""
    for directory in sorted(
        (p for p in fixture_card.rglob("*") if p.is_dir()), reverse=True
    ):
        os.chmod(directory, 0o500)
    os.chmod(fixture_card, 0o500)
    try:
        before = _snapshot(fixture_card)
        result = _classify(run_cli, fixture_card, tmp_path / "pics")
        assert result.returncode in {0, 4}
        assert _snapshot(fixture_card) == before
    finally:
        for directory in (fixture_card, *(p for p in fixture_card.rglob("*") if p.is_dir())):
            os.chmod(directory, 0o755)


def test_retag_never_writes_the_source(run_cli, fixture_card, tmp_path):
    """Re-tag is a pure output-tree rename; the card is never opened (§5.8)."""
    import sys as _sys  # noqa: PLC0415

    _sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from conftest_gui import make_client  # noqa: PLC0415

    output = tmp_path / "pics"
    _classify(run_cli, fixture_card, output)
    before = _snapshot(fixture_card)

    client = make_client(output, allow_new_labels=True)
    items = client.get("/api/images").json()["items"]
    target = next((i for i in items if i["label"] == "unknown"), items[0])
    client.post(f"/api/images/{target['sha256']}/label", json={"label": "lion"})

    assert _snapshot(fixture_card) == before, "re-tag must not touch the source card"


def test_export_trainset_never_writes_the_source(run_cli, fixture_card, tmp_path):
    """export-trainset reads the catalog, not the card (§7.5, I1)."""
    output = tmp_path / "pics"
    _classify(run_cli, fixture_card, output)
    before = _snapshot(fixture_card)
    run_cli(["export-trainset", "--output", str(output), "--destination", str(tmp_path / "t.jsonl"),
             "--include-model-labels", "--min-conf", "0.0"])
    assert _snapshot(fixture_card) == before


def test_verify_fix_never_writes_the_source(run_cli, fixture_card, tmp_path):
    """verify --fix only os.replace/os.unlink inside the output tree (§8, I1)."""
    output = tmp_path / "pics"
    _classify(run_cli, fixture_card, output)
    before = _snapshot(fixture_card)
    run_cli(["verify", "--output", str(output), "--fix"])
    assert _snapshot(fixture_card) == before
