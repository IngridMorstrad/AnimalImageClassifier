"""E19, mode-specific legs: re-tag in all three modes, and 409 under contention (§5.8).

Re-tag is one ``os.replace`` of a **directory entry**, which is why it works
identically for all three modes — and each mode has a property that would break if it
were implemented as a copy instead:

- **``--link``, card moved away** — the symlink must move *as a symlink*, with
  ``os.readlink`` unchanged and still dangling. Any implementation that dereferenced it
  would raise here, and this is the *normal* reviewing state: the card is unplugged.
- **``--hardlink``** — the entry must keep its ``st_ino``. A copy would allocate a new
  inode and quietly break the aliasing E3 asserts.
- **copy** — content unchanged.

Plus the contention rule: a competing writer holding ``BEGIN IMMEDIATE`` makes the
re-tag fail fast with **409** (the GUI's 250 ms ``busy_timeout``, not the writer's
10 s), and **nothing on disk or in the DB changes**.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import build_card, classify, make_client


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _target_sha(client) -> str:
    return next(
        i for i in client.get("/api/images").json()["items"] if i["label"] == "unknown"
    )["sha256"]


def test_copy_mode_retag_preserves_content(cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    before = _sha(output / "unknown" / "animal.jpg")

    client = make_client(output, allow_new_labels=True)
    sha = _target_sha(client)
    assert client.post(f"/api/images/{sha}/label", json={"label": "lion"}).status_code == 200

    moved = output / "lion" / "animal.jpg"
    assert moved.is_file()
    assert _sha(moved) == before, "content unchanged by the rename"
    assert not (output / "unknown" / "animal.jpg").exists()


def test_link_mode_retag_moves_a_dangling_symlink_as_a_symlink(cli_path, tmp_path):
    """The normal reviewing state: the card is gone, so the link dangles."""
    import shutil

    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output, "--link")

    original = output / "unknown" / "animal.jpg"
    assert original.is_symlink()
    target_before = os.readlink(original)

    # Move the card away entirely — every link now dangles.
    shutil.rmtree(card)
    assert original.is_symlink() and not original.exists(), "dangling, as intended"

    client = make_client(output, allow_new_labels=True)
    sha = _target_sha(client)
    resp = client.post(f"/api/images/{sha}/label", json={"label": "lion"})
    assert resp.status_code == 200, resp.text

    moved = output / "lion" / "animal.jpg"
    assert moved.is_symlink(), "it moved as a symlink, not as a dereferenced copy"
    assert os.readlink(moved) == target_before, "the link target is untouched"
    assert not moved.exists(), "still dangling — nothing was dereferenced or fetched"
    assert not original.is_symlink()


def test_hardlink_mode_retag_keeps_the_inode(cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output, "--hardlink")

    original = output / "unknown" / "animal.jpg"
    inode_before = original.stat().st_ino
    source = card / "DCIM" / "animal.jpg"
    assert source.stat().st_ino == inode_before, "hardlink shares the card's inode"

    client = make_client(output, allow_new_labels=True)
    sha = _target_sha(client)
    assert client.post(f"/api/images/{sha}/label", json={"label": "lion"}).status_code == 200

    moved = output / "lion" / "animal.jpg"
    assert moved.stat().st_ino == inode_before, (
        "the inode must survive a re-tag; a copy would allocate a new one and break "
        "the aliasing E3 asserts"
    )
    assert source.stat().st_ino == inode_before, "the card's entry is untouched"


def test_retag_under_write_contention_is_409_and_changes_nothing(cli_path, tmp_path):
    """§5.9: contention surfaces as a fast 409, never a hung request or a partial write."""
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)

    client = make_client(output, allow_new_labels=True)
    sha = _target_sha(client)
    tree_before = sorted(
        str(p.relative_to(output)) for p in output.rglob("*")
        if p.is_file() and not p.name.startswith(".catalog")
    )

    db = output / ".catalog.db"
    holder_ready = threading.Event()
    release = threading.Event()

    def _hold_write_lock() -> None:
        con = sqlite3.connect(db, timeout=30)
        con.execute("BEGIN IMMEDIATE")
        con.execute("UPDATE images SET provider_status = 'holding'")
        holder_ready.set()
        release.wait(timeout=10)
        con.rollback()
        con.close()

    holder = threading.Thread(target=_hold_write_lock, daemon=True)
    holder.start()
    assert holder_ready.wait(timeout=10), "could not acquire the competing write lock"

    try:
        started = time.monotonic()
        resp = client.post(f"/api/images/{sha}/label", json={"label": "lion"})
        elapsed = time.monotonic() - started
    finally:
        release.set()
        holder.join(timeout=10)

    assert resp.status_code == 409, f"expected a fast 409, got {resp.status_code}: {resp.text}"
    assert elapsed < 5.0, f"the GUI waited {elapsed:.1f}s; its busy_timeout is 250 ms"

    tree_after = sorted(
        str(p.relative_to(output)) for p in output.rglob("*")
        if p.is_file() and not p.name.startswith(".catalog")
    )
    assert tree_after == tree_before, "a refused re-tag must not move anything"

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT label, label_source FROM images WHERE sha256=?", (sha,)).fetchone()
        overrides = con.execute(
            "SELECT COUNT(*) AS n FROM overrides WHERE sha256=?", (sha,)
        ).fetchone()["n"]
    finally:
        con.close()
    assert row["label"] == "unknown", "the label is unchanged"
    assert overrides == 0, "no override row was written for a refused re-tag"
