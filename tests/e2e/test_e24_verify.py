"""E24: verify's specific exit codes and the two --fix reconciliations (§8).

The exit codes are the documented ones, not a generic non-zero:

- ``3`` for a required asset missing (no catalog at all);
- ``4`` for a catalog/filesystem inconsistency (a missing ``dest_path``, a pending
  re-tag), and for a finding ``--fix`` will not touch;
- ``0`` after ``--fix`` completes a pending re-tag.

The pending re-tag is the §5.8 crash window: an ``overrides`` row states the intent,
the row is ``materializing``, the file still sits at the old label. ``--fix``
completes the rename forward and never invents intent.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import build_card, classify  # noqa: E402


def test_missing_catalog_is_exit_3(run_cli, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = run_cli(["verify", "--output", str(empty)])
    assert result.returncode == 3, result.stderr


def test_clean_tree_is_exit_0(run_cli, cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    result = run_cli(["verify", "--output", str(output)])
    assert result.returncode == 0, result.stderr


def test_missing_dest_file_is_exit_4(run_cli, cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    # Delete a filed image behind the tool's back → catalog/fs disagree.
    filed = next(p for p in output.rglob("*.jpg") if not p.name.startswith(".catalog"))
    filed.unlink()
    result = run_cli(["verify", "--output", str(output)])
    assert result.returncode == 4, result.stderr


def test_pending_retag_is_exit_4_then_fixed(run_cli, cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)

    # Manufacture the §5.8 crash window: an image left 'materializing' with an
    # overrides row and the file still at the old label path.
    db = output / ".catalog.db"
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT sha256, label, dest_path FROM images WHERE label='unknown'").fetchone()
    sha, old_label, old_dest = row["sha256"], row["label"], Path(row["dest_path"])
    new_dir = output / "lion"
    new_dir.mkdir(exist_ok=True)
    new_dest = new_dir / old_dest.name
    con.execute(
        "INSERT INTO overrides(sha256, old_label, new_label, created_at, note) VALUES(?,?,?,?,?)",
        (sha, old_label, "lion", "2026-01-01T00:00:00Z", "pending"),
    )
    con.execute(
        "UPDATE images SET status='materializing', label='lion', label_source='human', dest_path=? WHERE sha256=?",
        (str(new_dest), sha),
    )
    con.commit()
    con.close()
    assert old_dest.exists() and not new_dest.exists(), "the crash window: file still at old path"

    # verify reports it (exit 4)...
    assert run_cli(["verify", "--output", str(output)]).returncode == 4

    # ...and --fix completes the rename forward (exit 0).
    fixed = run_cli(["verify", "--output", str(output), "--fix"])
    assert fixed.returncode == 0, fixed.stderr
    assert new_dest.exists(), "the rename was completed forward"
    assert not old_dest.exists()

    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    status = con.execute("SELECT status FROM images WHERE sha256=?", (sha,)).fetchone()["status"]
    con.close()
    assert status == "done"


def test_json_output_is_machine_readable(run_cli, cli_path, tmp_path):
    import json  # noqa: PLC0415

    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    result = run_cli(["verify", "--output", str(output), "--json"])
    report = json.loads(result.stdout)
    assert "checks" in report
    assert all("status" in c for c in report["checks"])
