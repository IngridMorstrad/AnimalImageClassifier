"""E20: the override lifecycle — ``--ignore-overrides`` suppresses, never destroys (§5.9).

The sequence, and why each step matters:

1. **re-tag** → the human label is filed and an ``overrides`` row is appended.
2. **``--reclassify``** → the human label is *kept*. A model re-run must not quietly
   undo a correction the user made.
3. **``--reclassify --ignore-overrides``** → the model label is restored **and the
   ``overrides`` row is still there**. This is the assertion that makes the flag's
   name honest: it suppresses the override for one run, it does not delete it.
4. **``--reclassify`` again** → the human label and its destination *return*, because
   the override was never destroyed.

Step 4 is the one that would fail if `--ignore-overrides` had been implemented as a
delete, and it is the reason the design forbids that: permanently discarding a
correction is done by re-tagging (which appends a newer override), never by a flag.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import build_card, classify, make_client


def _catalog(output):
    con = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _row(output, sha):
    with _catalog(output) as con:
        return con.execute(
            "SELECT label, label_source, dest_path FROM images WHERE sha256=?", (sha,)
        ).fetchone()


def _override_count(output, sha):
    with _catalog(output) as con:
        return con.execute(
            "SELECT COUNT(*) AS n FROM overrides WHERE sha256=?", (sha,)
        ).fetchone()["n"]


def test_ignore_overrides_suppresses_without_destroying(cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)

    client = make_client(output, allow_new_labels=True)
    target = next(
        i for i in client.get("/api/images").json()["items"] if i["label"] == "unknown"
    )
    sha = target["sha256"]

    # 1. Re-tag to a human label.
    assert client.post(f"/api/images/{sha}/label", json={"label": "lion"}).status_code == 200
    after_retag = _row(output, sha)
    assert after_retag["label"] == "lion"
    assert after_retag["label_source"] == "human"
    assert _override_count(output, sha) == 1
    assert (output / "lion").is_dir()

    # 2. --reclassify keeps the human label.
    classify(cli_path, card, output, "--reclassify")
    kept = _row(output, sha)
    assert kept["label"] == "lion", "a model re-run must not undo a human correction"
    assert kept["label_source"] == "human"
    assert Path(kept["dest_path"]).parent.name == "lion"

    # 3. --ignore-overrides restores the model label BUT keeps the override row.
    classify(cli_path, card, output, "--reclassify", "--ignore-overrides")
    suppressed = _row(output, sha)
    assert suppressed["label"] == "unknown", "the model's own decision is restored"
    assert suppressed["label_source"] == "model"
    assert _override_count(output, sha) == 1, (
        "--ignore-overrides must SUPPRESS, not delete: the overrides row is the "
        "human's decision and is append-only (I6)"
    )
    assert Path(suppressed["dest_path"]).parent.name == "unknown"

    # 4. A later run without the flag re-applies the override — it was never lost.
    classify(cli_path, card, output, "--reclassify")
    restored = _row(output, sha)
    assert restored["label"] == "lion", (
        "the human label must return: --ignore-overrides only suppressed it for one run"
    )
    assert restored["label_source"] == "human"
    assert Path(restored["dest_path"]).parent.name == "lion"
    assert (output / "lion" / Path(restored["dest_path"]).name).exists()


def test_overrides_are_append_only_across_two_retags(cli_path, tmp_path):
    """I6: a second correction appends; the first is never rewritten."""
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    client = make_client(output, allow_new_labels=True)
    sha = next(
        i for i in client.get("/api/images").json()["items"] if i["label"] == "unknown"
    )["sha256"]

    client.post(f"/api/images/{sha}/label", json={"label": "lion"})
    client.post(f"/api/images/{sha}/label", json={"label": "leopard"})

    with _catalog(output) as con:
        rows = con.execute(
            "SELECT old_label, new_label FROM overrides WHERE sha256=? ORDER BY id", (sha,)
        ).fetchall()
    assert len(rows) == 2, "both corrections are recorded"
    assert rows[0]["new_label"] == "lion"
    assert rows[1]["old_label"] == "lion", "the second records what it superseded"
    assert rows[1]["new_label"] == "leopard"
    assert _row(output, sha)["label"] == "leopard", "the newest override wins"
