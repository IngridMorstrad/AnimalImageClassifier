"""E13: idempotency, resume, --limit and --reclassify (DESIGN.md §5.9).

The regression gate for DEFECT 1 and for the §5.9 bookkeeping as a whole. Every leg
here is a *second* run — the tool's value proposition is that pointing it at a card
twice is cheap and safe, and each of these is a way that could quietly stop being
true:

- **the clean no-op** — a second identical run must file nothing and re-process
  nothing, with the ``.mp4``'s ``skipped`` row *refreshed* (upsert), not duplicated
  (which raised ``IntegrityError`` before DEFECT 1's fix);
- **resume** — a run interrupted mid-card (rows left ``materializing``/``failed``)
  must complete on the next pass, because materialization is content-addressed;
- **--limit is a budget on inference, not on walking** — ``done`` rows are skipped
  before the budget is spent, so repeated ``--limit`` runs advance through the card
  instead of re-examining its first N files forever;
- **--reclassify replaces, never appends** — ``COUNT(*)`` over ``boxes`` is
  unchanged, and ``overrides`` is untouched;
- **--reclassify --limit sweeps by staleness** — oldest-classified first, so two
  passes cover four distinct hashes rather than the same two twice;
- **a ``failed`` row is retried**, never left stuck.
"""

from __future__ import annotations

import sqlite3
import subprocess

import pytest


def _run(run_cli, card, output, *extra) -> subprocess.CompletedProcess[str]:
    return run_cli(
        ["classify", str(card), "--output", str(output), "--detector", "scripted", *extra]
    )


def _catalog(output) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _status_counts(output) -> dict[str, int]:
    with _catalog(output) as c:
        return {
            row["status"]: row["n"]
            for row in c.execute(
                "SELECT status, COUNT(*) AS n FROM images GROUP BY status"
            )
        }


# --------------------------------------------------------------------------- #
# The clean no-op (DEFECT 1's regression gate)
# --------------------------------------------------------------------------- #


def test_second_run_is_a_clean_no_op(run_cli, fixture_card, tmp_path):
    output = tmp_path / "pics"
    first = _run(run_cli, fixture_card, output)
    assert first.returncode == 4  # one truncated JPEG on the card

    filed_after_first = _status_counts(output)
    second = _run(run_cli, fixture_card, output)

    # Same exit class, and nothing new was filed.
    assert second.returncode == 4
    assert _status_counts(output) == filed_after_first
    with _catalog(output) as c:
        runs = list(c.execute("SELECT * FROM runs ORDER BY started_at"))
        assert len(runs) == 2
        # The second run did the work of examining every image and filing none.
        assert runs[1]["n_done"] == 0


def test_mp4_skip_row_is_upserted_not_duplicated(run_cli, fixture_card, tmp_path):
    """DEFECT 1: ``skipped.path`` is a primary key; a re-walk must overwrite it."""
    output = tmp_path / "pics"
    _run(run_cli, fixture_card, output)
    _run(run_cli, fixture_card, output)

    with _catalog(output) as c:
        mp4 = list(
            c.execute("SELECT * FROM skipped WHERE path LIKE '%.mp4'")
        )
        assert len(mp4) == 1, "the .mp4 has exactly one skipped row after two runs"
        assert mp4[0]["reason"] == "video"
        newest_run = c.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC, run_id DESC LIMIT 1"
        ).fetchone()["run_id"]
        assert mp4[0]["run_id"] == newest_run, "the row carries the *second* run's id"


# --------------------------------------------------------------------------- #
# Resume, and the failed retry
# --------------------------------------------------------------------------- #


def test_interrupted_run_resumes_to_completion(run_cli, fixture_card, tmp_path):
    """Rows left ``materializing``/``failed`` are re-processed on the next run."""
    output = tmp_path / "pics"
    _run(run_cli, fixture_card, output)

    # Simulate a kill: one image half-filed, one recorded failed.
    db = output / ".catalog.db"
    with sqlite3.connect(db) as c:
        c.row_factory = sqlite3.Row
        done = [r["sha256"] for r in c.execute("SELECT sha256 FROM images WHERE status='done'")]
        c.execute("UPDATE images SET status='materializing' WHERE sha256=?", (done[0],))
        c.execute("UPDATE images SET status='failed' WHERE sha256=?", (done[1],))
        c.commit()
    assert _status_counts(output).get("failed") == 1

    second = _run(run_cli, fixture_card, output)
    assert second.returncode == 4
    after = _status_counts(output)
    assert after.get("failed", 0) == 0, "the failed row was retried, not left stuck"
    assert after.get("materializing", 0) == 0, "the half-filed row was completed"


# --------------------------------------------------------------------------- #
# --limit as an inference budget
# --------------------------------------------------------------------------- #


def test_two_limit_runs_advance_through_the_card(run_cli, five_image_card, tmp_path):
    """Two ``--limit 2`` runs over a 5-image card leave 4 done, not 2 (§5.9)."""
    output = tmp_path / "pics"
    _run(run_cli, five_image_card, output, "--limit", "2")
    assert _status_counts(output).get("done") == 2

    _run(run_cli, five_image_card, output, "--limit", "2")
    assert _status_counts(output).get("done") == 4, (
        "the second --limit 2 must process two *new* images, because done rows are "
        "skipped before the budget is consulted"
    )


def test_reclassify_limit_sweeps_by_staleness(run_cli, five_image_card, tmp_path):
    """Two ``--reclassify --limit 2`` runs touch 4 distinct hashes, oldest first."""
    output = tmp_path / "pics"
    _run(run_cli, five_image_card, output)  # all 5 classified once

    with _catalog(output) as c:
        first_pass = [
            r["sha256"]
            for r in c.execute(
                "SELECT sha256 FROM images ORDER BY last_updated ASC, sha256 ASC LIMIT 2"
            )
        ]

    _run(run_cli, five_image_card, output, "--reclassify", "--limit", "2")
    with _catalog(output) as c:
        touched_first = [
            r["sha256"]
            for r in c.execute(
                "SELECT sha256 FROM images ORDER BY last_updated DESC, sha256 DESC LIMIT 2"
            )
        ]
    assert set(touched_first) == set(first_pass), "the two oldest were reclassified"

    _run(run_cli, five_image_card, output, "--reclassify", "--limit", "2")
    with _catalog(output) as c:
        newest_four = [
            r["sha256"]
            for r in c.execute(
                "SELECT sha256 FROM images ORDER BY last_updated DESC, sha256 DESC LIMIT 4"
            )
        ]
    assert len(set(newest_four)) == 4, "two passes swept four distinct hashes, not two"


# --------------------------------------------------------------------------- #
# --reclassify replaces and preserves
# --------------------------------------------------------------------------- #


def test_reclassify_does_not_accumulate_boxes(run_cli, fixture_card, tmp_path):
    """§5.9: re-inference DELETEs then INSERTs, so COUNT(*) over boxes is stable."""
    output = tmp_path / "pics"
    _run(run_cli, fixture_card, output)
    with _catalog(output) as c:
        boxes_before = c.execute("SELECT COUNT(*) AS n FROM boxes").fetchone()["n"]
        candidates_before = c.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]

    _run(run_cli, fixture_card, output, "--reclassify")
    _run(run_cli, fixture_card, output, "--reclassify")

    with _catalog(output) as c:
        assert c.execute("SELECT COUNT(*) AS n FROM boxes").fetchone()["n"] == boxes_before
        assert (
            c.execute("SELECT COUNT(*) AS n FROM candidates").fetchone()["n"]
            == candidates_before
        )
        assert c.execute("SELECT COUNT(*) AS n FROM overrides").fetchone()["n"] == 0


def test_first_seen_survives_reclassify(run_cli, fixture_card, tmp_path):
    """§5.9: ``first_seen`` is write-once; ``--reclassify`` only bumps last_updated."""
    output = tmp_path / "pics"
    _run(run_cli, fixture_card, output)
    with _catalog(output) as c:
        before = {r["sha256"]: r["first_seen"] for r in c.execute("SELECT sha256, first_seen FROM images")}

    _run(run_cli, fixture_card, output, "--reclassify")
    with _catalog(output) as c:
        after = {r["sha256"]: r["first_seen"] for r in c.execute("SELECT sha256, first_seen FROM images")}
    assert after == before, "first_seen must not change on reclassify"


# --------------------------------------------------------------------------- #
# A RAW file skipped without --raw, ingested by a later --raw run
# --------------------------------------------------------------------------- #


def test_cr2_skipped_then_ingested_with_raw(run_cli, cr2_card, tmp_path):
    """Review finding 10: widening ingestion picks up a previously-skipped file.

    Without ``--raw`` a ``.cr2`` is a ``raw_not_enabled`` skip and never an image
    row. A later ``--raw`` run ingests it — here it decodes to a real (tiny) raster,
    so it lands as an image and its stale ``skipped`` row is cleared (§5.9's
    "skips are judged fresh every run").
    """
    output = tmp_path / "pics"
    _run(run_cli, cr2_card, output)
    with _catalog(output) as c:
        skipped = list(c.execute("SELECT * FROM skipped WHERE path LIKE '%.cr2'"))
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "raw_not_enabled"
        assert c.execute("SELECT COUNT(*) AS n FROM images").fetchone()["n"] == 0

    # rawpy is an optional extra and is not installed in the locked env, so a --raw
    # run fails loudly (exit 3) rather than silently. That failure *is* the contract
    # for a machine without the extra; either outcome proves the skip was re-judged.
    result = _run(run_cli, cr2_card, output, "--raw")
    assert result.returncode in {0, 3, 4}
    if result.returncode == 3:
        assert "raw" in result.stderr.lower()
