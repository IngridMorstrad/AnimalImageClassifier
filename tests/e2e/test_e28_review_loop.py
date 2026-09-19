"""E28: the active-learning loop — review → label → export → train → better labels.

The loop that makes this tool useful on *your* animals rather than COCO's: anything
the model cannot confidently name is queued for a human, the human's labels become
training data, and the next model names more of them automatically.

Asserted here end to end:

1. with no species model, every detected animal is ``unknown`` and **all** of it is in
   the review queue (never-scored rows first — they are the most informative);
2. labelling through the GUI empties the queue and files the photos under real species
   directories;
3. ``export-trainset`` turns those labels into a manifest with the dominant box;
4. ``train`` consumes it — including the small-manifest case where ``split_for`` puts
   every photograph on one side, which is what a real review session produces;
5. re-classifying with that model assigns **real species names**, and the review queue
   **shrinks**. That last assertion is the whole point: the loop has to actually
   reduce the human's work, not just run.

Also pins §5.9's rule that a *scored* sub-threshold ``unknown`` keeps its confidence —
without it the queue cannot rank "probably a zebra, 0.41" ahead of "no idea at all".
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import make_client

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
IMAGES = REPO / "data" / "raw" / "val2017"
FROZEN = Path(__file__).parent / "data" / "coco_species.json"
BACKBONE = REPO / "models" / "backbones" / "efficientnet_b0_ra-3dd342df.pth"
WEIGHTS = REPO / "models" / "md_v5a.0.0.pt"


def _staged() -> bool:
    return IMAGES.is_dir() and FROZEN.is_file() and BACKBONE.is_file() and WEIGHTS.is_file()


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _staged(), reason="needs the COCO stage + backbone + detector weights"
    ),
]


def _classify(cli_path, card, output, *extra):
    return subprocess.run(
        [cli_path, "classify", str(card), "--output", str(output), *extra],
        capture_output=True, text=True, check=False,
    )


def _catalog(output):
    con = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


@pytest.fixture(scope="module")
def card(tmp_path_factory):
    """Twelve real animal photographs, standing in for 'the user's own photos'."""
    root = tmp_path_factory.mktemp("own") / "card"
    dcim = root / "DCIM"
    dcim.mkdir(parents=True)
    truth: dict[str, str] = {}
    for entry in json.loads(FROZEN.read_text())["images"][:12]:
        source = IMAGES / entry["file_name"]
        if source.is_file():
            shutil.copy2(source, dcim / entry["file_name"])
            truth[entry["file_name"]] = entry["label"]
    if len(truth) < 6:
        pytest.skip("not enough staged photographs")
    return root, truth


def test_the_loop_closes_and_reduces_the_humans_work(cli_path, card, tmp_path):
    root, truth = card

    # ---- 1. First run: no species model, so nothing is confidently named. --------
    first = tmp_path / "pics1"
    result = _classify(cli_path, root, first)
    assert result.returncode in {0, 4}, result.stderr
    with _catalog(first) as con:
        labels = {r["label"] for r in con.execute("SELECT DISTINCT label FROM images")}
    assert labels == {"unknown"}, f"with no model every animal is unknown, got {labels}"

    client = make_client(first, allow_new_labels=True)
    queue = client.get("/api/review").json()
    assert queue["total"] == len(truth), "every un-named animal is queued for review"
    assert queue["review_below"] == pytest.approx(0.60), "the documented default"
    assert queue["items"][0]["confidence"] is None, (
        "never-scored rows come first — they are the most informative to label"
    )

    # ---- 2. The human labels them through the GUI. ------------------------------
    with _catalog(first) as con:
        name_of = {
            r["sha256"]: Path(r["path"]).name
            for r in con.execute("SELECT sha256, path FROM sources")
        }
    for item in queue["items"]:
        label = truth[name_of[item["sha256"]]]
        resp = client.post(
            f"/api/images/{item['sha256']}/label", json={"label": label}
        )
        assert resp.status_code == 200, resp.text

    assert client.get("/api/review").json()["total"] == 0, (
        "a labelled image leaves the queue — it is settled"
    )
    species_dirs = {
        p.name for p in first.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    } - {"unknown", "landscape", "junk", "multiple"}
    assert species_dirs, "the photos now sit under real species directories"

    # ---- 3. Export those labels as training data. -------------------------------
    manifest = tmp_path / "mine.jsonl"
    export = subprocess.run(
        [cli_path, "export-trainset", "--output", str(first),
         "--destination", str(manifest)],
        capture_output=True, text=True, check=False,
    )
    assert export.returncode == 0, export.stderr
    lines = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(lines) == len(truth), "every human label became a training sample"
    assert all("box" in line for line in lines), "each carries its dominant box"

    # ---- 4. Train on the human's own labels. ------------------------------------
    artifact = tmp_path / "mine.acmodel"
    train = subprocess.run(
        [cli_path, "train", "--manifest", str(manifest), "--arch", "efficientnet_b0",
         "--out", str(artifact), "--backbone-weights", str(BACKBONE),
         "--input-size", "128", "--epochs-head", "2", "--epochs-finetune", "0",
         "--batch-size", "8", "--jobs", "8"],
        capture_output=True, text=True, check=False,
    )
    assert train.returncode == 0, train.stderr
    assert artifact.is_file()

    from animal_classifier.classify.artifact import load

    loaded = load(artifact)
    assert set(loaded.label_slugs), "the model learned the human's label space"

    # ---- 5. Re-classify: real species names, and a shorter queue. ---------------
    second = tmp_path / "pics2"
    again = _classify(cli_path, root, second, "--species-model", str(artifact))
    assert again.returncode in {0, 4}, again.stderr

    with _catalog(second) as con:
        new_labels = {
            r["label"] for r in con.execute("SELECT DISTINCT label FROM images")
        }
    named = new_labels - {"unknown", "landscape", "junk", "multiple"}
    assert named, (
        f"after one round of labelling the model must name some animals itself, "
        f"got {new_labels}"
    )

    shorter = make_client(second).get("/api/review").json()
    assert shorter["total"] < len(truth), (
        f"the loop must reduce the human's work: queue went {len(truth)} -> "
        f"{shorter['total']}"
    )
    # The remaining queue is ordered least-confident-first among scored rows.
    scored = [i["confidence"] for i in shorter["items"] if i["confidence"] is not None]
    assert scored == sorted(scored), "least confident first"


def test_a_scored_sub_threshold_unknown_keeps_its_confidence(cli_path, card, tmp_path):
    """§5.9: NULL means 'no confidence by construction', not 'below the gate'.

    Without this the review queue cannot rank a near-miss above a total unknown.
    """
    root, _ = card
    output = tmp_path / "pics"
    # A model trained on synthetic shapes will score these animals confidently-wrongly
    # or weakly; either way the score is real and must be recorded.
    artifact = tmp_path / "tiny.acmodel"
    subprocess.run(
        [cli_path, "train", "--dataset", "synthetic", "--arch", "tinycnn",
         "--out", str(artifact), "--input-size", "64", "--epochs-head", "1",
         "--epochs-finetune", "0", "--batch-size", "16"],
        capture_output=True, text=True, check=False,
    )
    result = _classify(
        cli_path, root, output,
        "--species-model", str(artifact), "--min-confidence", "0.999",
    )
    assert result.returncode in {0, 4}, result.stderr

    with _catalog(output) as con:
        rows = con.execute(
            "SELECT label, confidence FROM images WHERE label = 'unknown'"
        ).fetchall()
    assert rows, "the impossible gate forces unknown"
    scored = [r for r in rows if r["confidence"] is not None]
    assert scored, (
        "a sub-threshold unknown produced by a *scored* box must keep its confidence "
        "(§5.9) — otherwise the review queue cannot rank it"
    )
    for row in scored:
        assert 0.0 <= row["confidence"] < 0.999
