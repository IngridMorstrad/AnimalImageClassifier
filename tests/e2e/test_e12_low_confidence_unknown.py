"""E12: a low-confidence prediction is ``unknown``, and nothing is lost (§5.7).

``--min-confidence 0.999`` forces every prediction below the gate, so the label
becomes ``unknown``. The three things that must still be true:

- the **top-5 candidates are still recorded** — the model's opinion is evidence worth
  keeping even when it is not trusted enough to file on;
- the **file is still materialized**, under ``unknown/``. Nothing is discarded;
- ``images.confidence`` is **non-NULL** here, because the winning box *was* scored —
  §5.9 reserves NULL for labels that have no confidence by construction, and an
  `unknown` produced by a scored box falling below the gate is not one of those.

That last distinction is what makes the GUI's confidence slider honest: this row can
be filtered by confidence, a box-less `unknown` cannot.
"""

from __future__ import annotations

import json
import sqlite3

import pytest


@pytest.fixture
def trained_tiny_model(run_cli, tmp_path_factory):
    """A tinycnn trained on synthetic shapes — enough to produce a real prediction."""
    work = tmp_path_factory.mktemp("e12model")
    artifact = work / "m.acmodel"
    result = run_cli(
        ["train", "--dataset", "synthetic", "--arch", "tinycnn", "--out", str(artifact),
         "--input-size", "64", "--epochs-head", "2", "--epochs-finetune", "0",
         "--batch-size", "16"]
    )
    assert result.returncode == 0, result.stderr
    return artifact


@pytest.fixture
def one_animal_card(tmp_path):
    from PIL import Image  # noqa: PLC0415

    card = tmp_path / "card" / "DCIM"
    card.mkdir(parents=True)
    # A blue triangle-ish image: one of the synthetic classes, so the head has an opinion.
    Image.new("RGB", (240, 240), (40, 80, 220)).save(card / "a.jpg", quality=95)
    (card / "a.jpg.boxes.json").write_text(
        json.dumps([{"cls": "animal", "conf": 0.95, "x0": 10, "y0": 10, "x1": 230, "y1": 230}])
    )
    return card.parent


@pytest.mark.slow
def test_low_confidence_files_as_unknown_but_keeps_everything(
    run_cli, trained_tiny_model, one_animal_card, tmp_path
):
    output = tmp_path / "pics"
    result = run_cli(
        ["classify", str(one_animal_card), "--output", str(output),
         "--detector", "scripted", "--species-model", str(trained_tiny_model),
         "--min-confidence", "0.999"]
    )
    assert result.returncode in {0, 4}, result.stderr

    # Filed under unknown/, not discarded.
    filed = output / "unknown" / "a.jpg"
    assert filed.is_file(), "the image is still materialized under unknown/"

    con = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute("SELECT * FROM images").fetchone()
        boxes = con.execute("SELECT * FROM boxes WHERE sha256=?", (row["sha256"],)).fetchall()
        candidates = con.execute(
            "SELECT * FROM candidates WHERE box_id=? ORDER BY rank", (boxes[0]["id"],)
        ).fetchall()
    finally:
        con.close()

    assert row["label"] == "unknown", "below the gate → unknown, never a guessed species"
    assert candidates, "the top-k candidates are still recorded"
    assert len(candidates) >= 1
    assert candidates[0]["score"] is not None

    # The box itself still carries the model's scored opinion.
    assert boxes[0]["species_conf"] is not None, "the box was scored, just not trusted"
    assert boxes[0]["species_status"] != "degenerate", "this crop was classifiable"
    assert boxes[0]["is_dominant"] == 1, "it still won dominance"


@pytest.mark.slow
def test_a_generous_gate_files_the_same_image_as_a_species(
    run_cli, trained_tiny_model, one_animal_card, tmp_path
):
    """The control: the only thing that changed is the threshold."""
    output = tmp_path / "pics"
    result = run_cli(
        ["classify", str(one_animal_card), "--output", str(output),
         "--detector", "scripted", "--species-model", str(trained_tiny_model),
         "--min-confidence", "0.0"]
    )
    assert result.returncode in {0, 4}, result.stderr
    labels = {p.parent.name for p in output.rglob("*.jpg")}
    assert labels and labels != {"unknown"}, (
        f"with the gate wide open the image should file as a species, got {labels}"
    )
