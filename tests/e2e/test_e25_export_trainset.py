"""E25: export-trainset's explicit label filter and counted summary (§7.5).

Closes the training loop: human corrections become the next manifest. The load-
bearing assertion is that a ``junk`` override produces **no** manifest line by
default and is reported in the counted summary — so a user can see where re-tagged
images went instead of assuming data loss. Species labels export with their
dominant box; ``multiple``/``unknown`` are always skipped.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import build_card, classify, make_client


@pytest.fixture
def retagged_output(cli_path, tmp_path):
    """A classified card: one image re-tagged to a species (human), the model's
    ``landscape``/``junk`` scene labels left as model labels.

    ``junk`` and ``landscape`` are *reserved* (§5.8), so a human can never assign
    them in the GUI — they only ever occur as model labels. The filter's counted
    skips are therefore exercised via ``--include-model-labels``.
    """
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    client = make_client(output, allow_new_labels=True)
    items = client.get("/api/images").json()["items"]
    animal = next(i for i in items if i["label"] == "unknown")
    client.post(f"/api/images/{animal['sha256']}/label", json={"label": "lion"})
    return output


def test_species_exported_scene_skipped_by_default(run_cli, retagged_output, tmp_path):
    """Default: only human/high-conf species; model landscape+junk are skipped+counted."""
    manifest = tmp_path / "train.jsonl"
    result = run_cli(
        ["export-trainset", "--output", str(retagged_output), "--destination", str(manifest),
         "--include-model-labels", "--min-conf", "0.0"]
    )
    assert result.returncode == 0, result.stderr
    assert "exported=1" in result.stdout, "only the human-labelled lion is a species"
    assert "skipped_landscape=1" in result.stdout, "the model landscape is counted, not lost"
    assert "skipped_junk=1" in result.stdout, "the model junk is counted, not lost"

    lines = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert lines[0]["label"] == "lion"
    assert "box" in lines[0], "a species line carries the dominant box"


def test_include_non_species_emits_scene_classes(run_cli, retagged_output, tmp_path):
    manifest = tmp_path / "train.jsonl"
    result = run_cli(
        ["export-trainset", "--output", str(retagged_output), "--destination", str(manifest),
         "--include-model-labels", "--min-conf", "0.0", "--include-non-species"]
    )
    assert result.returncode == 0, result.stderr
    labels = {json.loads(line)["label"] for line in manifest.read_text().splitlines() if line.strip()}
    assert "junk" in labels, "with the flag, junk is exported as a scene class"
    assert "landscape" in labels
    assert "lion" in labels


def test_export_never_writes_the_source(run_cli, cli_path, tmp_path):
    """§7.5/I1: source images are read-only, and default export reads none of them."""
    import hashlib

    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    before = {
        str(p.relative_to(card)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(card.rglob("*")) if p.is_file()
    }
    run_cli(["export-trainset", "--output", str(output), "--destination", str(tmp_path / "t.jsonl")])
    after = {
        str(p.relative_to(card)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(card.rglob("*")) if p.is_file()
    }
    assert after == before
