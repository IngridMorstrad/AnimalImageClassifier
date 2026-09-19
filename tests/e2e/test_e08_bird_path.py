"""E8: bird species file into human-readable directories — DEFECT 2 (§5.8, §7.1).

DEFECT 2 was that CUB-200's directory names (``022.Chuck_will_widow``) and its
apostrophe-carrying common names (``Chuck-will's-widow``) could leak into the output
tree as ``022_chuck_will_widow`` or an illegal name. The fix lives in the taxonomy
layer: :func:`slug` normalises common names to clean directory components, and
``cub_key_from_dirname`` strips the ``NNN.`` prefix. This test pins that a bird head
whose labels are real CUB names files into directories a human would recognise —
never a digit-prefixed or apostrophe-mangled one.

It trains a tiny bird head (``tinycnn``) on synthetic images labelled with real CUB
common names, then classifies through it, and asserts the destination directory is
the clean slug.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.slow

# Real CUB-200 common names with the orthography that broke DEFECT 2.
BIRD_NAMES = {
    "Chuck-will's-widow": "chuck_will_s_widow",
    "Brewer's Blackbird": "brewer_s_blackbird",
}


def test_slug_produces_clean_bird_directories() -> None:
    """The taxonomy guarantee behind E8, asserted directly (fast, no training)."""
    from animal_classifier.taxonomy import cub_key_from_dirname, slug

    for common, expected in BIRD_NAMES.items():
        got = slug(common)
        assert got == expected, f"{common!r} -> {got!r}, expected {expected!r}"
        assert not got[0].isdigit(), "no digit-prefixed directory (DEFECT 2)"
        assert "'" not in got and "." not in got

    # The NNN. directory prefix is stripped, never carried into the label.
    assert cub_key_from_dirname("022.Chuck_will_widow") == "chuck_will_widow"
    assert not cub_key_from_dirname("059.California_Gull")[0].isdigit()


def _train_bird_head(run_cli, tmp_path, labels):
    """Train a tinycnn 'bird head' on synthetic crops labelled with CUB names."""
    from PIL import Image

    data = tmp_path / "birddata"
    data.mkdir()
    lines = []
    for i, name in enumerate(labels):
        for j in range(6):
            # Distinct solid colour per class so tinycnn separates them trivially.
            img = data / f"{i}_{j}.png"
            Image.new("RGB", (64, 64), (40 + i * 80, 60, 200 - i * 60)).save(img)
            lines.append(json.dumps({"path": str(img.resolve()), "label": name,
                                     "split": "val" if j == 0 else "train"}))
    manifest = data / "birds.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    artifact = tmp_path / "birds.acmodel"
    result = run_cli(
        ["train", "--manifest", str(manifest), "--arch", "tinycnn", "--out", str(artifact),
         "--input-size", "64", "--epochs-head", "2", "--epochs-finetune", "1", "--batch-size", "8"]
    )
    assert result.returncode == 0, result.stderr
    return artifact


def test_bird_head_files_into_human_readable_directory(run_cli, tmp_path):
    labels = list(BIRD_NAMES.keys())
    artifact = _train_bird_head(run_cli, tmp_path, labels)

    # The artifact's slugs are the clean names.
    from animal_classifier.classify.artifact import load

    loaded = load(artifact)
    assert set(loaded.label_slugs) == set(BIRD_NAMES.values())
    for slug in loaded.label_slugs:
        assert not slug[0].isdigit(), slug

    # Classify a photo through it and assert the destination directory is a clean slug.
    from PIL import Image

    card = tmp_path / "card" / "DCIM"
    card.mkdir(parents=True)
    Image.new("RGB", (200, 200), (40, 60, 200)).save(card / "bird.jpg")
    (card / "bird.jpg.boxes.json").write_text(
        json.dumps([{"cls": "animal", "conf": 0.95, "x0": 0, "y0": 0, "x1": 200, "y1": 200}])
    )
    output = tmp_path / "pics"
    result = run_cli(
        ["classify", str(card.parent), "--output", str(output), "--detector", "scripted",
         "--species-model", str(artifact), "--min-confidence", "0.0"]
    )
    assert result.returncode in {0, 4}, result.stderr

    filed_dirs = {p.name for p in output.iterdir() if p.is_dir()}
    species_dirs = filed_dirs - {"landscape", "junk", "multiple", "unknown"}
    assert species_dirs, "the bird was filed into a species directory"
    for directory in species_dirs:
        assert directory in BIRD_NAMES.values(), directory
        assert not directory[0].isdigit(), f"digit-prefixed dir leaked: {directory}"
