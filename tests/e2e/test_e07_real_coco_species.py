"""E7: real training and real species identity on COCO (§7.2, §7.3, §11.2).

The full-strength version of the training proof, and the one that asserts the stated
accuracy thresholds instead of only the plumbing. It runs entirely through the real
CLI:

``build_coco_manifest.py`` (7 classes, ``--split train``) → ``train --arch
efficientnet_b0 --backbone-weights <timm> --input-size 128`` → ``eval --split val``
→ ``classify --detector megadetector --species-model <artifact>`` over the frozen
single-animal list in ``tests/e2e/data/coco_species.json``.

**Thresholds, and where they come from.** ``val_top1 >= 0.55`` is §11.2's gate
(chance is 0.143 across 7 classes, the majority-class baseline 0.229). The identity
count is ``>= ceil(0.8 * len(frozen list))``, read from the JSON — no literal. The
``--input-size 128`` is §7.3's stated test-budget trade-off against the shipped 224,
not a silent weakening.

**One naming subtlety, asserted rather than papered over.** The manifest label for
COCO category 21 is ``cow``, but the taxonomy's common name for it is ``Cattle``, so
the artifact's slug — and therefore the output directory — is ``cattle``. The
taxonomy owns that mapping deliberately (it files under the human-readable name), so
the test compares through it instead of assuming the manifest key reaches the disk.

Skips cleanly (never fails) when the COCO stage, the backbone or the detector weights
are absent — see ``docs/ARTIFACTS.md`` for the staging commands.
"""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess
from pathlib import Path

import pytest

from animal_classifier.detect.megadetector import EXPECTED_BYTES

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[2]
ANNOTATIONS = REPO / "data" / "raw" / "annotations" / "instances_val2017.json"
IMAGES = REPO / "data" / "raw" / "val2017"
BACKBONE = REPO / "models" / "backbones" / "efficientnet_b0_ra-3dd342df.pth"
WEIGHTS = REPO / "models" / "md_v5a.0.0.pt"
FROZEN = Path(__file__).parent / "data" / "coco_species.json"

E7_CLASSES = "zebra,elephant,giraffe,bear,cow,sheep,bird"
VAL_TOP1_GATE = 0.55

#: The taxonomy files COCO's ``cow`` under its common name ``Cattle`` (§13.3), so the
#: label that reaches disk is ``cattle``. Mapping the frozen list's GT key through
#: this is the honest comparison.
GT_TO_SLUG = {"cow": "cattle"}


def _staged() -> bool:
    return (
        ANNOTATIONS.is_file()
        and IMAGES.is_dir()
        and BACKBONE.is_file()
        and WEIGHTS.is_file()
        and WEIGHTS.stat().st_size == EXPECTED_BYTES
        and FROZEN.is_file()
    )


pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not _staged(),
        reason=(
            "E7 needs the COCO stage (data/raw/val2017 + annotations), the timm "
            "backbone and the MegaDetector weights — see docs/ARTIFACTS.md"
        ),
    ),
]


@pytest.fixture(scope="module")
def trained(cli_path, tmp_path_factory):
    """Build the manifests, run a real finetune, and eval it. Module-scoped: once."""
    work = tmp_path_factory.mktemp("e7")
    manifests = {}
    for split in ("train", "val"):
        out = work / f"e7_{split}.jsonl"
        result = subprocess.run(
            ["uv", "run", "--frozen", "python", str(REPO / "scripts" / "build_coco_manifest.py"),
             "--annotations", str(ANNOTATIONS), "--images", str(IMAGES),
             "--out", str(out), "--classes", E7_CLASSES, "--split", split],
            cwd=REPO, capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, result.stderr
        assert out.is_file()
        manifests[split] = out

    artifact = work / "species.acmodel"
    train = subprocess.run(
        [cli_path, "train", "--manifest", str(manifests["train"]),
         "--arch", "efficientnet_b0", "--out", str(artifact),
         "--backbone-weights", str(BACKBONE), "--input-size", "128",
         "--epochs-head", "2", "--epochs-finetune", "0",
         "--batch-size", "32", "--jobs", "8"],
        capture_output=True, text=True, check=False,
    )
    assert train.returncode == 0, train.stderr
    return work, manifests, artifact, train.stderr


def test_manifests_have_the_expected_support(trained):
    """§11.2's 1,625 train / 349 val crops over the seven classes."""
    _, manifests, _, _ = trained
    train_lines = manifests["train"].read_text().splitlines()
    val_lines = manifests["val"].read_text().splitlines()
    assert len(train_lines) > 1000, len(train_lines)
    assert len(val_lines) > 200, len(val_lines)
    classes = {json.loads(line)["label"] for line in train_lines}
    assert classes == set(E7_CLASSES.split(",")), classes
    # The leakage rule: no image basename appears on both sides (§7.2).
    train_names = {Path(json.loads(line)["path"]).name for line in train_lines}
    val_names = {Path(json.loads(line)["path"]).name for line in val_lines}
    assert not (train_names & val_names), "train/val must not share a photograph"


def test_artifact_metadata_and_identity_leg(trained):
    """§7.1: the artifact is self-describing and provably loadable by classify."""
    from animal_classifier.classify.artifact import load

    _, _, artifact, _ = trained
    loaded = load(artifact)
    assert loaded.arch == "efficientnet_b0"
    assert loaded.input_size == 128
    assert loaded.temperature == 1.0, "a freshly trained model has identity temperature"
    assert len(loaded.labels) == 7
    for entry in loaded.labels:
        assert entry.rank in {"species", "genus", "family", "order", "class"}, entry

    import torch

    blob = torch.load(artifact, map_location="cpu", weights_only=False)
    assert blob["train"]["calibrated_from"] is None
    assert blob["train"]["val_top1"] > 0.0


def test_val_top1_clears_the_gate(trained):
    """§11.2: ``val_top1 >= 0.55`` (chance 0.143, majority baseline 0.229)."""
    _, _, artifact, _ = trained
    import torch

    blob = torch.load(artifact, map_location="cpu", weights_only=False)
    val_top1 = blob["train"]["val_top1"]
    print(f"  val_top1 = {val_top1:.3f} (gate >= {VAL_TOP1_GATE})")
    assert val_top1 >= VAL_TOP1_GATE, (
        f"val_top1 {val_top1:.3f} is below §11.2's {VAL_TOP1_GATE}. The sanctioned "
        "remedy is raising --input-size toward the shipped 224 (§7.3), never "
        "lowering this threshold."
    )


def test_eval_agrees_with_the_artifact(trained, cli_path):
    """``eval`` recomputes the artifact's own recorded top-1 (§7.4)."""
    _, manifests, artifact, _ = trained
    result = subprocess.run(
        [cli_path, "eval", "--model", str(artifact),
         "--manifest", str(manifests["val"]), "--split", "val"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    top1 = float(result.stdout.split("top1=")[1].split()[0])
    import torch

    recorded = torch.load(artifact, map_location="cpu", weights_only=False)["train"]["val_top1"]
    assert abs(top1 - recorded) < 1e-6, f"eval {top1} vs artifact {recorded}"


def test_real_photos_get_their_ground_truth_species(trained, cli_path, tmp_path):
    """The whole point: real photos, real detector, real head → the right species."""
    import shutil

    _, _, artifact, _ = trained
    frozen = json.loads(FROZEN.read_text())["images"]

    card = tmp_path / "card" / "DCIM"
    card.mkdir(parents=True)
    truth: dict[str, str] = {}
    for entry in frozen:
        source = IMAGES / entry["file_name"]
        if source.is_file():
            shutil.copy2(source, card / entry["file_name"])
            truth[entry["file_name"]] = entry["label"]
    assert truth, "no frozen images available"

    output = tmp_path / "pics"
    result = subprocess.run(
        [cli_path, "classify", str(card.parent), "--output", str(output),
         "--detector", "megadetector", "--species-model", str(artifact)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode in {0, 4}, result.stderr

    con = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        predicted = {
            Path(row["path"]).name: row["label"]
            for row in con.execute(
                "SELECT s.path, i.label FROM images i JOIN sources s USING(sha256)"
            )
        }
    finally:
        con.close()

    hits = 0
    for name, gt in sorted(truth.items()):
        expected = GT_TO_SLUG.get(gt, gt)
        got = predicted.get(name)
        ok = got == expected
        hits += ok
        print(f"  {name} truth={gt:<9} predicted={got:<10} {'ok' if ok else 'MISS'}")

    required = math.ceil(0.8 * len(truth))
    print(f"  identity: {hits}/{len(truth)} correct (need >= {required})")
    assert hits >= required, (
        f"only {hits}/{len(truth)} frozen single-animal photos received their "
        f"ground-truth species through the real CLI (need >= {required})"
    )
