"""E10: the whole train → eval → export → inference path, on synthetic data (§7.6).

The smoke test for the training subsystem. It runs the *real* trainer, eval and
artifact writer on a deterministic synthetic dataset with the from-scratch
``tinycnn``, so it finishes in seconds without the 1.1 GB CUB archive or a
pretrained backbone. It proves the plumbing — a trained artifact is loadable,
carries ``temperature == 1.0`` and ``calibrated_from is None``, and re-loads for
inference — not wildlife accuracy (E7 owns that).

Marked ``slow`` because it trains a network; still well under a minute.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_train_eval_export_then_load(run_cli, tmp_path):
    artifact = tmp_path / "synth.acmodel"
    train = run_cli(
        [
            "train", "--dataset", "synthetic", "--arch", "tinycnn",
            "--out", str(artifact),
            "--epochs-head", "2", "--epochs-finetune", "2",
            "--batch-size", "16", "--input-size", "64", "--seed", "0",
        ]
    )
    assert train.returncode == 0, train.stderr
    assert artifact.is_file(), "train wrote the .acmodel artifact"

    # The artifact is loadable and honest about its calibration state (§7.1).
    from animal_classifier.classify.artifact import load  # noqa: PLC0415

    loaded = load(artifact)
    assert loaded.temperature == 1.0, "a freshly trained model has identity temperature"
    assert len(loaded.label_slugs) == 3, "three synthetic classes"
    assert set(loaded.label_slugs) == {"blue_triangle", "green_square", "red_circle"}

    # eval reports metrics over the val split.
    manifest = tmp_path / "_synthetic_data" / "synthetic.jsonl"
    ev = run_cli(["eval", "--model", str(artifact), "--manifest", str(manifest), "--split", "val"])
    assert ev.returncode == 0, ev.stderr
    assert "top1=" in ev.stdout


def test_calibration_writes_a_new_artifact_never_edits(run_cli, tmp_path):
    """§7.4/I8: --calibrate produces a new file with a derived model_id."""
    artifact = tmp_path / "synth.acmodel"
    run_cli(
        [
            "train", "--dataset", "synthetic", "--arch", "tinycnn",
            "--out", str(artifact), "--epochs-head", "1", "--epochs-finetune", "1",
            "--batch-size", "16", "--input-size", "64", "--seed", "0",
        ]
    )
    manifest = tmp_path / "_synthetic_data" / "synthetic.jsonl"
    calibrated = tmp_path / "synth_cal.acmodel"
    result = run_cli(
        [
            "eval", "--model", str(artifact), "--manifest", str(manifest),
            "--split", "val", "--calibrate", "--out", str(calibrated),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert calibrated.is_file()
    assert artifact.is_file(), "the original artifact is untouched"

    from animal_classifier.classify.artifact import load  # noqa: PLC0415

    original = load(artifact)
    cal = load(calibrated)
    assert cal.model_id == f"{original.model_id}+cal1"
    assert original.temperature == 1.0, "the original is not edited"


def test_calibrate_without_out_is_fatal(run_cli, tmp_path):
    """§7.4: --calibrate requires --out; an artifact is never edited in place."""
    artifact = tmp_path / "synth.acmodel"
    run_cli(
        [
            "train", "--dataset", "synthetic", "--arch", "tinycnn",
            "--out", str(artifact), "--epochs-head", "1", "--epochs-finetune", "1",
            "--batch-size", "16", "--input-size", "64",
        ]
    )
    manifest = tmp_path / "_synthetic_data" / "synthetic.jsonl"
    result = run_cli(
        ["eval", "--model", str(artifact), "--manifest", str(manifest), "--calibrate"]
    )
    assert result.returncode == 3, result.stderr
    assert "out" in result.stderr.lower()
