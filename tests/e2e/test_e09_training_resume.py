"""E9: training checkpoint and ``--resume`` (§7.3).

A checkpoint is written after every epoch, and ``--resume`` restores it. The two
behaviours that matter:

- **resuming continues rather than restarting** — the checkpoint records the epoch,
  so a run interrupted after epoch 1 finishes the remaining epochs;
- **a mismatched resume is fatal, exit 3, naming both ids** — resuming across a
  different manifest or a different architecture would silently blend two training
  runs into one artifact whose ``manifest_sha256`` is a lie.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def _synthetic_manifest(run_cli, tmp_path, name="m.acmodel"):
    """Train one epoch to produce both an artifact and its .ckpt."""
    artifact = tmp_path / name
    result = run_cli(
        ["train", "--dataset", "synthetic", "--arch", "tinycnn", "--out", str(artifact),
         "--input-size", "64", "--epochs-head", "1", "--epochs-finetune", "0",
         "--batch-size", "16", "--seed", "0"]
    )
    assert result.returncode == 0, result.stderr
    return artifact, tmp_path / "_synthetic_data" / "synthetic.jsonl"


def test_checkpoint_is_written_and_carries_provenance(run_cli, tmp_path):
    artifact, _manifest = _synthetic_manifest(run_cli, tmp_path)
    checkpoint = artifact.with_suffix(artifact.suffix + ".ckpt")
    assert checkpoint.is_file(), "a checkpoint is written after every epoch (§7.3)"

    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert ckpt["arch"] == "tinycnn"
    assert ckpt["manifest_sha256"], "the checkpoint records which manifest it trained on"
    assert ckpt["epoch"] == 1
    assert "optimizer" in ckpt and "model" in ckpt


def test_resume_continues_the_schedule(run_cli, tmp_path):
    """Resuming a 1-epoch run with a 3-epoch schedule finishes the remaining two."""
    artifact, manifest = _synthetic_manifest(run_cli, tmp_path)

    resumed = run_cli(
        ["train", "--manifest", str(manifest), "--arch", "tinycnn", "--out", str(artifact),
         "--input-size", "64", "--epochs-head", "2", "--epochs-finetune", "1",
         "--batch-size", "16", "--seed", "0", "--resume"]
    )
    assert resumed.returncode == 0, resumed.stderr
    # It resumed at epoch 1 rather than starting over: epochs 2 and 3 are the ones run.
    assert "epoch 2/3" in resumed.stderr, resumed.stderr
    assert "epoch 1/3" not in resumed.stderr, "epoch 1 must not be re-run"

    import torch

    ckpt = torch.load(artifact.with_suffix(artifact.suffix + ".ckpt"), map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 3, "the schedule completed"


def test_resume_across_a_different_manifest_is_fatal(run_cli, tmp_path):
    """§7.3: resuming on different data would make manifest_sha256 a lie."""
    artifact, manifest = _synthetic_manifest(run_cli, tmp_path)

    # A different manifest: same images, one line dropped -> different sha256.
    lines = manifest.read_text().splitlines()
    other = tmp_path / "other.jsonl"
    other.write_text("\n".join(lines[:-3]) + "\n")

    result = run_cli(
        ["train", "--manifest", str(other), "--arch", "tinycnn", "--out", str(artifact),
         "--input-size", "64", "--epochs-head", "2", "--epochs-finetune", "0",
         "--batch-size", "16", "--resume"]
    )
    assert result.returncode == 3, result.stderr
    assert "manifest" in result.stderr.lower()


def test_resume_across_a_different_arch_is_fatal(run_cli, tmp_path):
    artifact, manifest = _synthetic_manifest(run_cli, tmp_path)
    result = run_cli(
        ["train", "--manifest", str(manifest), "--arch", "efficientnet_b0",
         "--out", str(artifact), "--input-size", "64", "--epochs-head", "1",
         "--epochs-finetune", "0", "--batch-size", "16", "--resume"]
    )
    assert result.returncode == 3, result.stderr
    assert "arch" in result.stderr.lower()
