"""E22: a missing or invalid required value fails loudly, never silently (§10.1, I7).

The invariant this whole project is built around: a required value that is missing,
unknown or out of range is a fatal error naming the offending value — it is never
replaced by a default and the run never limps on. Each case asserts:

- exit **3** (the config/asset class), or **2** for a typer usage error that lists
  the valid set (``--formats raw``, which typer rejects at parse time);
- the offending name appears in stderr, so the message is actionable;
- **no destination directory is created** — a failed config must not leave a
  half-built output tree behind.

Two cases are conditional on build progress and say so, rather than asserting the
wrong thing: ``--formats raw`` is exit 2 (enum validation) rather than 3 here, and
the missing-``species_model`` case is skipped until ``config.SPECIES_INFERENCE_WIRED``
flips (F27) — asserting exit 3 for it now would bake in behaviour the tool does not
yet have.
"""

from __future__ import annotations

import pytest

from animal_classifier.config import SPECIES_INFERENCE_WIRED

LABEL_DIRS = {"landscape", "junk", "unknown", "multiple"}


def _no_label_dirs(output) -> bool:
    if not output.exists():
        return True
    return not any(p.is_dir() and p.name in LABEL_DIRS for p in output.iterdir())


def test_missing_detector_weights_are_fatal_only_under_no_download(run_cli, tmp_path):
    """Absent weights are a *download*, not a config error — unless you opted out.

    This row of §10.1 moved deliberately (see E27): refusing to run because a
    hash-pinned artifact has not been fetched yet made a fresh clone unable to
    classify anything. With ``--no-download`` the original fail-loud contract stands,
    and that is what is asserted here.
    """
    card = tmp_path / "card"
    card.mkdir()
    output = tmp_path / "out"
    absent_weights = tmp_path / "absent_md.pt"  # point away from any cached checkpoint
    result = run_cli(
        ["classify", str(card), "-o", str(output), "--detector", "megadetector",
         "--detector-weights", str(absent_weights), "--no-download"]
    )
    assert result.returncode == 3, result.stderr
    assert "md_v5a.0.0.pt" in result.stderr or "MegaDetector" in result.stderr
    assert _no_label_dirs(output)


def test_dominance_ratio_below_one_is_fatal(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    output = tmp_path / "out"
    result = run_cli(
        ["classify", str(card), "-o", str(output), "--detector", "scripted",
         "--dominance-ratio", "0.5"]
    )
    assert result.returncode == 3, result.stderr
    assert "dominance_ratio" in result.stderr or "dominance-ratio" in result.stderr
    assert "0.5" in result.stderr
    assert _no_label_dirs(output)


def test_output_nested_in_source_is_fatal(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    nested = card / "pics"
    result = run_cli(
        ["classify", str(card), "-o", str(nested), "--detector", "scripted"]
    )
    assert result.returncode == 3, result.stderr
    assert "inside" in result.stderr.lower() or "nested" in result.stderr.lower()
    assert _no_label_dirs(nested)


def test_device_cuda_on_a_cpu_host_is_fatal(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    output = tmp_path / "out"
    result = run_cli(
        ["classify", str(card), "-o", str(output), "--detector", "scripted",
         "--device", "cuda"]
    )
    assert result.returncode == 3, result.stderr
    assert "cuda" in result.stderr.lower()
    assert _no_label_dirs(output)


def test_hosted_bird_api_selected_is_fatal(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    output = tmp_path / "out"
    result = run_cli(
        ["classify", str(card), "-o", str(output), "--detector", "scripted",
         "--bird-provider", "hosted_bird_api"]
    )
    assert result.returncode == 3, result.stderr
    assert "hosted_bird_api" in result.stderr
    assert _no_label_dirs(output)


def test_ebird_enrich_without_key_is_fatal(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    output = tmp_path / "out"
    result = run_cli(
        ["classify", str(card), "-o", str(output), "--detector", "scripted",
         "--bird-provider", "ebird_enrich"]
    )
    assert result.returncode == 3, result.stderr
    assert "EBIRD_API_KEY" in result.stderr
    assert _no_label_dirs(output)


def test_unknown_toml_key_is_fatal(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    output = tmp_path / "out"
    config = tmp_path / "config.toml"
    config.write_text('output_root = "~/animal_pics"\nnonexistent_knob = 3\n')
    result = run_cli(
        ["--config", str(config), "classify", str(card), "-o", str(output),
         "--detector", "scripted"]
    )
    assert result.returncode == 3, result.stderr
    assert "nonexistent_knob" in result.stderr
    assert _no_label_dirs(output)


def test_missing_config_file_is_fatal(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    result = run_cli(
        ["--config", str(tmp_path / "nope.toml"), "classify", str(card),
         "-o", str(tmp_path / "out"), "--detector", "scripted"]
    )
    assert result.returncode == 3, result.stderr


def test_formats_raw_is_rejected(run_cli, tmp_path):
    """``raw`` is not a family (§3.1). typer rejects the enum value at parse time."""
    card = tmp_path / "card"
    card.mkdir()
    output = tmp_path / "out"
    result = run_cli(
        ["classify", str(card), "-o", str(output), "--detector", "scripted",
         "--formats", "raw"]
    )
    assert result.returncode == 2, "an invalid --formats value is a typer usage error"
    assert _no_label_dirs(output)


@pytest.mark.skipif(
    not SPECIES_INFERENCE_WIRED,
    reason="species_model is not required until classify/own_model.py lands (F27)",
)
def test_missing_species_model_names_the_train_command(run_cli, tmp_path):
    """A real classification run (--detector megadetector) requires the artifacts.

    The species model is required only for the real detector — ``--detector
    scripted`` is a testing affordance that runs pass-through with no model. So this
    uses ``megadetector``, and asserts the *species_model* message specifically by
    pointing the weights at a real (or plausibly present) path is unnecessary: the
    detector-weights check fires first if they are absent, so we assert exit 3 and
    that the message names a producing command either way.
    """
    card = tmp_path / "card"
    card.mkdir()
    output = tmp_path / "out"
    # An *explicitly named* species model that does not exist is fatal (the user
    # asked for a specific model). scripted detector, so the detector-weights check
    # does not fire first and we isolate the species_model message.
    result = run_cli(
        ["classify", str(card), "-o", str(output), "--detector", "scripted",
         "--species-model", str(tmp_path / "absent.acmodel")]
    )
    assert result.returncode == 3, result.stderr
    assert "animal-classifier train" in result.stderr
    assert "absent.acmodel" in result.stderr
    assert _no_label_dirs(output)


def test_absent_default_species_model_is_pass_through_not_fatal(run_cli, tmp_path):
    """A run pointing at no model is the pass-through path, not an error (F27)."""
    import json

    from PIL import Image

    card = tmp_path / "card" / "DCIM"
    card.mkdir(parents=True)
    Image.new("RGB", (400, 300), "green").save(card / "a.jpg")
    (card / "a.jpg.boxes.json").write_text(
        json.dumps([{"cls": "animal", "conf": 0.9, "x0": 20, "y0": 20, "x1": 380, "y1": 280}])
    )
    output = tmp_path / "out"
    result = run_cli(["classify", str(card.parent), "-o", str(output), "--detector", "scripted"])
    assert result.returncode in {0, 4}, result.stderr
    assert (output / "unknown" / "a.jpg").is_file(), "animal filed as unknown (no model)"
