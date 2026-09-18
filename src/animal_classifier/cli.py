"""Command line interface for animal-classifier (DESIGN.md §8).

Only ``classify`` is wired to a real pipeline; the other five commands are
placeholders that say so and exit 1, because a command that silently does nothing
is worse than one that refuses.

Two §8 details are hand-rolled on purpose, since click does not do either for you:

- **``--link`` / ``--hardlink`` mutual exclusion.** Both are booleans, and the
  callback raises :class:`typer.BadParameter` when both are given, which typer maps
  to **exit 2** — §10.1's usage class.
- **``--formats`` is repeatable and replaces, never extends.** ``--formats jpeg
  --formats png`` means exactly those two families (§3.1), so a user narrowing the
  card's formats is not silently given the configured list as well.

**Flags are only forwarded when actually passed.** Every option defaults to
``None``, and :meth:`Config.resolve` drops ``None`` values, so an unspecified flag
cannot mask a TOML file or an environment variable. Passing ``Mode.COPY`` whenever
``--link`` was absent would quietly beat a ``mode = "link"`` config file — which is
the bug this shape avoids.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import typer

from .config import (
    BirdProvider,
    Command,
    Config,
    DetectorKind,
    Device,
    Format,
    Mode,
)
from .errors import EXIT_UNEXPECTED, AnimalClassifierError

log = logging.getLogger("animal_classifier")

app = typer.Typer(
    add_completion=False,
    help="Detect, classify and file safari photographs by animal species.",
)

_NOT_IMPLEMENTED = "not implemented yet"

#: §5.4 requires this to be user-facing in both `classify --help` and README.md,
#: because it will surprise anyone whose card also holds family photos.
_PERSON_VEHICLE_NOTE = (
    "Images containing only people or vehicles are filed as `landscape` (or "
    "`junk` if blurry); their person/vehicle boxes are still kept in the catalog "
    "and drawn in the GUI."
)

#: Set by the top-level callback so each command can read the global options.
_GLOBAL: dict[str, Any] = {"config_path": None}


@app.callback()
def main(
    config: Path = typer.Option(
        None, "--config", help="TOML config file. Must exist if given."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Log at DEBUG."),
    quiet: bool = typer.Option(False, "--quiet", help="Log at WARNING and above."),
) -> None:
    """Global options shared by every command (§8)."""
    if verbose and quiet:
        raise typer.BadParameter("--verbose and --quiet are mutually exclusive")
    _configure_logging(verbose=verbose, quiet=quiet)
    _GLOBAL["config_path"] = config


def _configure_logging(*, verbose: bool, quiet: bool) -> None:
    """Diagnostics go to stderr only (§10.1); stdout is for results.

    ``force=True`` because a second invocation inside one process (the test suite,
    or ``gui`` importing this module) must not silently keep the first call's level.
    """
    level = logging.INFO
    if verbose:
        level = logging.DEBUG
    elif quiet:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(levelname)s %(message)s",
        force=True,
    )


@app.command(
    help=(
        "Classify every image under SOURCE and file it under OUTPUT/<label>/.\n\n"
        f"{_PERSON_VEHICLE_NOTE}\n\n"
        "Labels are <species>, multiple, landscape, junk or unknown. The only "
        "size comparison in the pipeline is --dominance-ratio, which is relative: "
        "a lone animal is filed as its species however small it is in frame."
    )
)
def classify(
    source: Path = typer.Argument(..., help="SD-card directory to walk recursively."),
    output: Path = typer.Option(
        None, "--output", "-o", help="Destination root [default: ~/animal_pics]."
    ),
    link: bool = typer.Option(False, "--link", help="Symlink instead of copying."),
    hardlink: bool = typer.Option(
        False, "--hardlink", help="Hardlink instead of copying."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Plan only; write nothing to the output tree."
    ),
    reclassify: bool = typer.Option(
        False, "--reclassify", help="Re-run inference on images already done."
    ),
    ignore_overrides: bool = typer.Option(
        False, "--ignore-overrides", help="Ignore human label corrections."
    ),
    raw: bool = typer.Option(
        False, "--raw", help="Enable RAW decode (needs the `raw` extra)."
    ),
    formats: list[Format] = typer.Option(
        None,
        "--formats",
        help="Eligible families; repeatable. Replaces the configured list.",
    ),
    max_file_bytes: int = typer.Option(
        None, "--max-file-bytes", help="Scan-time file-size cap [default: 512 MiB]."
    ),
    follow_source_symlinks: bool = typer.Option(
        False,
        "--follow-source-symlinks",
        help="Ingest symlinked files inside the card (default: skip them).",
    ),
    dominance_ratio: float = typer.Option(
        None,
        "--dominance-ratio",
        help="Largest animal wins if its box area is this many times the next [1.6].",
    ),
    min_confidence: float = typer.Option(
        None,
        "--min-confidence",
        help="Below this species confidence an image is `unknown` [0.45].",
    ),
    blur_threshold: float = typer.Option(
        None,
        "--blur-threshold",
        help="Animal-free images below this sharpness are `junk` [100.0].",
    ),
    detector: DetectorKind = typer.Option(
        None, "--detector", help="Detection backend."
    ),
    detector_weights: Path = typer.Option(
        None, "--detector-weights", help="MegaDetector checkpoint path."
    ),
    detector_confidence: float = typer.Option(
        None, "--detector-confidence", help="Detection confidence threshold [0.20]."
    ),
    species_model: Path = typer.Option(
        None, "--species-model", help="Species `.acmodel` artifact."
    ),
    bird_model: Path = typer.Option(
        None, "--bird-model", help="Bird `.acmodel` artifact."
    ),
    bird_provider: BirdProvider = typer.Option(
        None, "--bird-provider", help="Bird refinement provider."
    ),
    force_bird_head: bool = typer.Option(
        False, "--force-bird-head", help="Always use the own bird head."
    ),
    jobs: int = typer.Option(None, "--jobs", help="Worker threads for decode/hash."),
    limit: int = typer.Option(
        None, "--limit", help="Stop after this many images are submitted to inference."
    ),
    device: Device = typer.Option(None, "--device", help="Compute device."),
) -> None:
    """Walk SOURCE, classify every image, and file it under OUTPUT/<label>/."""
    if link and hardlink:
        # §8: hand-rolled because click will not do it; typer maps this to exit 2.
        raise typer.BadParameter(
            "--link and --hardlink are mutually exclusive; omit both to copy"
        )
    mode = Mode.LINK if link else Mode.HARDLINK if hardlink else None

    overrides: dict[str, Any] = {
        "source_root": str(source),
        "output_root": str(output) if output is not None else None,
        "mode": mode,
        "dry_run": dry_run or None,
        "reclassify": reclassify or None,
        "ignore_overrides": ignore_overrides or None,
        "raw": raw or None,
        "formats": [str(item) for item in formats] if formats else None,
        "max_file_bytes": max_file_bytes,
        "follow_source_symlinks": follow_source_symlinks or None,
        "dominance_ratio": dominance_ratio,
        "min_species_confidence": min_confidence,
        "blur_threshold": blur_threshold,
        "detector": detector,
        "detector_weights": str(detector_weights) if detector_weights else None,
        "detector_confidence": detector_confidence,
        "species_model": str(species_model) if species_model else None,
        "bird_model": str(bird_model) if bird_model else None,
        "bird_provider": bird_provider,
        "force_bird_head": force_bird_head or None,
        "jobs": jobs,
        "limit": limit,
        "device": device,
    }

    code = _classify(overrides)
    raise typer.Exit(code)


def _classify(overrides: dict[str, Any]) -> int:
    """Resolve config, run the pipeline, and map any failure to its exit code.

    Every deliberate failure carries its own ``exit_code`` (§10.1), so this
    translates rather than decides. An unexpected exception is exit 1 with a
    traceback only at ``-v``, because a stack trace is a diagnostic and the default
    output should stay readable.
    """
    from .pipeline import classify_run  # noqa: PLC0415 - imports torch-adjacent deps

    try:
        config = Config.resolve(
            command=Command.CLASSIFY,
            cli=overrides,
            config_path=_GLOBAL["config_path"],
        )
        summary = classify_run(config, argv=sys.argv[1:])
    except AnimalClassifierError as error:
        log.error("%s", error)
        return error.exit_code
    except KeyboardInterrupt:
        log.error("interrupted; the run is resumable — re-run the same command")
        return EXIT_UNEXPECTED
    except Exception as error:
        log.error(
            "unexpected failure: %s: %s",
            type(error).__name__,
            error,
            exc_info=log.isEnabledFor(logging.DEBUG),
        )
        return EXIT_UNEXPECTED

    _report(summary)
    return summary.exit_code


def _report(summary: Any) -> None:
    """Log the run's outcome. Diagnostics to stderr (§10.1), never ``print``."""
    log.info(
        "run %s %s: %d image(s), %d filed, %d already done, %d skipped, %d failed",
        summary.run_id,
        "planned" if summary.dry_run else "complete",
        summary.n_total,
        summary.n_done,
        summary.n_already_done,
        summary.n_skipped,
        summary.n_failed,
    )
    if summary.outcomes:
        log.info(
            "filing: %s",
            ", ".join(f"{name}={count}" for name, count in sorted(summary.outcomes.items())),
        )
    if summary.label_counts:
        log.info(
            "labels: %s",
            ", ".join(
                f"{label}={count}"
                for label, count in sorted(summary.label_counts.items())
            ),
        )


@app.command()
def gui(
    output: Path = typer.Option(
        Path("~/animal_pics"), "--output", "-o", help="Labelled output root to browse."
    ),
    port: int = typer.Option(8765, "--port", help="Port to serve the GUI on."),
) -> None:
    """Serve the review GUI for an already-classified output tree."""
    typer.echo(f"gui: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


@app.command()
def train(
    manifest: Path = typer.Option(None, "--manifest", help="Training dataset manifest (JSONL)."),
    out: Path = typer.Option(..., "--out", help="Where to write the .acmodel artifact."),
    arch: str = typer.Option("efficientnet_b0", "--arch", help="Backbone architecture (or tinycnn)."),
    dataset: str = typer.Option("manifest", "--dataset", help="manifest | synthetic."),
    classes: str = typer.Option(None, "--classes", help="Comma-separated subset of labels."),
    epochs_head: int = typer.Option(3, "--epochs-head", help="Head-only epochs."),
    epochs_finetune: int = typer.Option(5, "--epochs-finetune", help="Finetune epochs."),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    input_size: int = typer.Option(224, "--input-size", help="Input resolution."),
    unfreeze_blocks: int = typer.Option(2, "--unfreeze-blocks", help="Top blocks to unfreeze."),
    backbone_weights: Path = typer.Option(None, "--backbone-weights", help="Local backbone weights."),
    resume: bool = typer.Option(False, "--resume", help="Resume from <out>.ckpt."),
    jobs: int = typer.Option(7, "--jobs", help="torch thread count."),
    seed: int = typer.Option(0, "--seed", help="RNG seed."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | cuda."),
) -> None:
    """Train or finetune the species classifier."""
    code = _train(
        manifest=manifest, out=out, arch=arch, dataset=dataset, input_size=input_size,
        epochs_head=epochs_head, epochs_finetune=epochs_finetune, batch_size=batch_size,
        unfreeze_blocks=unfreeze_blocks, backbone_weights=backbone_weights, resume=resume,
        jobs=jobs, seed=seed, device=device,
    )
    raise typer.Exit(code)


def _train(**kwargs: Any) -> int:
    from .training.run import run_train  # noqa: PLC0415

    try:
        run_train(**kwargs)
    except AnimalClassifierError as error:
        log.error("%s", error)
        return error.exit_code
    except Exception as error:
        log.error("train failed: %s: %s", type(error).__name__, error,
                  exc_info=log.isEnabledFor(logging.DEBUG))
        return EXIT_UNEXPECTED
    return 0


@app.command()
def eval(
    manifest: Path = typer.Option(..., "--manifest", help="Evaluation dataset manifest."),
    model: Path = typer.Option(..., "--model", help="Exported model artifact to evaluate."),
    split: str = typer.Option("val", "--split", help="train | val."),
    calibrate: bool = typer.Option(False, "--calibrate", help="Fit a temperature."),
    out: Path = typer.Option(None, "--out", help="New artifact for --calibrate."),
    device: str = typer.Option("auto", "--device", help="auto | cpu | cuda."),
) -> None:
    """Evaluate an exported model artifact (and optionally calibrate it)."""
    code = _eval(manifest=manifest, model=model, split=split, calibrate=calibrate,
                 out=out, device=device)
    raise typer.Exit(code)


def _eval(*, manifest: Path, model: Path, split: str, calibrate: bool,
          out: Path | None, device: str) -> int:
    from .classify.artifact import load as load_artifact  # noqa: PLC0415
    from .errors import ConfigError  # noqa: PLC0415
    from .training import evaluate as ev  # noqa: PLC0415
    from .training.manifest import load_manifest  # noqa: PLC0415
    from .training.run import resolve_device  # noqa: PLC0415

    try:
        if calibrate and out is None:
            raise ConfigError("eval --calibrate requires --out; artifacts are immutable")
        if out is not None and out.exists():
            raise ConfigError(
                f"--out {out} already exists; calibration writes a NEW artifact, "
                "never edits one (its model_id would then be ambiguous)"
            )
        resolved = resolve_device(device)
        artifact = load_artifact(model)
        samples = load_manifest(manifest, split=split)
        metrics = ev.evaluate(artifact, samples, device=resolved)
        log.info(
            "eval %s on %d %s samples: top1=%.3f top5=%.3f macro_recall=%.3f",
            artifact.model_id, metrics.support, split, metrics.top1, metrics.top5,
            metrics.macro_recall,
        )
        typer.echo(
            f"top1={metrics.top1:.4f} top5={metrics.top5:.4f} "
            f"macro_recall={metrics.macro_recall:.4f} support={metrics.support}"
        )
        if calibrate:
            _calibrate_to_new_artifact(artifact, samples, model, out, resolved, ev)
    except AnimalClassifierError as error:
        log.error("%s", error)
        return error.exit_code
    except Exception as error:
        log.error("eval failed: %s: %s", type(error).__name__, error,
                  exc_info=log.isEnabledFor(logging.DEBUG))
        return EXIT_UNEXPECTED
    return 0


def _calibrate_to_new_artifact(artifact, samples, model_path, out, device, ev) -> None:
    """Fit a temperature and write a NEW artifact with a derived model_id (§7.4)."""
    import torch  # noqa: PLC0415

    from .classify.artifact import save as save_artifact  # noqa: PLC0415

    temperature = ev.fit_temperature(artifact, samples, device=device)
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    existing_cal = artifact.model_id.count("+cal")
    new_id = f"{artifact.model_id}+cal{existing_cal + 1}"
    train_meta = dict(blob.get("train", {}))
    train_meta["calibrated_from"] = artifact.model_id
    save_artifact(
        out,
        model_id=new_id,
        arch=artifact.arch,
        input_size=artifact.input_size,
        mean=list(artifact.mean),
        std=list(artifact.std),
        labels=blob["labels"],
        state_dict=blob["state_dict"],
        temperature=temperature,
        train_meta=train_meta,
    )
    log.info("calibrated: T=%.4f -> %s (%s)", temperature, out, new_id)
    typer.echo(f"calibrated temperature={temperature:.4f} -> {out}")


@app.command("export-trainset")
def export_trainset(
    output: Path = typer.Option(
        Path("~/animal_pics"), "--output", "-o", help="Labelled output root to export from."
    ),
    destination: Path = typer.Option(..., "--destination", help="Where to write the manifest."),
) -> None:
    """Export labelled crops from a classified tree as a training manifest."""
    typer.echo(f"export-trainset: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


@app.command()
def verify() -> None:
    """Report which models, providers and network resources are reachable."""
    typer.echo(f"verify: {_NOT_IMPLEMENTED}", err=True)
    raise typer.Exit(1)


if __name__ == "__main__":
    app()
