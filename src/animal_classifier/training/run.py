"""CLI-facing orchestration for ``train`` and ``eval`` (DESIGN.md §7).

Thin glue between the CLI and the training modules: resolve the device, build the
dataset (real manifest or the synthetic smoke set), run the two-stage trainer, and
write the ``.acmodel`` artifact. Kept out of ``cli.py`` so the heavy torch imports
happen only when these commands actually run.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..classify import artifact as artifact_mod
from ..errors import ConfigError
from ..taxonomy import merged
from . import trainer as trainer_mod
from .dataset import IMAGENET_MEAN, IMAGENET_STD
from .manifest import Sample, load_manifest

log = logging.getLogger(__name__)


def resolve_device(requested: str) -> str:
    """`auto`→cpu/cuda; explicit `cuda` on a CPU host is fatal (§7.3, I7)."""
    import torch  # noqa: PLC0415

    if requested == "cuda" and not torch.cuda.is_available():
        raise ConfigError(
            "--device cuda requested but torch reports no available CUDA device; "
            "omit the flag or pass --device cpu"
        )
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _label_metadata(slugs: list[str]) -> list[dict[str, Any]]:
    """Attach scientific name / class / rank from the taxonomy where known (§7.1).

    A slug with no taxonomy row (the synthetic classes) still gets a legal entry:
    its own name as ``common``, ``scientific=None``, ``rank='species'`` — honest
    absence of the optional fields, not a fabricated binomial.
    """
    table = merged()  # Mapping[str, Taxon]
    out: list[dict[str, Any]] = []
    for slug in slugs:
        taxon = table.get(slug)
        if taxon is not None:
            out.append(
                {
                    "key": slug,
                    "common": taxon.common,
                    "scientific": taxon.scientific,
                    "class": taxon.taxon_class,
                    "rank": str(taxon.rank),
                }
            )
        else:
            out.append(
                {
                    "key": slug,
                    "common": slug.replace("_", " ").title(),
                    "scientific": None,
                    "class": None,
                    "rank": "species",
                }
            )
    return out


def run_train(
    *,
    manifest: Path | None,
    out: Path,
    arch: str,
    dataset: str,
    input_size: int,
    epochs_head: int,
    epochs_finetune: int,
    batch_size: int,
    unfreeze_blocks: int,
    backbone_weights: Path | None,
    resume: bool,
    jobs: int,
    seed: int,
    device: str,
) -> trainer_mod.TrainResult:
    """Build the dataset, train, and write the artifact. Returns the metrics."""
    import torch  # noqa: PLC0415

    resolved_device = resolve_device(device)
    torch.set_num_threads(max(1, jobs))

    if dataset == "synthetic":
        from .synthetic import generate  # noqa: PLC0415

        synth_root = out.parent / "_synthetic_data"
        manifest = generate(synth_root, seed=seed)
        log.info("generated synthetic dataset at %s", manifest)
    elif manifest is None:
        raise ConfigError("train needs --manifest (or --dataset synthetic)")

    samples: list[Sample] = load_manifest(manifest)
    manifest_id = trainer_mod.manifest_sha256(manifest)

    config = trainer_mod.TrainConfig(
        arch=arch,
        input_size=input_size,
        epochs_head=epochs_head,
        epochs_finetune=epochs_finetune,
        batch_size=batch_size,
        unfreeze_blocks=unfreeze_blocks,
        jobs=jobs,
        seed=seed,
        device=resolved_device,
    )
    model, result = trainer_mod.train(
        samples,
        config=config,
        out=out,
        backbone_weights=backbone_weights,
        resume=resume,
        manifest_id=manifest_id,
    )

    created = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_id = f"species-{arch}-{created}"
    artifact_mod.save(
        out,
        model_id=model_id,
        arch=arch,
        input_size=input_size,
        mean=list(IMAGENET_MEAN),
        std=list(IMAGENET_STD),
        labels=_label_metadata(result.classes),
        state_dict={k: v.cpu() for k, v in model.state_dict().items()},
        temperature=1.0,
        train_meta={
            "dataset": dataset,
            "manifest_sha256": manifest_id,
            "epochs": epochs_head + epochs_finetune,
            "val_top1": result.best_val_top1,
            "val_top5": result.best_val_top5,
            "backbone_weights": str(backbone_weights) if backbone_weights else None,
            "calibrated_from": None,
            "created_at": created,
        },
    )
    log.info(
        "trained %s: val top1=%.3f top5=%.3f -> %s",
        model_id,
        result.best_val_top1,
        result.best_val_top5,
        out,
    )
    return result
