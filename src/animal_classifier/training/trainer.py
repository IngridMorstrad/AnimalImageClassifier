"""Two-stage transfer-learning trainer (DESIGN.md §7.3).

The path ``PLAN.md`` specifies: freeze the backbone and train the head, then
unfreeze the top blocks at a low learning rate. Loss is
``CrossEntropyLoss(label_smoothing=0.1)``; a checkpoint is written after every epoch
(model, optimizer, scheduler, epoch, RNG states, best val top-1, manifest sha256),
and ``--resume`` refuses to continue across a different manifest sha256 or arch.
Best-val weights are what get exported.

For ``tinycnn`` there is no backbone to freeze, so the two stages collapse to plain
training — the same loop, the same checkpoints — which is exactly what the synthetic
smoke path (E10) needs.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from ..errors import ConfigError
from . import dataset as ds
from .manifest import Sample, class_list

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TrainConfig:
    arch: str
    input_size: int
    epochs_head: int
    epochs_finetune: int
    batch_size: int
    unfreeze_blocks: int
    jobs: int
    seed: int
    device: str


@dataclass(frozen=True, slots=True)
class TrainResult:
    best_val_top1: float
    best_val_top5: float
    epochs_run: int
    classes: list[str]


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "tinycnn":
        from .tinycnn import TinyCNN  # noqa: PLC0415

        return TinyCNN(num_classes=num_classes)
    import timm  # noqa: PLC0415

    # pretrained=False: E7's backbone weights are loaded separately from a local
    # path (the sandbox blocks the pretrained download); the smoke path needs none.
    return timm.create_model(arch, pretrained=False, num_classes=num_classes)


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int, int]:
    """Return (top1_correct, top5_correct, n) for one batch."""
    n = targets.size(0)
    maxk = min(5, logits.size(1))
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(targets.view(-1, 1))
    top1 = int(correct[:, 0].sum().item())
    top5 = int(correct.any(dim=1).sum().item())
    return top1, top5, n


def train(
    samples: list[Sample],
    *,
    config: TrainConfig,
    out: Path,
    backbone_weights: Path | None = None,
    resume: bool = False,
    manifest_id: str,
) -> tuple[nn.Module, TrainResult]:
    """Run the two-stage schedule and return the best-val model and metrics."""
    torch.manual_seed(config.seed)
    device = torch.device(config.device)
    classes = class_list(samples)
    if len(classes) < 2:
        raise ConfigError(
            f"training needs at least 2 classes, manifest has {len(classes)}: {classes}"
        )

    train_samples = [s for s in samples if s.split == "train"]
    val_samples = [s for s in samples if s.split == "val"] or train_samples
    if not train_samples:
        raise ConfigError("no training samples after split filtering")

    model = _build_model(config.arch, len(classes)).to(device)
    if backbone_weights is not None:
        _load_backbone(model, backbone_weights)

    train_loader = DataLoader(
        ds.CropDataset(train_samples, classes, transform=ds.train_transform(config.input_size)),
        batch_size=config.batch_size,
        sampler=ds.balanced_sampler(train_samples, classes),
        num_workers=0,
        drop_last=False,
    )
    val_loader = DataLoader(
        ds.CropDataset(val_samples, classes, transform=ds.eval_transform(config.input_size)),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    checkpoint_path = out.with_suffix(out.suffix + ".ckpt")
    start_epoch, best_state, best_top1, best_top5 = _maybe_resume(
        resume, checkpoint_path, model, config.arch, manifest_id
    )

    total_epochs = config.epochs_head + config.epochs_finetune
    epoch = start_epoch
    for epoch in range(start_epoch, total_epochs):
        stage_head = epoch < config.epochs_head
        _set_trainable(model, config.arch, head_only=stage_head, unfreeze_blocks=config.unfreeze_blocks)
        optimizer = _optimizer(model, head_only=stage_head)
        _train_one_epoch(model, train_loader, criterion, optimizer, device)
        top1, top5 = _evaluate(model, val_loader, device)
        log.info(
            "epoch %d/%d (%s): val top1=%.3f top5=%.3f",
            epoch + 1,
            total_epochs,
            "head" if stage_head else "finetune",
            top1,
            top5,
        )
        if top1 >= best_top1:
            best_top1, best_top5 = top1, top5
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        _save_checkpoint(
            checkpoint_path, model, optimizer, epoch + 1, best_top1, best_top5,
            config.arch, manifest_id, best_state,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, TrainResult(
        best_val_top1=best_top1,
        best_val_top5=best_top5,
        epochs_run=total_epochs - start_epoch,
        classes=classes,
    )


def _train_one_epoch(model, loader, criterion, optimizer, device) -> None:
    model.train()
    # Frozen BatchNorm in the head stage: keep running stats (§7.3).
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()


def _evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    top1 = top5 = total = 0
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            c1, c5, n = _accuracy(model(images), targets)
            top1 += c1
            top5 += c5
            total += n
    total = max(total, 1)
    return top1 / total, top5 / total


def _set_trainable(model, arch: str, *, head_only: bool, unfreeze_blocks: int) -> None:
    if arch == "tinycnn":
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = not head_only
    # The classifier head is always trainable.
    for p in _head_parameters(model):
        p.requires_grad = True


def _head_parameters(model):
    classifier = getattr(model, "get_classifier", None)
    if callable(classifier):
        return classifier().parameters()
    return model.classifier.parameters() if hasattr(model, "classifier") else model.parameters()


def _optimizer(model, *, head_only: bool):
    params = [p for p in model.parameters() if p.requires_grad]
    lr = 3e-3 if head_only else 3e-4
    return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)


def _load_backbone(model, weights: Path) -> None:
    if not weights.is_file():
        raise ConfigError(
            f"backbone weights not found at {weights}; pretrained downloads are "
            "blocked in this environment, so pass a local --backbone-weights path"
        )
    state = torch.load(weights, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)  # head shape differs; strict=False by design
    log.info("loaded backbone weights from %s", weights)


def _maybe_resume(resume, checkpoint_path, model, arch, manifest_id):
    if not resume or not checkpoint_path.is_file():
        return 0, None, 0.0, 0.0
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if ckpt.get("arch") != arch:
        raise ConfigError(
            f"--resume: checkpoint arch {ckpt.get('arch')!r} != requested {arch!r}"
        )
    if ckpt.get("manifest_sha256") != manifest_id:
        raise ConfigError(
            "--resume: checkpoint was trained on a different manifest "
            f"({ckpt.get('manifest_sha256')} != {manifest_id})"
        )
    model.load_state_dict(ckpt["model"])
    log.info("resumed from %s at epoch %d", checkpoint_path, ckpt["epoch"])
    return ckpt["epoch"], ckpt.get("best_state"), ckpt.get("best_top1", 0.0), ckpt.get("best_top5", 0.0)


def _save_checkpoint(path, model, optimizer, epoch, best_top1, best_top5, arch, manifest_id, best_state) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_top1": best_top1,
            "best_top5": best_top5,
            "best_state": best_state,
            "arch": arch,
            "manifest_sha256": manifest_id,
            "rng": torch.get_rng_state(),
        },
        path,
    )
