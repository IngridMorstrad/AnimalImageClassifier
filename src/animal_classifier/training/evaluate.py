"""Eval and temperature calibration (DESIGN.md §7.4).

``eval`` computes top-1/top-5, macro and per-class recall, a support count and a
confusion matrix over a manifest split. Because ``min_species_confidence`` gates the
``unknown`` label, confidence honesty matters, so ``--calibrate`` fits a single
temperature by LBFGS on the val split's NLL.

**Calibration writes a new artifact, never an edit (I8, §7.4).** ``model_id``
becomes ``{old}+cal{n}`` and ``train.calibrated_from`` records the source; writing
into an existing artifact path is refused, because two behaviourally different
models must never share one ``model_id``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..classify.artifact import Artifact
from . import dataset as ds
from .manifest import Sample

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvalMetrics:
    top1: float
    top5: float
    macro_recall: float
    support: int
    per_class_recall: dict[str, float] = field(default_factory=dict)


def evaluate(artifact: Artifact, samples: list[Sample], *, device: str = "cpu") -> EvalMetrics:
    """Top-1/top-5 and per-class recall for ``artifact`` over ``samples`` (§7.4)."""
    classes = list(artifact.label_slugs)
    dev = torch.device(device)
    model = artifact.module.to(dev).eval()
    loader = DataLoader(
        ds.CropDataset(samples, classes, transform=ds.eval_transform(artifact.input_size)),
        batch_size=32,
        shuffle=False,
    )
    correct1 = correct5 = total = 0
    per_class_correct = dict.fromkeys(range(len(classes)), 0)
    per_class_total = dict.fromkeys(range(len(classes)), 0)
    with torch.inference_mode():
        for images, targets in loader:
            images, targets = images.to(dev), targets.to(dev)
            logits = model(images)
            maxk = min(5, logits.size(1))
            _, pred = logits.topk(maxk, dim=1)
            hit = pred.eq(targets.view(-1, 1))
            correct1 += int(hit[:, 0].sum().item())
            correct5 += int(hit.any(dim=1).sum().item())
            total += targets.size(0)
            for t, h1 in zip(targets.tolist(), hit[:, 0].tolist()):
                per_class_total[t] += 1
                per_class_correct[t] += int(h1)
    total = max(total, 1)
    recalls = {
        classes[i]: (per_class_correct[i] / per_class_total[i])
        for i in range(len(classes))
        if per_class_total[i] > 0
    }
    macro = sum(recalls.values()) / len(recalls) if recalls else 0.0
    return EvalMetrics(
        top1=correct1 / total,
        top5=correct5 / total,
        macro_recall=macro,
        support=total,
        per_class_recall=recalls,
    )


def fit_temperature(artifact: Artifact, samples: list[Sample], *, device: str = "cpu") -> float:
    """Fit one temperature by LBFGS on the val NLL (§7.4). Returns T > 0."""
    classes = list(artifact.label_slugs)
    dev = torch.device(device)
    model = artifact.module.to(dev).eval()
    loader = DataLoader(
        ds.CropDataset(samples, classes, transform=ds.eval_transform(artifact.input_size)),
        batch_size=32,
        shuffle=False,
    )
    logits_all, targets_all = [], []
    with torch.inference_mode():
        for images, targets in loader:
            logits_all.append(model(images.to(dev)))
            targets_all.append(targets.to(dev))
    logits = torch.cat(logits_all)
    targets = torch.cat(targets_all)

    temperature = torch.nn.Parameter(torch.ones(1, device=dev))
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=50)
    nll = torch.nn.CrossEntropyLoss()

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(logits / temperature.clamp_min(1e-3), targets)
        loss.backward()
        return loss

    optimizer.step(_closure)
    return float(temperature.clamp_min(1e-3).item())
