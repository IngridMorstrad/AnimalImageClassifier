"""Dataset, transforms and sampler (DESIGN.md §7.2, §7.3).

The dataset crops with the **same 0.08 margin as inference** so train and inference
see the same framing, and uses a ``WeightedRandomSampler(1/sqrt(count))`` for the
long tail. Transforms are §7.3 verbatim: train does
``RandomResizedCrop(scale=(0.65, 1.0))`` + horizontal flip + colour jitter; eval
does ``Resize(input_size * 1.14)`` → ``CenterCrop`` — horizontal flip only, because
wildlife photos are never upside down.
"""

from __future__ import annotations

import math
from collections import Counter

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from ..images import crop, decode
from .manifest import Sample

#: ImageNet normalization — the statistics timm backbones expect (§7.1).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Same crop margin as inference (§5.3), so the head never sees a different framing.
CROP_MARGIN = 0.08


def train_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(round(input_size * 1.14)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CropDataset(Dataset):
    """Samples cropped to their box (if any) and transformed for the model."""

    def __init__(
        self,
        samples: list[Sample],
        classes: list[str],
        *,
        transform: transforms.Compose,
    ) -> None:
        self._samples = samples
        self._class_to_index = {name: i for i, name in enumerate(classes)}
        self._transform = transform

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self._samples[index]
        decoded = decode(sample.path)
        image = decoded.image
        if sample.box is not None:
            piece = crop(image, sample.box, crop_margin=CROP_MARGIN)
            if piece.image is not None:
                image = piece.image
        tensor = self._transform(image)
        return tensor, self._class_to_index[sample.label]


def balanced_sampler(samples: list[Sample], classes: list[str]) -> WeightedRandomSampler:
    """``1/sqrt(class_count)`` per-sample weights for the long tail (§7.2)."""
    counts = Counter(s.label for s in samples)
    class_weight = {name: 1.0 / math.sqrt(counts.get(name, 1)) for name in classes}
    weights = [class_weight[s.label] for s in samples]
    return WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
    )
