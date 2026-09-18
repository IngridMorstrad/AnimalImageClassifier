"""A tiny from-scratch CNN for the synthetic smoke path (DESIGN.md §7.6).

Four conv blocks, ~180 k params, global-pooled into a linear head. It exists so the
e2e suite can prove the whole train → eval → export → inference path in *seconds*
on a synthetic dataset, without the 1.1 GB CUB archive or a pretrained backbone. It
is explicitly **not** the proof of the training subsystem — E7's real EfficientNet
finetune is — but it is a real network trained by the real trainer, so E10 exercises
the same code path a real run does.
"""

from __future__ import annotations

from torch import nn


class TinyCNN(nn.Module):
    """A small classifier: 4 conv blocks → global average pool → linear head."""

    def __init__(self, num_classes: int, *, in_channels: int = 3) -> None:
        super().__init__()
        channels = (16, 32, 64, 128)
        blocks: list[nn.Module] = []
        previous = in_channels
        for width in channels:
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(previous, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),
                )
            )
            previous = width
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)
