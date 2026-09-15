"""Probe that the locally downloaded ImageNet backbones load into timm and run a forward pass.

Run: uv run python scripts/probe_backbones.py

The usual timm/torchvision weight hosts (huggingface.co, download.pytorch.org) are unreachable
from this sandbox, so weights are fetched from GitHub release assets and loaded from disk with
`pretrained=False` + an explicit `load_state_dict`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import timm
import torch

BACKBONE_DIR = Path(__file__).resolve().parent.parent / "models" / "backbones"

# (timm architecture name, checkpoint filename)
BACKBONES = [
    ("efficientnet_b0", "efficientnet_b0_ra-3dd342df.pth"),
    ("convnext_nano", "convnext_nano_d1h-7eb4bdea.pth"),
]


def probe(arch: str, filename: str) -> bool:
    path = BACKBONE_DIR / filename
    if not path.exists():
        print(f"{arch}: MISSING {path}")
        return False

    state = torch.load(path, map_location="cpu", weights_only=True)
    state = state.get("state_dict", state)

    model = timm.create_model(arch, pretrained=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    n_params = sum(p.numel() for p in model.parameters())

    model.eval()
    with torch.no_grad():
        logits = model(torch.zeros(1, 3, 224, 224))

    print(
        f"{arch}: LOAD OK params={n_params:,} "
        f"missing={len(missing)} unexpected={len(unexpected)} "
        f"logits={tuple(logits.shape)}"
    )
    if missing or unexpected:
        print(f"  missing[:5]={missing[:5]} unexpected[:5]={unexpected[:5]}")
    return not missing and not unexpected


def main() -> int:
    print(f"timm {timm.__version__} / torch {torch.__version__}")
    results = [probe(arch, filename) for arch, filename in BACKBONES]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
