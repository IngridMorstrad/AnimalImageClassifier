"""Probe the MegaDetector v5a checkpoint: confirm it loads and report its structure.

Run: uv run python scripts/probe_md_checkpoint.py
Written for the recon step; kept in-tree so the load can be re-verified at any time.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch

CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "md_v5a.0.0.pt"

# The MegaDetector v5 checkpoint was pickled from the upstream yolov5 repo, whose packages sit
# at the top level (`models.yolo`, `utils.*`). The PyPI `yolov5` distribution namespaces the
# same modules under `yolov5.*`, so the unpickler needs the original names aliased onto it.
YOLOV5_MODULE_ALIASES = ("models", "utils")


def install_yolov5_aliases() -> None:
    for name in YOLOV5_MODULE_ALIASES:
        package = importlib.import_module(f"yolov5.{name}")
        sys.modules.setdefault(name, package)


def main() -> int:
    install_yolov5_aliases()
    print(f"torch {torch.__version__}")
    print(f"checkpoint {CHECKPOINT} ({CHECKPOINT.stat().st_size} bytes)")

    try:
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - recon: report the exact failure
        print(f"LOAD FAILED: {type(exc).__module__}.{type(exc).__qualname__}: {exc}")
        return 1

    print(f"LOAD OK: top-level type = {type(ckpt)}")
    if isinstance(ckpt, dict):
        print(f"top-level keys = {sorted(ckpt.keys())}")
        for key in ("epoch", "best_fitness", "date", "version"):
            if key in ckpt:
                print(f"  {key} = {ckpt[key]!r}")
        model = ckpt.get("model")
        if model is not None:
            print(f"  model class = {type(model).__module__}.{type(model).__qualname__}")
            n_params = sum(p.numel() for p in model.parameters())
            print(f"  model parameters = {n_params:,}")
            names = getattr(model, "names", None)
            print(f"  model.names = {names}")
            stride = getattr(model, "stride", None)
            print(f"  model.stride = {stride}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
