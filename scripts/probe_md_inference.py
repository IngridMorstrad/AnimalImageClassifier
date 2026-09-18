"""Probe that the MegaDetector v5a checkpoint runs a real forward pass on CPU.

Run: uv run python scripts/probe_md_inference.py
Recon-step script: proves the weights are usable, not just loadable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probe_md_checkpoint import CHECKPOINT, install_yolov5_aliases


def main() -> int:
    install_yolov5_aliases()
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model = (ckpt.get("ema") or ckpt["model"]).float().eval()

    # MegaDetector v5 runs at 1280px; use a single 640px frame here to keep the probe cheap.
    frame = torch.zeros(1, 3, 640, 640)
    started = time.perf_counter()
    with torch.no_grad():
        out = model(frame)
    elapsed = time.perf_counter() - started

    raw = out[0] if isinstance(out, (list, tuple)) else out
    print(f"forward OK in {elapsed:.2f}s on CPU")
    print(f"  raw prediction tensor shape = {tuple(raw.shape)}")
    print(f"  class names = {model.names}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
