"""Shared helpers for the GUI e2e tests (in-process TestClient over a real run)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def build_card(root: Path) -> Path:
    """A small card: one dominant animal, one landscape, one blurry (junk)."""
    from PIL import Image, ImageFilter  # noqa: PLC0415
    from conftest import make_e2e_fixtures  # noqa: PLC0415

    dcim = root / "DCIM"
    dcim.mkdir(parents=True)
    make_e2e_fixtures.noisy((400, 300), 1).save(dcim / "animal.jpg", quality=95)
    (dcim / "animal.jpg.boxes.json").write_text(
        json.dumps([{"cls": "animal", "conf": 0.9, "x0": 0, "y0": 0, "x1": 300, "y1": 250}])
    )
    make_e2e_fixtures.noisy((400, 300), 2).save(dcim / "scene.jpg", quality=95)
    make_e2e_fixtures.noisy((400, 300), 3).filter(
        ImageFilter.GaussianBlur(radius=12)
    ).save(dcim / "blurry.jpg", quality=95)
    return root


def classify(cli_path: str, card: Path, output: Path, *extra: str) -> None:
    subprocess.run(
        [cli_path, "classify", str(card), "--output", str(output), "--detector", "scripted", *extra],
        capture_output=True,
        text=True,
        check=False,
    )


def make_client(output: Path, *, allow_new_labels: bool = False):
    from starlette.testclient import TestClient  # noqa: PLC0415

    from animal_classifier.config import Command, Config  # noqa: PLC0415
    from animal_classifier.gui.app import create_app  # noqa: PLC0415

    config = Config.resolve(
        command=Command.GUI,
        cli={
            "output_root": str(output),
            "allow_new_labels": allow_new_labels or None,
        },
    )
    return TestClient(create_app(config))
