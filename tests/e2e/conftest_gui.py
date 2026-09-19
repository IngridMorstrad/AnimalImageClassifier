"""Shared helpers for the GUI e2e tests (in-process TestClient over a real run)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def build_card(root: Path) -> Path:
    """A small card: one dominant animal, one landscape, one blurry (junk)."""
    from conftest import make_e2e_fixtures
    from PIL import ImageFilter

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
    from starlette.testclient import TestClient

    from animal_classifier.config import Command, Config
    from animal_classifier.gui.app import create_app

    config = Config.resolve(
        command=Command.GUI,
        cli={
            "output_root": str(output),
            "allow_new_labels": allow_new_labels or None,
        },
    )
    return TestClient(create_app(config))



def build_orientation6_card(root: Path) -> Path:
    """A card holding one EXIF-orientation-6 photo whose stored raster is landscape.

    Orientation 6 means "rotate 90° CW to display", so a 400x300 *stored* raster is a
    300x400 *displayed* image. §5.2's single frame is the displayed one, so the catalog
    must record width < height — that is invariant I9, and it is the whole point of the
    fixture.
    """
    import json


    dcim = root / "DCIM"
    dcim.mkdir(parents=True)
    photo = dcim / "portrait.jpg"

    # A landscape raster (wider than tall) ...
    from conftest import make_e2e_fixtures

    image = make_e2e_fixtures.noisy((400, 300), 7)
    exif = image.getexif()
    exif[0x0112] = 6  # Orientation = 6
    image.save(photo, exif=exif, quality=95)

    # ... with a box stated in the TRANSPOSED frame (300 wide x 400 tall).
    (dcim / "portrait.jpg.boxes.json").write_text(
        json.dumps([{"cls": "animal", "conf": 0.9, "x0": 20, "y0": 30, "x1": 280, "y1": 370}])
    )
    return root
