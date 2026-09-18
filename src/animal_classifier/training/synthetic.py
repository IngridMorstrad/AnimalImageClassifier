"""A deterministic synthetic dataset for the smoke path (DESIGN.md §7.6).

Coloured-shape images generated from a seed, so the training smoke test (E10) is
reproducible and finishes in seconds. Each class is a distinct (shape, colour)
combination that a tiny CNN can learn well enough to prove the train → eval →
export → inference path end to end — which is all this is for. It is not wildlife
and does not pretend to be.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

#: The synthetic classes: shape + colour, each trivially separable.
SYNTHETIC_CLASSES = (
    ("red_circle", "circle", (220, 40, 40)),
    ("green_square", "square", (40, 200, 40)),
    ("blue_triangle", "triangle", (40, 80, 220)),
)


def _draw(shape: str, colour: tuple[int, int, int], size: int, rng: np.random.Generator) -> Image.Image:
    """One noisy image with a jittered shape of the given kind and colour."""
    background = rng.integers(0, 60, (size, size, 3), dtype=np.uint8)
    image = Image.fromarray(background, mode="RGB")
    draw = ImageDraw.Draw(image)
    margin = size // 5
    jitter = int(size * 0.08)
    dx, dy = (int(rng.integers(-jitter, jitter + 1)) for _ in range(2))
    box = [margin + dx, margin + dy, size - margin + dx, size - margin + dy]
    if shape == "circle":
        draw.ellipse(box, fill=colour)
    elif shape == "square":
        draw.rectangle(box, fill=colour)
    else:  # triangle
        draw.polygon(
            [(box[0], box[3]), ((box[0] + box[2]) // 2, box[1]), (box[2], box[3])],
            fill=colour,
        )
    return image


def generate(
    root: Path, *, per_class: int = 40, size: int = 64, seed: int = 0
) -> Path:
    """Write a synthetic dataset and its manifest under ``root``. Returns the manifest.

    Every image gets an explicit ``split`` (one in five ``val``) so the smoke test
    does not depend on :func:`split_for`'s basename hashing for such tiny counts.
    """
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest = root / "synthetic.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for label, shape, colour in SYNTHETIC_CLASSES:
            class_dir = root / label
            class_dir.mkdir(exist_ok=True)
            for index in range(per_class):
                image = _draw(shape, colour, size, rng)
                image_path = class_dir / f"{label}_{index:03d}.png"
                image.save(image_path)
                split = "val" if index % 5 == 0 else "train"
                # Absolute paths, so load_manifest resolves them unambiguously
                # regardless of the manifest's own location.
                handle.write(
                    json.dumps(
                        {"path": str(image_path.resolve()), "label": label, "split": split}
                    )
                    + "\n"
                )
    return manifest
