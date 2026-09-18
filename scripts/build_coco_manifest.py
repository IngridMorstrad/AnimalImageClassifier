#!/usr/bin/env python
"""Build a species training manifest from COCO val2017 (DESIGN.md §7.2).

Reads ``data/raw/annotations/instances_val2017.json``, takes the ten animal
categories (ids 16-25: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra,
giraffe), drops the ``iscrowd=1`` instances, and emits one JSONL sample per
instance with its ``bbox`` converted from COCO's ``[x, y, w, h]`` to
``[x0, y0, x1, y1]``. The split is assigned by :func:`split_for` on the basename
so every crop of one photograph lands on the same side (§7.2's leakage rule).

Usage:

    uv run --frozen python scripts/build_coco_manifest.py \\
        --annotations data/raw/annotations/instances_val2017.json \\
        --images data/raw/val2017 \\
        --out data/manifests/coco_species.jsonl \\
        [--classes zebra,elephant,giraffe] [--split train]

``--classes`` selects a subset; ``--split`` filters the output (a training run must
be fed ``--split train`` only, per §7.2). Fails loudly if the annotations file is
absent, naming the download.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# COCO category id -> our label slug (ids 16-25, §7.2).
COCO_ANIMAL_CATEGORIES = {
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
}

DOWNLOAD_HINT = (
    "download COCO val2017 annotations from "
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip "
    "and images from http://images.cocodataset.org/zips/val2017.zip"
)


def split_for(file_name: str) -> str:
    """The §7.2 split, duplicated here so the builder has no package import."""
    import hashlib
    import os

    key = os.path.basename(file_name)
    return "val" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def build(
    annotations: Path,
    images_dir: Path,
    out: Path,
    *,
    classes: set[str] | None,
    split: str | None,
) -> int:
    if not annotations.is_file():
        sys.stderr.write(f"annotations not found: {annotations}\n  {DOWNLOAD_HINT}\n")
        return 3

    data = json.loads(annotations.read_text())
    images = {img["id"]: img["file_name"] for img in data["images"]}

    kept = 0
    dropped_crowd = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for ann in sorted(data["annotations"], key=lambda a: (a["image_id"], a["id"])):
            label = COCO_ANIMAL_CATEGORIES.get(ann["category_id"])
            if label is None:
                continue
            if classes is not None and label not in classes:
                continue
            if ann.get("iscrowd", 0) == 1:
                dropped_crowd += 1
                continue
            file_name = images.get(ann["image_id"])
            if file_name is None:
                continue
            sample_split = split_for(file_name)
            if split is not None and sample_split != split:
                continue
            x, y, w, h = ann["bbox"]
            handle.write(
                json.dumps(
                    {
                        "path": str(images_dir / file_name),
                        "label": label,
                        "box": [x, y, x + w, y + h],
                        "split": sample_split,
                    }
                )
                + "\n"
            )
            kept += 1

    sys.stderr.write(
        f"wrote {kept} samples to {out} "
        f"(dropped {dropped_crowd} iscrowd instances)\n"
    )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", type=str, default=None)
    parser.add_argument("--split", type=str, default=None, choices=["train", "val"])
    args = parser.parse_args(argv[1:])
    classes = set(args.classes.split(",")) if args.classes else None
    return build(args.annotations, args.images, args.out, classes=classes, split=args.split)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
