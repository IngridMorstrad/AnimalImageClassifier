#!/usr/bin/env python
"""Freeze the real-image expectation lists for E4 and E7 (DESIGN.md §11.1).

Emits two JSON files from ``instances_val2017.json``, both restricted to
``split_for(file_name) == "val"`` (§7.2's leakage rule) and sorted by ``image_id``:

- ``tests/e2e/data/coco_dominance.json`` — ``{threshold_high, threshold_low, high,
  low}``. ``high`` is **every** qualifying val-bucket image whose ground-truth area
  ratio (largest animal ÷ second-largest) exceeds ``threshold_high``; ``low`` is
  every one below ``threshold_low``. E4 asserts ``>= ceil(0.8 * len(list))`` on each
  side and reads the lengths from this file, so **no list length is ever hard-coded**
  — that is what makes §11.2's "re-freeze at a stricter ratio" remedy actually
  possible.
- ``tests/e2e/data/coco_species.json`` — every val-bucket image with **exactly one**
  non-crowd animal instance whose ``area_frac >= min_area_frac``, GT class in the
  selected set. E7 asserts species identity over these.

``iscrowd=1`` instances are dropped everywhere (they are not single animals).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

COCO_ANIMAL_CATEGORIES = {
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
}

#: E7's seven classes (§11.2) — the subset with enough val support to assert on.
E7_CLASSES = ("zebra", "elephant", "giraffe", "bear", "cow", "sheep", "bird")


def split_for(file_name: str) -> str:
    key = os.path.basename(file_name)
    return "val" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def build(
    annotations: Path,
    out_dir: Path,
    *,
    threshold_high: float,
    threshold_low: float,
    min_area_frac: float,
) -> int:
    if not annotations.is_file():
        sys.stderr.write(
            f"annotations not found: {annotations}\n  download "
            "http://images.cocodataset.org/annotations/annotations_trainval2017.zip\n"
        )
        return 3

    data = json.loads(annotations.read_text())
    images = {img["id"]: img for img in data["images"]}
    per_image: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        label = COCO_ANIMAL_CATEGORIES.get(ann["category_id"])
        if label is None or ann.get("iscrowd", 0) == 1:
            continue
        per_image[ann["image_id"]].append({"label": label, "area": float(ann["area"])})

    high: list[dict] = []
    low: list[dict] = []
    species: list[dict] = []

    for image_id in sorted(per_image):
        image = images[image_id]
        file_name = image["file_name"]
        if split_for(file_name) != "val":
            continue
        instances = sorted(per_image[image_id], key=lambda a: a["area"], reverse=True)
        frame_area = float(image["width"]) * float(image["height"])

        # The single-animal species list (E7).
        if len(instances) == 1 and instances[0]["label"] in E7_CLASSES:
            if instances[0]["area"] / frame_area >= min_area_frac:
                species.append(
                    {
                        "image_id": image_id,
                        "file_name": file_name,
                        "label": instances[0]["label"],
                        "area_frac": instances[0]["area"] / frame_area,
                    }
                )

        # The dominance lists (E4) need at least two animals to have a ratio.
        if len(instances) < 2:
            continue
        second = instances[1]["area"]
        if second <= 0:
            continue
        ratio = instances[0]["area"] / second
        entry = {
            "image_id": image_id,
            "file_name": file_name,
            "gt_ratio": ratio,
            "top_class": instances[0]["label"],
            "second_class": instances[1]["label"],
        }
        if ratio > threshold_high:
            high.append(entry)
        elif ratio < threshold_low:
            low.append(entry)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coco_dominance.json").write_text(
        json.dumps(
            {
                "threshold_high": threshold_high,
                "threshold_low": threshold_low,
                "high": high,
                "low": low,
            },
            indent=2,
        )
        + "\n"
    )
    (out_dir / "coco_species.json").write_text(
        json.dumps({"min_area_frac": min_area_frac, "images": species}, indent=2) + "\n"
    )
    sys.stderr.write(
        f"coco_dominance.json: high={len(high)} (ratio > {threshold_high}), "
        f"low={len(low)} (ratio < {threshold_low})\n"
        f"coco_species.json: {len(species)} single-animal val images "
        f"(area_frac >= {min_area_frac})\n"
    )
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("tests/e2e/data"))
    parser.add_argument("--threshold-high", type=float, default=3.0)
    parser.add_argument("--threshold-low", type=float, default=1.3)
    # 0.10, not §11.1's 0.20: re-measured against instances_val2017.json, the
    # val-bucket single-animal images at >= 0.20 number only 9, too thin to assert
    # species identity on. At >= 0.10 there are exactly 20 — the frozen list size
    # §11.1 intends. Recorded in PROGRESS.md, per the design's re-freeze rule; the
    # test reads the length from the JSON, so nothing downstream hard-codes it.
    parser.add_argument("--min-area-frac", type=float, default=0.10)
    args = parser.parse_args(argv[1:])
    return build(
        args.annotations,
        args.out_dir,
        threshold_high=args.threshold_high,
        threshold_low=args.threshold_low,
        min_area_frac=args.min_area_frac,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
