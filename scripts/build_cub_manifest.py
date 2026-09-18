#!/usr/bin/env python
"""Build a bird training manifest from CUB-200-2011 (DESIGN.md §7.2).

Reads an extracted ``CUB_200_2011`` tree and emits one JSONL sample per image with
the species from its directory name, the box from ``bounding_boxes.txt``, and the
**official** split from ``train_test_split.txt`` (an explicit ``split`` field always
wins over :func:`split_for`, §7.2).

The species key is derived by stripping CUB's ``NNN.`` directory prefix and
normalising the rest — this is DEFECT 2's fix: ``022.Chuck_will_widow`` becomes
``chuck_will_widow``, never ``022_chuck_will_widow``, so a trained head files into a
human-readable directory.

Usage:

    uv run --frozen python scripts/build_cub_manifest.py \\
        --root data/raw/CUB_200_2011 \\
        --out data/manifests/cub_birds.jsonl \\
        [--classes 5]        # keep only the first N species (for a fast test head)

Fails loudly with the download command if the tree is absent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DOWNLOAD_HINT = (
    "extract CUB-200-2011 into data/raw/CUB_200_2011 — the archive is at "
    "https://www.vision.caltech.edu/datasets/cub_200_2011/ (a HuggingFace mirror "
    "also works if caltech.edu rate-limits)"
)


def species_key(directory_name: str) -> str:
    """``022.Chuck_will_widow`` → ``chuck_will_widow`` (DEFECT 2)."""
    _, _, rest = directory_name.partition(".")
    name = rest or directory_name
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _read_table(path: Path) -> dict[str, str]:
    """CUB's ``<id> <value>`` whitespace tables, keyed by image id."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        image_id, _, value = line.partition(" ")
        out[image_id] = value.strip()
    return out


def build(root: Path, out: Path, *, keep_classes: int | None) -> int:
    images_txt = root / "images.txt"
    boxes_txt = root / "bounding_boxes.txt"
    split_txt = root / "train_test_split.txt"
    images_dir = root / "images"
    for required in (images_txt, boxes_txt, split_txt, images_dir):
        if not required.exists():
            sys.stderr.write(f"CUB tree incomplete: {required} missing\n  {DOWNLOAD_HINT}\n")
            return 3

    names = _read_table(images_txt)      # id -> "022.Chuck_will_widow/xxx.jpg"
    boxes = _read_table(boxes_txt)       # id -> "x y w h"
    splits = _read_table(split_txt)      # id -> "1" (train) | "0" (test/val)

    selected: set[str] | None = None
    if keep_classes is not None:
        all_keys = sorted({species_key(rel.split("/")[0]) for rel in names.values()})
        selected = set(all_keys[:keep_classes])

    kept = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for image_id, relative in sorted(names.items(), key=lambda kv: int(kv[0])):
            directory = relative.split("/")[0]
            label = species_key(directory)
            if selected is not None and label not in selected:
                continue
            path = images_dir / relative
            if not path.is_file():
                sys.stderr.write(f"image listed but missing: {path}\n")
                return 3
            x, y, w, h = (float(v) for v in boxes[image_id].split())
            # CUB's flag is 1 for *train*; everything else is our val split.
            split = "train" if splits.get(image_id) == "1" else "val"
            handle.write(
                json.dumps(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "box": [x, y, x + w, y + h],
                        "split": split,
                    }
                )
                + "\n"
            )
            kept += 1

    sys.stderr.write(f"wrote {kept} samples to {out}\n")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=None, dest="keep_classes")
    args = parser.parse_args(argv[1:])
    return build(args.root, args.out, keep_classes=args.keep_classes)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
