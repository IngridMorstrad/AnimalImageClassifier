"""Re-measure every COCO-derived number DESIGN.md iteration 3 asserts on.

Read-only. Prints measured vs claimed so a design reviewer can check the
E4 / E7 / §11.1 arithmetic instead of trusting the document.
"""

import hashlib
import json
import math
import os
from collections import Counter, defaultdict

ANN = "data/raw/annotations/instances_val2017.json"

# DESIGN.md §7.2 split_for(), copied verbatim so the reviewer measures the
# design's function and not an approximation of it.
def split_for(path: str) -> str:
    key = os.path.basename(path)
    return "val" if int(hashlib.sha1(key.encode()).hexdigest(), 16) % 5 == 0 else "train"


def main() -> None:
    with open(ANN) as fh:
        d = json.load(fh)

    animal_names = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
    cats = {c["id"]: c["name"] for c in d["categories"] if c["name"] in animal_names}
    print(f"animal category ids: {sorted(cats)}")

    images = {im["id"]: im for im in d["images"]}
    per_image = defaultdict(list)
    crowd = 0
    for a in d["annotations"]:
        if a["category_id"] not in cats:
            continue
        if a.get("iscrowd", 0) == 1:
            crowd += 1
            continue
        per_image[a["image_id"]].append(a)

    n_inst = sum(len(v) for v in per_image.values())
    print(f"\n[§1/§7.2] non-crowd animal instances = {n_inst} (claim 2666); crowd dropped = {crowd} (claim 34)")
    print(f"[§1/§7.2] images with >=1 animal      = {len(per_image)} (claim 1016)")

    cls_counts = Counter(cats[a["category_id"]] for v in per_image.values() for a in v)
    print(f"[§1] per-class non-crowd counts       = {dict(cls_counts.most_common())}")

    tr = sum(1 for iid in per_image if split_for(images[iid]["file_name"]) == "train")
    va = len(per_image) - tr
    print(f"[§7.2] split over the 1016            = {tr} train / {va} val (claim 813/203)")

    multi = {iid: v for iid, v in per_image.items() if len(v) > 1}
    print(f"\n[§11.1] multi-animal images           = {len(multi)} (claim 471)")

    def ratio(v):
        areas = sorted((a["bbox"][2] * a["bbox"][3] for a in v), reverse=True)
        return math.inf if areas[1] == 0 else areas[0] / areas[1]

    dom = sum(1 for v in multi.values() if ratio(v) >= 1.6)
    print(f"[§1] dominance GT at the 1.6 gate     = {dom} dominant / {len(multi) - dom} multiple (claim 235/236)")

    val_multi = {i: v for i, v in multi.items() if split_for(images[i]["file_name"]) == "val"}
    print(f"[§11.1] multi-animal images in val    = {len(val_multi)}")
    for th, claim in ((3.0, 22), (4.0, 19)):
        n = sum(1 for v in val_multi.values() if ratio(v) > th)
        print(f"[§11.1] val ratio > {th}   high list = {n:3d} (claim {claim})  -> E4 asserts >= ceil(0.8*n) = {math.ceil(0.8 * n)}")
    n_low = sum(1 for v in val_multi.values() if ratio(v) < 1.3)
    print(f"[§11.1] val ratio < 1.3    low  list = {n_low:3d} (claim 29)  -> E4 asserts >= ceil(0.8*n) = {math.ceil(0.8 * n_low)}")

    seven = {"zebra", "elephant", "giraffe", "bear", "cow", "sheep", "bird"}
    single = 0
    for iid, v in per_image.items():
        if len(v) != 1 or split_for(images[iid]["file_name"]) != "val":
            continue
        a = v[0]
        im = images[iid]
        if cats[a["category_id"]] not in seven:
            continue
        if (a["bbox"][2] * a["bbox"][3]) / (im["width"] * im["height"]) >= 0.20:
            single += 1
    print(f"\n[§11.1] coco_species.json candidates = {single} (claim 24; the file is frozen at 20 and E7 asserts >= 14/20)")

    tr7 = va7 = 0
    for iid, v in per_image.items():
        s = split_for(images[iid]["file_name"])
        for a in v:
            if cats[a["category_id"]] in seven:
                if s == "train":
                    tr7 += 1
                else:
                    va7 += 1
    print(f"[E7]  7-class crops                  = {tr7} train / {va7} val (claim 1625/349)")
    if va7:
        maj = Counter(
            cats[a["category_id"]]
            for iid, v in per_image.items()
            if split_for(images[iid]["file_name"]) == "val"
            for a in v
            if cats[a["category_id"]] in seven
        ).most_common(1)[0]
        print(f"[E7]  val majority class             = {maj[0]} {maj[1]}/{va7} = {maj[1] / va7:.3f} (claim 0.229); chance = {1 / 7:.3f}")
        bear_val = sum(
            1
            for iid, v in per_image.items()
            if split_for(images[iid]["file_name"]) == "val"
            for a in v
            if cats[a["category_id"]] == "bear"
        )
        print(f"[E7]  bear val instances             = {bear_val} (claim 11)")


if __name__ == "__main__":
    main()
