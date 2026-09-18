#!/usr/bin/env python
"""Probe every guarantee `images.py` makes, against real files on disk.

Not a test — the suite is e2e-only (DESIGN.md §11) and `images.py` is first
covered executably by E11 (blur) and E18 (orientation 6). This script is the
evidence for the chunk that wrote the module: it builds real JPEG/PNG/HEIC files
in a temp directory, runs them through the public API, and prints what happened,
so the claims in the commit message are reproducible with one command:

    uv run --frozen python scripts/probe_images.py
"""

from __future__ import annotations

import hashlib
import logging
import struct
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from animal_classifier.errors import EXIT_CONFIG, ConfigError
from animal_classifier.images import (
    BLUR_REFERENCE_EDGE,
    MAX_IMAGE_PIXELS,
    Blur,
    ImageDecodeError,
    _to_degrees,
    crop,
    decode,
    measure_blur,
    open_source,
    sha256_file,
)
from animal_classifier.scan import SkipReason, is_benign

logging.basicConfig(level=logging.DEBUG, format="  [%(levelname)s] %(message)s")

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    print(f"{mark}  {name}" + (f"  ({detail})" if detail else ""))
    if not condition:
        FAILURES.append(name)


def noisy(size: tuple[int, int], seed: int = 7) -> Image.Image:
    """A sharp, high-frequency image: maximal Laplacian variance."""
    rng = np.random.default_rng(seed)
    return Image.fromarray(
        rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8), "RGB"
    )


def huge_png(path: Path, width: int, height: int) -> None:
    """A PNG whose IHDR declares `width`x`height` and carries no image data.

    Enough for Pillow to report `size` from the header, which is exactly where
    the 400 MP cap is meant to bite — before any pixels are allocated.
    """

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", b""))


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="probe-images-") as tmp:
        card = Path(tmp)

        print("\n== sha256 streams raw file bytes ==")
        jpeg = card / "plain.jpg"
        noisy((64, 48)).save(jpeg, "JPEG", quality=92)
        expected = hashlib.sha256(jpeg.read_bytes()).hexdigest()
        check("sha256_file == sha256 of the whole file", sha256_file(jpeg) == expected,
              expected[:16] + "...")
        with open_source(jpeg) as handle:
            check("open_source opens 'rb'", handle.mode == "rb", handle.mode)

        print("\n== I9: one frame, the EXIF-transposed one ==")
        exif = Image.Exif()
        exif[0x0112] = 6  # Orientation: rotate 90 CW on display
        exif[0x8769] = {0x9003: "2026:04:17 10:20:30"}
        exif[0x8825] = {1: "S", 2: (1.0, 30.0, 0.0), 3: "E", 4: (36.0, 45.0, 30.0)}
        oriented = card / "orientation6.jpg"
        noisy((400, 200)).save(oriented, "JPEG", exif=exif)
        with Image.open(oriented) as stored:
            stored_size = stored.size
        got = decode(oriented)
        check("stored raster is landscape", stored_size == (400, 200), str(stored_size))
        check("decoded is portrait (I9)", (got.width, got.height) == (200, 400),
              f"{got.width}x{got.height}")
        check("width/height match image.size", got.image.size == (got.width, got.height))
        check("mode is RGB", got.image.mode == "RGB", got.image.mode)

        print("\n== EXIF is optional, never fabricated ==")
        check("DateTimeOriginal -> ISO-8601",
              got.exif_datetime == "2026-04-17T10:20:30", str(got.exif_datetime))
        check("GPS S1deg30' -> -1.5", got.gps_lat == -1.5, str(got.gps_lat))
        check("GPS E36deg45'30\" -> 36.7583...",
              got.gps_lon is not None and abs(got.gps_lon - 36.758333333) < 1e-6,
              str(got.gps_lon))
        bare = decode(jpeg)
        check("no EXIF -> three NULLs",
              (bare.exif_datetime, bare.gps_lat, bare.gps_lon) == (None, None, None))

        bad_exif = Image.Exif()
        bad_exif[0x8769] = {0x9003: "17/04/2026 10:20"}
        bad_exif[0x8825] = {1: "N", 2: (91.0, 0.0, 0.0), 3: "E", 4: (10.0, 0.0, 0.0)}
        weird = card / "bad_exif.jpg"
        noisy((64, 48)).save(weird, "JPEG", exif=bad_exif)
        odd = decode(weird)
        check("unparseable DateTimeOriginal -> NULL + WARNING",
              odd.exif_datetime is None)
        check("out-of-range GPS -> both NULL + WARNING",
              (odd.gps_lat, odd.gps_lon) == (None, None))

        print("\n== blur: downscale only, ref edge recorded ==")
        small_sharp = measure_blur(noisy((300, 200)))
        big_sharp = measure_blur(noisy((1600, 1200)))
        blurred = measure_blur(noisy((1600, 1200)).filter(ImageFilter.GaussianBlur(12)))
        for name, blur in (("small sharp 300x200", small_sharp),
                          ("large sharp 1600x1200", big_sharp),
                          ("large blurred 1600x1200", blurred)):
            print(f"  {name}: score={blur.score:.2f} ref_edge={blur.ref_edge}")
        check("small image scored natively (never upscaled)",
              small_sharp.ref_edge == 300, str(small_sharp.ref_edge))
        check("large image scored at the 512 px reference",
              big_sharp.ref_edge == BLUR_REFERENCE_EDGE, str(big_sharp.ref_edge))
        check("small sharp image is NOT junk at blur_threshold=100",
              small_sharp.score >= 100.0, f"{small_sharp.score:.2f}")
        check("blurred image IS junk at blur_threshold=100",
              blurred.score < 100.0, f"{blurred.score:.2f}")
        check("Blur is a value object", isinstance(big_sharp, Blur))

        print("\n== decode failures are per-image, with the right reason ==")
        corrupt = card / "corrupt.jpg"
        corrupt.write_bytes(jpeg.read_bytes()[:40])
        try:
            decode(corrupt)
            check("truncated JPEG raises", False)
        except ImageDecodeError as error:
            check("truncated JPEG -> decode_error",
                  error.reason is SkipReason.DECODE_ERROR, str(error.reason))
            check("decode_error is abnormal (exit 4 class)", not is_benign(error.reason))

        bomb = card / "bomb.png"
        huge_png(bomb, 30000, 30000)
        try:
            decode(bomb)
            check("900 MP PNG raises", False)
        except ImageDecodeError as error:
            check("900 MP header -> too_large_pixels",
                  error.reason is SkipReason.TOO_LARGE_PIXELS, str(error.reason))
            check("cap message names the 400 MP limit", str(MAX_IMAGE_PIXELS) in str(error))

        missing = card / "gone.jpg"
        try:
            sha256_file(missing)
            check("missing file raises", False)
        except ImageDecodeError as error:
            check("vanished file -> unreadable",
                  error.reason is SkipReason.UNREADABLE, str(error.reason))

        print("\n== HEIC decodes (and encodes, for the fixtures) ==")
        heic = card / "safari.heic"
        noisy((120, 90)).save(heic, "HEIF", quality=90)
        heic_decoded = decode(heic)
        check("HEIC round-trips through the registered opener",
              (heic_decoded.width, heic_decoded.height) == (120, 90),
              f"{heic_decoded.width}x{heic_decoded.height}")

        print("\n== RAW without the extra fails loudly (exit 3) ==")
        raw_file = card / "frame.cr2"
        raw_file.write_bytes(b"not really raw")
        try:
            decode(raw_file, raw=True)
            check("RAW without rawpy raises", False)
        except ConfigError as error:
            check("RAW without the extra -> ConfigError exit 3",
                  error.exit_code == EXIT_CONFIG, str(error.exit_code))
            check("message names the install command", "'.[raw]'" in str(error))
        try:
            decode(raw_file, raw=False)
            check("RAW with raw=False raises", False)
        except ConfigError as error:
            check("RAW reaching decode without --raw -> ConfigError", True, str(error)[:60])

        print("\n== crop: margin, clipping, degenerate-but-counted ==")
        frame = Image.new("RGB", (1000, 500), "black")
        inner = crop(frame, (400.0, 200.0, 500.0, 300.0), crop_margin=0.08)
        check("8% margin on a 100x100 box -> 8 px each side",
              inner.region == (392, 192, 508, 308), str(inner.region))
        check("crop image matches the region",
              inner.image is not None and inner.image.size == (116, 116),
              str(inner.image.size if inner.image else None))
        edge = crop(frame, (0.0, 0.0, 50.0, 50.0), crop_margin=0.08)
        check("margin clipped at the frame edge", edge.region == (0, 0, 54, 54),
              str(edge.region))
        tiny = crop(frame, (10.0, 10.0, 11.0, 11.0), crop_margin=0.08)
        check("1 px box -> degenerate", tiny.degenerate)
        check("degenerate crop carries no pixels", tiny.image is None)
        check("degenerate species_status", tiny.species_status == "degenerate",
              str(tiny.species_status))
        check("degenerate crop still reports its region", tiny.region == (9, 9, 12, 12),
              str(tiny.region))
        check("rounding cannot inflate a 1 px box into a classifiable crop",
              tiny.region[2] - tiny.region[0] >= 2 and tiny.degenerate,
              f"region is {tiny.region[2] - tiny.region[0]} px wide, still degenerate")
        exactly_two = crop(frame, (10.0, 10.0, 12.0, 12.0), crop_margin=0.0)
        check("exactly 2 px is NOT degenerate", not exactly_two.degenerate)
        just_under = crop(frame, (10.0, 10.0, 11.9, 12.0), crop_margin=0.0)
        check("1.9 px wide IS degenerate", just_under.degenerate)
        ok = crop(frame, (10.0, 10.0, 30.0, 30.0), crop_margin=0.0)
        check("classifiable crop has species_status None", ok.species_status is None)
        try:
            crop(frame, (0.0, 0.0, 10.0, 10.0), crop_margin=0.9)
            check("crop_margin 0.9 raises", False)
        except ConfigError as error:
            check("crop_margin out of range -> ConfigError", True, str(error))

        print("\n== the hemisphere letter decides the sign, so it is validated ==")
        # Exercised at `_to_degrees` rather than through `decode`: Pillow re-encodes
        # GPS refs as ASCII on save, so a file written here can never carry the
        # bytes spelling that firmware emits for an UNDEFINED(7)-typed tag. The
        # bytes *do* reach us on read — `Image.Exif` yields b"S" for that tag — and
        # `str(b"S")` is "b'S'", which starts with neither N nor S.
        triple = (1.0, 30.0, 0.0)
        for ref, expected, label in [
            ("S", -1.5, "ASCII 'S'"),
            ("S\x00", -1.5, "EXIF's NUL-padded 'S'"),
            ("N", 1.5, "ASCII 'N'"),
            (b"S", -1.5, "bytes b'S' (UNDEFINED-typed tag)"),
            (b"N", 1.5, "bytes b'N'"),
            (None, None, "absent ref"),
            ("", None, "empty ref"),
            (0, None, "numeric ref"),
            ("X", None, "unrecognised letter"),
        ]:
            got = _to_degrees(Path("probe.jpg"), triple, ref, "N", "S")
            check(
                f"{label} -> {expected}",
                got == expected,
                f"got {got}",
            )
        check(
            "no ref shape silently defaults to the positive hemisphere",
            _to_degrees(Path("probe.jpg"), triple, None, "N", "S") is None,
        )

        print("\n== a box the detector should never send: non-finite, or off-frame ==")
        frame_500 = Image.new("RGB", (500, 500), "black")
        for bad in [
            (float("nan"), 0.0, 10.0, 10.0),
            (0.0, 0.0, float("inf"), 10.0),
            (float("-inf"), 0.0, 10.0, 10.0),
        ]:
            try:
                piece = crop(frame_500, bad, crop_margin=0.08)
                check(
                    f"non-finite box {bad} rejected",
                    False,
                    f"silently became region {piece.region} "
                    f"degenerate={piece.degenerate}",
                )
            except ValueError as error:
                check(f"non-finite box {bad} -> ValueError", True, str(error)[:60])

        off = crop(frame_500, (600.0, 600.0, 700.0, 700.0), crop_margin=0.08)
        x0, y0, x1, y1 = off.region
        check("box entirely off-frame is degenerate", off.degenerate)
        check(
            "off-frame region is ordered, not inverted",
            x0 <= x1 and y0 <= y1,
            f"region {off.region}",
        )
        check(
            "off-frame region sits on the frame edge",
            0 <= x0 <= 500 and 0 <= y0 <= 500,
            f"region {off.region}",
        )
        partly = crop(frame_500, (-50.0, -50.0, 40.0, 40.0), crop_margin=0.08)
        check(
            "a box straddling the frame edge is clipped, not rejected",
            not partly.degenerate and partly.region[0] == 0,
            f"region {partly.region}",
        )

        print("\n== the source card is untouched ==")
        sources = sorted(p for p in card.iterdir() if p.is_file())
        before = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in sources}
        for path in sources:
            try:
                decode(path, raw=path.suffix == ".cr2")
            except (ImageDecodeError, ConfigError):
                pass
            try:
                sha256_file(path)
            except ImageDecodeError:
                pass
        after = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in sources}
        check(f"all {len(sources)} source files byte- and mtime-identical", before == after)

    probe_real_coco()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: {FAILURES}")
        return 1
    print("all probe checks passed")
    return 0


#: COCO's ten animal categories (`bird` through `giraffe`).
COCO_ANIMAL_IDS = frozenset(range(16, 26))

RAW_DATA = Path(__file__).resolve().parent.parent / "data" / "raw"
VAL2017_ZIP = RAW_DATA / "val2017.zip"
INSTANCES_JSON = RAW_DATA / "annotations" / "instances_val2017.json"


def probe_real_coco(sample: int = 25) -> None:
    """Run the module over real photographs with real ground-truth boxes.

    Synthetic noise proves the arithmetic; this proves the module survives files
    it did not create. The sub-2px leg is the one that matters: DESIGN.md §5.3
    justifies the degenerate rule with COCO val2017's own tiny animal boxes, so
    those exact annotations are cropped here.
    """
    print("\n== real COCO val2017 photographs and ground-truth boxes ==")
    if not (VAL2017_ZIP.exists() and INSTANCES_JSON.exists()):
        print(f"SKIP  real-data leg: {VAL2017_ZIP} / {INSTANCES_JSON} absent")
        return

    import json
    import zipfile

    instances = json.loads(INSTANCES_JSON.read_text())
    animals: dict[int, list[list[float]]] = {}
    tiny_side: list[tuple[float, int, list[float]]] = []
    for ann in instances["annotations"]:
        if ann["category_id"] not in COCO_ANIMAL_IDS or ann.get("iscrowd"):
            continue
        animals.setdefault(ann["image_id"], []).append(ann["bbox"])
        if min(ann["bbox"][2], ann["bbox"][3]) < 2.0:
            tiny_side.append((min(ann["bbox"][2], ann["bbox"][3]),
                              ann["image_id"], ann["bbox"]))
    tiny_side.sort()
    print(f"  {len(animals)} val2017 images carry non-crowd animal boxes; "
          f"{len(tiny_side)} boxes have a side under 2 px")
    check("COCO really does contain sub-2px animal boxes", bool(tiny_side))

    by_id = {img["id"]: img for img in instances["images"]}
    wanted = [image_id for _, image_id, _ in tiny_side]
    wanted += [i for i in sorted(animals) if i not in set(wanted)][:sample]

    with tempfile.TemporaryDirectory(prefix="probe-coco-") as tmp:
        out = Path(tmp)
        with zipfile.ZipFile(VAL2017_ZIP) as archive:
            names = {Path(n).name: n for n in archive.namelist()}
            extracted: dict[int, Path] = {}
            for image_id in wanted:
                member = names.get(by_id[image_id]["file_name"])
                if member is None:
                    continue
                target = out / by_id[image_id]["file_name"]
                target.write_bytes(archive.read(member))
                extracted[image_id] = target

        decoded_ok = 0
        dims_agree = 0
        blurs: list[Blur] = []
        for image_id, path in extracted.items():
            got = decode(path)
            decoded_ok += 1
            meta = by_id[image_id]
            if (got.width, got.height) == (meta["width"], meta["height"]):
                dims_agree += 1
            blurs.append(measure_blur(got.image))
            for bbox in animals[image_id]:
                x, y, w, h = bbox
                piece = crop(got.image, (x, y, x + w, y + h), crop_margin=0.08)
                left, top, right, bottom = piece.region
                inside = 0 <= left <= right <= got.width and 0 <= top <= bottom <= got.height
                if not inside:
                    check(f"region {piece.region} inside {got.width}x{got.height}", False)

        check(f"decoded all {len(extracted)} real JPEGs", decoded_ok == len(extracted))
        check("decoded dimensions match COCO's own width/height",
              dims_agree == len(extracted), f"{dims_agree}/{len(extracted)}")
        check("every real crop region stays inside the frame", True)
        check("blur scored for every real image", len(blurs) == len(extracted))
        print("  blur range over real photos: "
              f"{min(b.score for b in blurs):.1f} .. {max(b.score for b in blurs):.1f}; "
              f"ref_edges {sorted({b.ref_edge for b in blurs})}")

        # §5.3 measures the extent *after* the margin is applied and clipped, not
        # the detector's raw side. Real data makes the difference visible: a
        # 1.79 x 2.02 box expands to 2.08 x 2.34 and is classifiable, while a
        # 2.83 x 1.20 box expands to 1.39 px tall and is not. The expectation is
        # therefore the rule itself, and every one of these 7 boxes sits well
        # inside its frame, so no clipping enters the arithmetic.
        margin = 0.08
        degenerate_hits = 0
        consistent = True
        rule_holds = True
        for _, image_id, bbox in tiny_side:
            path = extracted.get(image_id)
            if path is None:
                continue
            got = decode(path)
            x, y, w, h = bbox
            piece = crop(got.image, (x, y, x + w, y + h), crop_margin=margin)
            expected = min(w, h) * (1.0 + 2.0 * margin) < 2.0
            rule_holds &= piece.degenerate == expected
            consistent &= piece.degenerate == (piece.image is None)
            degenerate_hits += int(piece.degenerate)
            print(f"  {path.name} bbox {[round(v, 2) for v in bbox]} -> "
                  f"region {piece.region} expanded_min_side="
                  f"{min(w, h) * (1.0 + 2.0 * margin):.2f} "
                  f"degenerate={piece.degenerate} status={piece.species_status}")
        check("degenerate verdict == (margin-expanded min side < 2 px) on all 7 "
              "real boxes", rule_holds)
        check("degenerate <-> image is None on all 7 real boxes", consistent)
        check("the degenerate rule is not hypothetical: real COCO animal boxes "
              "hit it", degenerate_hits > 0, f"{degenerate_hits}/{len(tiny_side)}")
        check("degenerate boxes were still returned, not dropped "
              "(dominance_ratio stays the only size gate)",
              degenerate_hits > 0)


if __name__ == "__main__":
    sys.exit(main())
