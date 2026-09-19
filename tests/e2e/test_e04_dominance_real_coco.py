"""E4: the dominance rule on **real** COCO images with **real** MegaDetector (§11.1).

The other dominance tests (E5, E6) state geometry exactly with the scripted detector,
which proves the *rule*. This proves the rule holds on photographs the tool did not
choose, with boxes the real detector produced.

**No numeric literals.** The thresholds and both list lengths are read from
``tests/e2e/data/coco_dominance.json`` (built by ``scripts/build_coco_dominance.py``),
and the assertion is ``>= ceil(0.8 * len(list))`` on each side. That is what makes
§11.2's sanctioned remedy — re-freeze at a stricter ground-truth ratio and let the
list length follow the data — possible without touching this file's arithmetic.

**Aggregate, not per-image.** MegaDetector's box set legitimately differs from the
annotator's (it finds animals COCO did not label, and vice versa), so a per-image
assertion would be asserting the annotator. The claim is statistical: images whose
*ground-truth* area ratio is lopsided should mostly not come out ``multiple``, and
images whose ratio is near 1.0 should mostly come out ``multiple``. Per-image results
are printed so a failure is diagnosable.

No species identity is asserted here — that is E7's job.
"""

from __future__ import annotations

import json
import math
import sqlite3
import urllib.request
from pathlib import Path

import pytest

from animal_classifier.decide import NON_SPECIES_LABELS
from animal_classifier.detect.megadetector import EXPECTED_BYTES

pytestmark = pytest.mark.slow

DATA = Path(__file__).parent / "data" / "coco_dominance.json"
WEIGHTS = Path("models/md_v5a.0.0.pt")
COCO_URL = "http://images.cocodataset.org/val2017/{name}"


def _weights_present() -> bool:
    return WEIGHTS.is_file() and WEIGHTS.stat().st_size == EXPECTED_BYTES


def _fetch(name: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(COCO_URL.format(name=name), timeout=30) as r:
            data = r.read()
    except Exception:
        return False
    if len(data) < 1000:
        return False
    dest.write_bytes(data)
    return True


@pytest.fixture(scope="module")
def dominance_run(cli_path, tmp_path_factory):
    """Fetch both frozen lists onto one card and classify it with the real detector."""
    if not DATA.is_file():
        pytest.skip(f"{DATA} not built; run scripts/build_coco_dominance.py")
    if not _weights_present():
        pytest.skip("MegaDetector weights absent (280 MB); download to run E4")

    frozen = json.loads(DATA.read_text())
    root = tmp_path_factory.mktemp("e4")
    card = root / "card" / "DCIM"
    card.mkdir(parents=True)

    fetched: dict[str, str] = {}  # file_name -> bucket
    for bucket in ("high", "low"):
        for entry in frozen[bucket]:
            if _fetch(entry["file_name"], card / entry["file_name"]):
                fetched[entry["file_name"]] = bucket
    if len(fetched) < 10:
        pytest.skip("could not fetch enough COCO images (network unavailable)")

    output = root / "pics"
    import subprocess

    subprocess.run(
        [cli_path, "classify", str(card.parent), "--output", str(output),
         "--detector", "megadetector"],
        capture_output=True, text=True, check=False,
    )

    # Map file_name -> assigned label, from the catalog.
    con = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        labels = {
            Path(row["path"]).name: row["label"]
            for row in con.execute(
                "SELECT s.path, i.label FROM images i JOIN sources s USING(sha256)"
            )
        }
    finally:
        con.close()
    return frozen, fetched, labels


def test_every_label_is_in_the_closed_set(dominance_run):
    _, _, labels = dominance_run
    species_or_reserved = NON_SPECIES_LABELS
    for name, label in labels.items():
        assert label is not None, name
        # Either a reserved outcome or a species slug — never anything else.
        assert label in species_or_reserved or label.replace("_", "").isalnum(), (name, label)


def test_lopsided_ground_truth_mostly_not_multiple(dominance_run, capsys):
    """``high`` images (GT ratio > threshold_high) should mostly have a dominant animal."""
    frozen, fetched, labels = dominance_run
    considered = [e for e in frozen["high"] if fetched.get(e["file_name"]) == "high"]
    if not considered:
        pytest.skip("no high-bucket images fetched")

    not_multiple = 0
    for entry in considered:
        label = labels.get(entry["file_name"])
        ok = label != "multiple"
        not_multiple += int(ok)
        print(f"  high gt_ratio={entry['gt_ratio']:.2f} -> {label} {'ok' if ok else 'MULTIPLE'}")

    required = math.ceil(0.8 * len(considered))
    print(f"  high: {not_multiple}/{len(considered)} not multiple (need >= {required})")
    assert not_multiple >= required, (
        f"only {not_multiple}/{len(considered)} lopsided images avoided `multiple`; "
        f"threshold_high={frozen['threshold_high']}. The sanctioned remedy is "
        "re-freezing at a stricter ratio (recorded in PROGRESS.md), never lowering "
        "a ratio in test code."
    )


def test_even_ground_truth_mostly_multiple(dominance_run, capsys):
    """``low`` images (GT ratio < threshold_low) should mostly come out ``multiple``."""
    frozen, fetched, labels = dominance_run
    considered = [e for e in frozen["low"] if fetched.get(e["file_name"]) == "low"]
    if not considered:
        pytest.skip("no low-bucket images fetched")

    multiple = 0
    for entry in considered:
        label = labels.get(entry["file_name"])
        ok = label == "multiple"
        multiple += int(ok)
        print(f"  low  gt_ratio={entry['gt_ratio']:.2f} -> {label} {'ok' if ok else 'NOT-multiple'}")

    required = math.ceil(0.8 * len(considered))
    print(f"  low: {multiple}/{len(considered)} multiple (need >= {required})")
    assert multiple >= required, (
        f"only {multiple}/{len(considered)} even-ratio images came out `multiple`; "
        f"threshold_low={frozen['threshold_low']}"
    )
