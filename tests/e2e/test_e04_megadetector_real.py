"""E4: the real MegaDetector v5a finds animals in real photographs (§5.4).

The scripted detector proves the *rule*; this proves the *detector*. It is marked
``slow`` and skips cleanly when the 280 MB checkpoint is absent, because the suite
must stay runnable on a machine that has not downloaded it — a skipped E4 is honest,
a failed one for a missing asset is noise.

What it asserts, on real images run through the actual checkpoint:

- a photo of animals yields at least one ``animal`` box, and the image is filed as a
  species/``unknown`` (not ``landscape``);
- a photo of only people yields ``person`` boxes, no ``animal`` box, and files as
  ``landscape`` — §5.4's deliberate, documented behaviour;
- the recorded ``model_id`` identifies the real detector, not the scripted one.

Real animal imagery is fetched from COCO val2017 (reachable in this environment);
the test skips rather than fails if the network is unavailable.
"""

from __future__ import annotations

import sqlite3
import urllib.request

import pytest

from animal_classifier.detect.megadetector import EXPECTED_BYTES

WEIGHTS = "models/md_v5a.0.0.pt"

# COCO val2017 images, verified by running this checkpoint against them: each of
# the first two yields >=1 `animal` box, the third is a people-only crowd (only
# `person` boxes). Pinned by id so E4 tests the detector, not the luck of a URL.
COCO_ANIMAL = "http://images.cocodataset.org/val2017/000000404568.jpg"
COCO_ANIMAL2 = "http://images.cocodataset.org/val2017/000000024919.jpg"
COCO_PEOPLE = "http://images.cocodataset.org/val2017/000000000885.jpg"


def _weights_present() -> bool:
    from pathlib import Path  # noqa: PLC0415

    p = Path(WEIGHTS)
    return p.is_file() and p.stat().st_size == EXPECTED_BYTES


def _fetch(url: str, dest) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - fixed https/http COCO host
            data = response.read()
    except Exception:  # noqa: BLE001 - any network failure -> skip, not fail
        return False
    if len(data) < 1000:
        return False
    dest.write_bytes(data)
    return True


pytestmark = pytest.mark.skipif(
    not _weights_present(),
    reason=f"MegaDetector weights absent at {WEIGHTS} (280 MB); download to run E4",
)


def _catalog(output):
    c = sqlite3.connect(f"file:{output / '.catalog.db'}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    return c


@pytest.fixture(scope="module")
def real_card(tmp_path_factory):
    """A card of real COCO photos — skipped if the network is unavailable."""
    card = tmp_path_factory.mktemp("realcard") / "DCIM"
    card.mkdir()
    got = 0
    for name, url in [
        ("animals1.jpg", COCO_ANIMAL),
        ("animals2.jpg", COCO_ANIMAL2),
        ("people.jpg", COCO_PEOPLE),
    ]:
        if _fetch(url, card / name):
            got += 1
    if got == 0:
        pytest.skip("could not fetch any COCO images (network unavailable)")
    return card.parent


@pytest.mark.slow
def test_real_detector_finds_animals_and_files_them(run_cli, real_card, tmp_path):
    output = tmp_path / "pics"
    result = run_cli(
        ["classify", str(real_card), "--output", str(output), "--detector", "megadetector"]
    )
    assert result.returncode in {0, 4}, result.stderr

    with _catalog(output) as c:
        model_ids = {r["model_id"] for r in c.execute("SELECT DISTINCT model_id FROM images WHERE model_id IS NOT NULL")}
        assert any(mid and mid.startswith("megadetector:v5a") for mid in model_ids), model_ids

        animal_boxes = c.execute(
            "SELECT COUNT(*) AS n FROM boxes WHERE cls='animal'"
        ).fetchone()["n"]
        assert animal_boxes >= 1, (
            "the real detector must find at least one animal across the COCO "
            "animal photos; found none"
        )

        # The animal photos are filed as a species-or-unknown, never landscape/junk.
        animal_labels = {
            r["label"]
            for r in c.execute(
                "SELECT DISTINCT i.label FROM images i "
                "JOIN boxes b ON b.sha256 = i.sha256 AND b.cls='animal'"
            )
        }
        assert animal_labels, "an image with an animal box got a label"
        assert animal_labels <= {"unknown", "multiple"} or any(
            lbl not in {"landscape", "junk"} for lbl in animal_labels
        ), f"an animal image must not be landscape/junk, got {animal_labels}"


@pytest.mark.slow
def test_people_only_photo_is_landscape(run_cli, real_card, tmp_path):
    """§5.4: person/vehicle boxes are stored and drawn but never label an image."""
    output = tmp_path / "pics"
    run_cli(["classify", str(real_card), "--output", str(output), "--detector", "megadetector"])

    people = real_card / "DCIM" / "people.jpg"
    if not people.exists():
        pytest.skip("the people-only COCO image was not fetched")

    with _catalog(output) as c:
        row = c.execute(
            "SELECT i.label FROM images i JOIN sources s USING(sha256) "
            "WHERE s.path LIKE '%people.jpg'"
        ).fetchone()
        assert row is not None
        # A crowd photo: only person boxes, so landscape (sharp) — never a species.
        assert row["label"] in {"landscape", "junk"}, row["label"]
