"""E15: two different images that share a filename both survive (§5.8).

Two photos on the card can legitimately have the same basename in different
directories (``100CANON/IMG_0001.jpg`` and ``101CANON/IMG_0001.jpg``) and receive
the same label. They must not overwrite each other: the second is filed as
``<stem>-<sha256[:8]><suffix>``, and both files end up present with their own
content. The hash suffix is the content hash, so the name is deterministic and a
re-run recognises it as ``already_present`` rather than making a third.
"""

from __future__ import annotations

import json

import pytest


def _classify(run_cli, card, output):
    return run_cli(
        ["classify", str(card), "--output", str(output), "--detector", "scripted"]
    )


@pytest.fixture
def name_collision_card(tmp_path_factory):
    """Two different animal photos sharing the basename ``IMG_0001.jpg``."""
    from conftest import make_e2e_fixtures

    root = tmp_path_factory.mktemp("collide")
    boxes = json.dumps([{"cls": "animal", "conf": 0.9, "x0": 50, "y0": 50, "x1": 450, "y1": 400}])
    for i, sub in enumerate(("100CANON", "101CANON")):
        d = root / "DCIM" / sub
        d.mkdir(parents=True)
        make_e2e_fixtures.noisy(make_e2e_fixtures.FRAME, 200 + i).save(d / "IMG_0001.jpg", quality=95)
        (d / "IMG_0001.jpg.boxes.json").write_text(boxes)
    return root


def test_same_name_different_content_both_survive(run_cli, name_collision_card, tmp_path):
    output = tmp_path / "pics"
    _classify(run_cli, name_collision_card, output)

    # Both are dominant single animals -> unknown/ (no species model yet).
    filed = sorted(
        p.name for p in (output / "unknown").iterdir() if p.is_file()
    )
    assert len(filed) == 2, f"both images filed, got {filed}"
    assert "IMG_0001.jpg" in filed
    suffixed = [n for n in filed if n != "IMG_0001.jpg"]
    assert len(suffixed) == 1
    # The suffix is 8 hex chars of the content hash.
    stem_suffixed = suffixed[0]
    assert stem_suffixed.startswith("IMG_0001-") and stem_suffixed.endswith(".jpg")
    hash8 = stem_suffixed[len("IMG_0001-"):-len(".jpg")]
    assert len(hash8) == 8 and all(ch in "0123456789abcdef" for ch in hash8)

    # Contents differ.
    a = (output / "unknown" / "IMG_0001.jpg").read_bytes()
    b = (output / "unknown" / stem_suffixed).read_bytes()
    assert a != b


def test_rerun_recognises_the_suffixed_name(run_cli, name_collision_card, tmp_path):
    """The hash-suffixed name is content-addressed, so a second run adds nothing."""
    output = tmp_path / "pics"
    _classify(run_cli, name_collision_card, output)
    before = sorted(p.name for p in (output / "unknown").iterdir())
    _classify(run_cli, name_collision_card, output)
    after = sorted(p.name for p in (output / "unknown").iterdir())
    assert after == before, "no third name was invented on the second run"
