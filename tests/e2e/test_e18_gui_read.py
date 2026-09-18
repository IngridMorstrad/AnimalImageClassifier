"""E18: the GUI read paths — labels, images, byte routes, unscored (§6).

Drives the app in-process with a ``TestClient`` over a real ``classify`` run.
Pins the parts §6 is most explicit about:

- **two numbers per label**: catalog ``count`` and filesystem ``files_on_disk``,
  including the ``--dry-run`` leg where every label has ``files_on_disk == 0`` and a
  non-zero ``count``;
- **unscored rows are never silently dropped**: a confidence filter matches
  NULL-confidence rows only with ``include_unscored``, and every response carries
  ``unscored_excluded``;
- **byte routes are 409, never a crash, when there is no readable file**: the
  ``planned`` (dry-run) case and the dangling-symlink case;
- **byte-serving safety**: a non-hex sha is 422, an unknown hash is 404.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import build_card, classify, make_client  # noqa: E402


@pytest.fixture
def classified_output(cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    return output


def test_labels_report_catalog_count_and_files_on_disk(classified_output):
    client = make_client(classified_output)
    labels = {row["label"]: row for row in client.get("/api/labels").json()["labels"]}
    # unknown (the animal), landscape (scene), junk (blurry).
    assert set(labels) == {"unknown", "landscape", "junk"}
    for row in labels.values():
        assert row["count"] == 1
        assert row["files_on_disk"] == 1, "one real run: catalog and disk agree"


def test_dry_run_has_count_but_zero_files_on_disk(cli_path, tmp_path):
    """§6/E17: a --dry-run catalog shows counts with files_on_disk == 0."""
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output, "--dry-run")
    client = make_client(output)
    labels = client.get("/api/labels").json()["labels"]
    assert labels, "planned rows still populate the sidebar"
    for row in labels:
        assert row["count"] >= 1
        assert row["files_on_disk"] == 0, "dry-run wrote no files, so disk is empty"


def test_images_lists_boxes_and_candidates(classified_output):
    client = make_client(classified_output)
    data = client.get("/api/images").json()
    assert data["total"] == 3
    animal = [i for i in data["items"] if i["label"] == "unknown"][0]
    assert animal["boxes"], "the animal image carries its box"
    assert animal["boxes"][0]["cls"] == "animal"


def test_confidence_filter_excludes_unscored_unless_toggled(classified_output):
    client = make_client(classified_output)
    # All three images here are unscored (unknown box-less, landscape, junk).
    filtered = client.get("/api/images?min_conf=0.5").json()
    assert filtered["unscored_excluded"] == 3
    assert filtered["total"] == 0, "unscored rows hidden by the confidence filter"

    shown = client.get("/api/images?min_conf=0.5&include_unscored=true").json()
    assert shown["total"] == 3, "the toggle brings them back"
    assert shown["unscored_excluded"] == 0


def test_byte_routes_serve_and_guard(classified_output):
    client = make_client(classified_output)
    sha = client.get("/api/images").json()["items"][0]["sha256"]
    assert client.get(f"/api/images/{sha}/thumb").status_code == 200
    assert client.get(f"/api/images/{sha}/full").status_code == 200
    assert client.get("/api/images/not-hex/thumb").status_code == 422
    assert client.get(f"/api/images/{'0' * 64}/full").status_code == 404


def test_planned_image_full_is_409(cli_path, tmp_path):
    """§6: a --dry-run (planned) image has no file → 409 with reason 'planned'."""
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output, "--dry-run")
    client = make_client(output)
    sha = client.get("/api/images").json()["items"][0]["sha256"]
    resp = client.get(f"/api/images/{sha}/full")
    assert resp.status_code == 409
    assert resp.json()["detail"]["reason"] == "planned"


def test_dangling_symlink_full_is_409(cli_path, tmp_path):
    """§6: a --link destination whose target is gone → 409 'dangling_symlink'."""
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output, "--link")
    # Remove the card so every symlink dangles (the normal reviewing state).
    import shutil  # noqa: PLC0415

    shutil.rmtree(card)
    client = make_client(output)
    sha = client.get("/api/images").json()["items"][0]["sha256"]
    resp = client.get(f"/api/images/{sha}/full")
    assert resp.status_code == 409
    assert resp.json()["detail"]["reason"] == "dangling_symlink"
