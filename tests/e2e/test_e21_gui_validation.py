"""E21: GUI input validation — every invalid input is refused, nothing changes (§6).

The validation matrix §11.2 specifies. The two that carry real weight:

- **a reserved label is 422 even under ``--allow-new-labels``.** The flag widens
  *which* labels are known; it never relaxes the syntax and never lets a user create a
  second ``unknown/`` that would make the pipeline's own outcome ambiguous.
- **nothing on disk changes on any rejection.** A 422 that had already moved the file
  would be worse than a 500.

Also covers a label that fails ``LABEL_RE`` (``Brewer's Blackbird`` — apostrophes and
spaces are illegal *as a directory*, which is why ``slug()`` exists), a malformed
sha256, an unknown hash, and out-of-range query parameters.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from conftest_gui import build_card, classify, make_client


def _tree(output: Path) -> dict[str, str]:
    """Every filed file and its content hash — the 'nothing changed' witness."""
    return {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file() and not p.name.startswith(".catalog")
    }


@pytest.fixture
def served(cli_path, tmp_path):
    card = build_card(tmp_path / "card")
    output = tmp_path / "pics"
    classify(cli_path, card, output)
    return output


@pytest.mark.parametrize("allow_new", [False, True])
def test_reserved_label_is_422_even_with_allow_new_labels(served, allow_new):
    """§5.8: a user can never create a second `unknown/`."""
    client = make_client(served, allow_new_labels=allow_new)
    sha = client.get("/api/images").json()["items"][0]["sha256"]
    before = _tree(served)
    for reserved in ("unknown", "landscape", "junk", "multiple"):
        resp = client.post(f"/api/images/{sha}/label", json={"label": reserved})
        assert resp.status_code == 422, (reserved, allow_new, resp.text)
    assert _tree(served) == before, "a rejected re-tag must not touch the output tree"


@pytest.mark.parametrize(
    "bad_label",
    ["Brewer's Blackbird", "has space", "UPPER", "-leading-dash", "", "a/b", "../escape"],
)
def test_syntactically_illegal_labels_are_422(served, bad_label):
    """LABEL_RE is enforced regardless of --allow-new-labels."""
    client = make_client(served, allow_new_labels=True)
    sha = client.get("/api/images").json()["items"][0]["sha256"]
    before = _tree(served)
    resp = client.post(f"/api/images/{sha}/label", json={"label": bad_label})
    assert resp.status_code == 422, (bad_label, resp.text)
    assert _tree(served) == before


def test_unknown_label_is_422_without_the_flag_and_200_with_it(served):
    client_strict = make_client(served, allow_new_labels=False)
    sha = client_strict.get("/api/images").json()["items"][0]["sha256"]
    assert client_strict.post(f"/api/images/{sha}/label", json={"label": "pangolin"}).status_code == 422

    client_open = make_client(served, allow_new_labels=True)
    assert client_open.post(f"/api/images/{sha}/label", json={"label": "pangolin"}).status_code == 200


def test_malformed_sha256_is_422(served):
    client = make_client(served, allow_new_labels=True)
    for bad in ("not-hex", "abc", "z" * 64, "0" * 63):
        assert client.get(f"/api/images/{bad}/thumb").status_code == 422, bad
        assert client.get(f"/api/images/{bad}/full").status_code == 422, bad
        assert client.post(f"/api/images/{bad}/label", json={"label": "lion"}).status_code == 422, bad


def test_unknown_hash_is_404(served):
    client = make_client(served, allow_new_labels=True)
    absent = "0" * 64
    assert client.get(f"/api/images/{absent}/thumb").status_code == 404
    assert client.get(f"/api/images/{absent}/full").status_code == 404
    assert client.post(f"/api/images/{absent}/label", json={"label": "lion"}).status_code == 404


def test_out_of_range_limit_is_clamped_not_crashed(served):
    """§6 caps `limit` at 200; a larger request is clamped, not a 500."""
    client = make_client(served)
    for limit in (0, -5, 10_000):
        resp = client.get(f"/api/images?limit={limit}")
        assert resp.status_code == 200, (limit, resp.text)
        assert len(resp.json()["items"]) <= 200


def test_non_bool_include_unscored_is_422(served):
    client = make_client(served)
    resp = client.get("/api/images?include_unscored=banana")
    assert resp.status_code == 422, resp.text
