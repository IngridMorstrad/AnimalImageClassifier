"""E3: ``--link`` and ``--hardlink`` file the image without copying bytes (§5.8).

``--link`` writes a symlink whose target is the absolute source path; ``--hardlink``
writes a second directory entry for the source's inode. The observable proofs:

- link mode → the destination ``is_symlink()`` and ``os.readlink`` points at the
  source;
- hardlink mode → the destination shares the source's ``st_ino`` and ``st_nlink``
  is 2 (the card entry plus ours).

Both are the mechanism behind I1: neither opens the source for writing, and the
hardlink's shared inode is exactly why §5.8 forbids ever opening a materialized file
for writing.
"""

from __future__ import annotations

import os


def _classify(run_cli, card, output, *extra):
    return run_cli(
        ["classify", str(card), "--output", str(output), "--detector", "scripted", *extra]
    )


def _one_filed_image(output):
    for path in output.rglob("*"):
        if path.parent.name != output.name and not path.name.startswith(".catalog"):
            if path.is_symlink() or path.is_file():
                return path
    raise AssertionError("no filed image found")


def test_link_mode_writes_symlinks_to_the_source(run_cli, five_image_card, tmp_path):
    output = tmp_path / "pics"
    _classify(run_cli, five_image_card, output, "--link")
    filed = _one_filed_image(output)
    assert filed.is_symlink()
    target = os.readlink(filed)
    assert os.path.isabs(target)
    # The target resolves to a real source file on the card.
    assert os.path.exists(target)
    assert five_image_card in filed.resolve().parents


def test_hardlink_mode_shares_the_inode(run_cli, five_image_card, tmp_path):
    output = tmp_path / "pics"
    _classify(run_cli, five_image_card, output, "--hardlink")
    filed = _one_filed_image(output)
    assert not filed.is_symlink()
    st = filed.stat()
    assert st.st_nlink == 2, "the card entry plus the filed entry"
    # Find the matching source by inode.
    sources = [p for p in five_image_card.rglob("*.jpg")]
    assert any(p.stat().st_ino == st.st_ino for p in sources), "shares a source inode"


def test_copy_mode_is_a_distinct_inode(run_cli, five_image_card, tmp_path):
    """The default: a real copy, so no inode is shared and mtime is preserved."""
    output = tmp_path / "pics"
    _classify(run_cli, five_image_card, output)
    filed = _one_filed_image(output)
    assert not filed.is_symlink()
    assert filed.stat().st_nlink == 1
