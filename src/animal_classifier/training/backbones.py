"""Pretrained backbone weights, fetched and verified on first use (§7.3).

``train`` without a backbone silently trains from random initialisation. On the
few dozen photographs a human actually labels in a review session that produces a
model which is confidently wrong — a bad outcome you cannot see from the log, only
from the accuracy. Making the user hunt down a URL to avoid that was a footgun, so
the known backbones are fetched automatically, verified against a pinned byte length
and sha256, and cached under ``models/backbones/``.

Unlike the detector checkpoint, these are **pure state dicts**, so they load with
``weights_only=True`` and cannot execute code on load. The hash pin is still enforced:
a mismatched download is discarded, never trained from.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from ..errors import AssetError

log = logging.getLogger(__name__)

_CHUNK: Final = 1 << 20


@dataclass(frozen=True, slots=True)
class Backbone:
    """A pinned pretrained checkpoint for one architecture."""

    arch: str
    filename: str
    url: str
    sha256: str
    size_bytes: int


#: The backbones this build knows how to fetch. Verified by downloading each and
#: recording its real size and sha256 — not copied from a README.
KNOWN_BACKBONES: Final[dict[str, Backbone]] = {
    "efficientnet_b0": Backbone(
        arch="efficientnet_b0",
        filename="efficientnet_b0_ra-3dd342df.pth",
        url=(
            "https://github.com/rwightman/pytorch-image-models/releases/download/"
            "v0.1-weights/efficientnet_b0_ra-3dd342df.pth"
        ),
        sha256="3dd342dfa1fee25ae65e7bbdf8998cad6e45d6e77e69d580f0bd14d3eeb0b3f3",
        size_bytes=21_376_743,
    ),
}

#: Architectures that need no pretrained weights: the from-scratch smoke net (§7.6).
NO_BACKBONE_NEEDED: Final = frozenset({"tinycnn"})


def default_backbone_dir() -> Path:
    return Path("models") / "backbones"


def ensure_backbone(
    arch: str,
    *,
    explicit: Path | None = None,
    allow_download: bool = True,
) -> Path | None:
    """The backbone to train ``arch`` from, fetching it if needed.

    Returns ``None`` only when the architecture genuinely needs no pretrained weights
    (``tinycnn``). An explicit ``--backbone-weights`` always wins and must exist —
    the user named a specific file, so its absence is a mistake, not a download.
    """
    if explicit is not None:
        if not explicit.is_file():
            raise AssetError(
                f"--backbone-weights {explicit} does not exist. Omit the flag to let "
                "the matching backbone be downloaded automatically."
            )
        return explicit

    if arch in NO_BACKBONE_NEEDED:
        return None

    known = KNOWN_BACKBONES.get(arch)
    if known is None:
        log.warning(
            "no pinned backbone is known for --arch %s, so it will train from random "
            "initialisation. On a small hand-labelled set that usually gives a poor "
            "model; pass --backbone-weights <file> with pretrained weights for %s.",
            arch,
            arch,
        )
        return None

    destination = default_backbone_dir() / known.filename
    if destination.is_file():
        _verify(destination, known)
        return destination

    if not allow_download:
        raise AssetError(
            f"backbone weights for {arch} are not present at {destination} and "
            f"--no-download was given. Fetch them with:\n  curl -L --create-dirs -o "
            f"{destination} {known.url}"
        )

    log.info(
        "downloading pretrained %s weights (%.0f MB) to %s. This happens once and "
        "makes a large difference on a small training set.",
        arch,
        known.size_bytes / 1_000_000,
        destination,
    )
    _download(known, destination)
    _verify(destination, known)
    return destination


def _download(known: Backbone, destination: Path) -> None:
    """Stream to ``<destination>.part``, verify, then rename — never trust a partial."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    written = 0
    try:
        with urllib.request.urlopen(known.url, timeout=60) as response:
            with partial.open("wb") as handle:
                while chunk := response.read(_CHUNK):
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
    except (urllib.error.URLError, OSError, TimeoutError) as error:
        _remove_quietly(partial)
        raise AssetError(
            f"could not download the {known.arch} backbone from {known.url}: "
            f"{type(error).__name__}: {error}. Fetch it manually with:\n"
            f"  curl -L --create-dirs -o {destination} {known.url}"
        ) from error

    if written != known.size_bytes or digest.hexdigest() != known.sha256:
        _remove_quietly(partial)
        raise AssetError(
            f"the downloaded {known.arch} backbone does not match its pinned "
            f"checkpoint (got {written} bytes / sha256 {digest.hexdigest()}, expected "
            f"{known.size_bytes} / {known.sha256}). The download was discarded."
        )
    os.replace(partial, destination)


def _verify(path: Path, known: Backbone) -> None:
    size = path.stat().st_size
    if size != known.size_bytes:
        raise AssetError(
            f"backbone at {path} is {size} bytes, expected {known.size_bytes}; it is "
            f"truncated or not {known.filename}. Delete it and re-run to re-fetch."
        )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    if digest.hexdigest() != known.sha256:
        raise AssetError(
            f"backbone at {path} has sha256 {digest.hexdigest()}, expected "
            f"{known.sha256}. Delete it and re-run to re-fetch."
        )


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as error:  # pragma: no cover - best-effort cleanup
        log.debug("could not remove %s: %s", path, error)
