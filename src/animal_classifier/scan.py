"""Read-only recursive walk of the source card (DESIGN.md §5.1).

This module is the *only* producer of work for the pipeline, and it is
deliberately incapable of writing: it calls ``os.walk``, ``os.lstat`` and
``os.stat`` and nothing else. Every path it accepts is yielded as a
:class:`Candidate`; every path it refuses is yielded as a :class:`Skipped`
carrying one of the enumerated :class:`SkipReason` values. Nothing is dropped
silently, which is what lets the ``skipped`` table answer "why is this photo not
in my output tree?" for every file on the card.

**One rule per reason.** §5.1 gives each reason exactly one defining rule and
this module implements them in one place, :func:`_judge_file`, in the fixed
precedence order documented there. Two consequences worth stating:

* ``max_file_bytes`` is a whole-file ``st_size`` cap. It is applied here, before
  any decode, so it structurally never sees a detection box and can never act as
  a box-area floor — ``dominance_ratio`` remains the only size gate (invariant
  I2). The decode-time *pixel* cap is the separate reason ``too_large_pixels``
  (§5.2), so the catalog can never confuse a huge file with a huge raster.
* There is no ``symlink_loop``. ``os.walk(followlinks=False)`` never descends a
  symlinked directory, so the reason was unreachable and §5.1 deleted it.

**Benign versus abnormal (design review finding 5).** The run's exit code is not
"did anything get skipped" — a card full of videos and thumbnails is a completely
successful run. :data:`BENIGN_REASONS` are the outcomes that mean *this file was
never ours to process*; :data:`ABNORMAL_REASONS` are the outcomes that mean *we
wanted this file and could not have it*. Only the latter (or a ``failed`` image)
makes the process exit 4 — see :func:`exit_code_for`.

The source root itself is never judged. The user named it explicitly, so a card
directory that happens to be hidden (``/media/.card``) is walked, while a hidden
directory *below* it is pruned.
"""

from __future__ import annotations

import enum
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .config import FAMILY_EXTENSIONS, Config, Format
from .errors import EXIT_OK, EXIT_PARTIAL, ConfigError


class SkipReason(enum.StrEnum):
    """The enumerated ``skipped.reason`` value set (§5.1, plus §5.2's two).

    The last two are produced by :mod:`animal_classifier.images` at decode time,
    not by the walk, but they live here because they are written to the same
    table and classified by the same benign/abnormal split.
    """

    HIDDEN = "hidden"
    SYSTEM_DIR = "system_dir"
    UNSUPPORTED_EXTENSION = "unsupported_extension"
    FORMAT_DISABLED = "format_disabled"
    VIDEO = "video"
    RAW_NOT_ENABLED = "raw_not_enabled"
    ZERO_BYTES = "zero_bytes"
    UNREADABLE = "unreadable"
    TOO_LARGE = "too_large"
    SYMLINK = "symlink"
    SYMLINK_ESCAPE = "symlink_escape"
    #: Decode-time (§5.2): more than 400 MP of pixels.
    TOO_LARGE_PIXELS = "too_large_pixels"
    #: Decode-time (§5.2): truncated or corrupt image data.
    DECODE_ERROR = "decode_error"


#: Outcomes that leave the run successful (exit 0). These say the file was never
#: eligible: it is not an image we were asked for, or it is not an image at all.
BENIGN_REASONS: Final[frozenset[SkipReason]] = frozenset(
    {
        SkipReason.HIDDEN,
        SkipReason.SYSTEM_DIR,
        SkipReason.UNSUPPORTED_EXTENSION,
        SkipReason.FORMAT_DISABLED,
        SkipReason.VIDEO,
        SkipReason.RAW_NOT_ENABLED,
        SkipReason.TOO_LARGE,
        SkipReason.SYMLINK,
    }
)

#: Outcomes that make the run partial (exit 4). Each one is an *eligible* image
#: we could not process: unreadable, empty, escaping the card, or undecodable.
ABNORMAL_REASONS: Final[frozenset[SkipReason]] = frozenset(
    {
        SkipReason.ZERO_BYTES,
        SkipReason.UNREADABLE,
        SkipReason.SYMLINK_ESCAPE,
        SkipReason.TOO_LARGE_PIXELS,
        SkipReason.DECODE_ERROR,
    }
)

# Structural, not decorative: a reason added to the enum without being classified
# would silently inherit "benign" at the first `not in ABNORMAL_REASONS` test and
# quietly turn a partial run into a successful one. Fail at import instead.
_unclassified = set(SkipReason) - BENIGN_REASONS - ABNORMAL_REASONS
if _unclassified or (BENIGN_REASONS & ABNORMAL_REASONS):
    raise AssertionError(  # pragma: no cover - import-time invariant
        "every SkipReason must be exactly one of benign or abnormal; "
        f"unclassified={sorted(_unclassified)} "
        f"overlapping={sorted(BENIGN_REASONS & ABNORMAL_REASONS)}"
    )
del _unclassified

#: Always skipped, per §5.1 — this tool does not look inside video.
VIDEO_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".mp4", ".mov", ".avi", ".m4v", ".mts", ".mpg", ".3gp"}
)

#: RAW is not a ``formats`` family (§3.1): it is governed solely by ``--raw``.
RAW_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2"}
)

#: Extension → family, for every family that exists. Membership here is what
#: separates ``format_disabled`` (known family, not requested) from
#: ``unsupported_extension`` (belongs to no family at all).
EXTENSION_FAMILY: Final[dict[str, Format]] = {
    ext: family for family, exts in FAMILY_EXTENSIONS.items() for ext in exts
}

#: Directory names whose contents are camera/OS bookkeeping, never photographs.
#: ``.Trash`` is a *prefix* because the real name varies (``.Trashes``,
#: ``.Trash-1000``); the other three are exact.
SYSTEM_DIR_PREFIXES: Final[tuple[str, ...]] = (".Trash",)
SYSTEM_DIR_NAMES: Final[frozenset[str]] = frozenset(
    {".thumbnails", ".Spotlight-V100", "__MACOSX"}
)


@dataclass(frozen=True, slots=True)
class Candidate:
    """An eligible file, with the two facts the walk already paid for.

    ``mtime_ns`` is §5.1's ``mtime`` in nanoseconds because that is exactly what
    ``sources.mtime_ns`` stores and what E1's immutability snapshot compares; a
    float seconds value would lose resolution on the way to the catalog.
    """

    path: Path
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class Skipped:
    """A refused path and the single rule that refused it.

    ``detail`` is the human sentence the ``skipped`` table stores next to the
    reason — the actual size for ``too_large``, the escaping target for
    ``symlink_escape``, the OS error for ``unreadable``. It is ``None`` when the
    reason is already the whole story (``video``, ``hidden``).
    """

    path: Path
    reason: SkipReason
    detail: str | None = None

    @property
    def benign(self) -> bool:
        return self.reason in BENIGN_REASONS


def is_benign(reason: SkipReason | str) -> bool:
    """Whether ``reason`` leaves the run successful.

    Accepts the raw string form so a reason read back out of the catalog can be
    classified. An unrecognised value raises ``ValueError`` from
    :class:`SkipReason` rather than defaulting to benign, because guessing here
    would turn a real failure into exit 0.
    """
    return SkipReason(reason) in BENIGN_REASONS


def exit_code_for(
    reasons: Iterable[SkipReason | str], *, n_failed: int = 0
) -> int:
    """The run's exit class: 0 if only benign skips happened, else 4.

    Design review finding 5 in one function. ``n_failed`` carries the per-image
    failures that are not skips at all (a decode that raised, a catalog write
    that lost the lock); any of those is abnormal by definition.
    """
    if n_failed > 0:
        return EXIT_PARTIAL
    if any(not is_benign(reason) for reason in reasons):
        return EXIT_PARTIAL
    return EXIT_OK


def scan(config: Config) -> Iterator[Candidate | Skipped]:
    """Walk ``config.source_root`` and judge every entry below it.

    Fails loudly when no source root is configured rather than substituting the
    working directory (invariant I7): a walk of the wrong tree would be silent
    and wrong, and the caller always knows which card it meant.
    """
    if config.source_root is None:
        raise ConfigError(
            "scan requires a source directory; pass it as the SOURCE argument to "
            "`animal-classifier classify SOURCE`"
        )
    return walk(
        config.source_root,
        extensions=config.eligible_extensions,
        max_file_bytes=config.max_file_bytes,
        raw=config.raw,
        follow_source_symlinks=config.follow_source_symlinks,
    )


def walk(
    source_root: Path,
    *,
    extensions: frozenset[str],
    max_file_bytes: int,
    raw: bool,
    follow_source_symlinks: bool,
) -> Iterator[Candidate | Skipped]:
    """The walk itself: deterministic, read-only, never descending a symlink.

    ``os.walk(followlinks=False)`` plus in-place sorting of both lists gives a
    total, reproducible order, so two runs over the same card visit the same
    files in the same sequence and ``--limit`` means something stable.

    Pruned directories are yielded as a single :class:`Skipped` for the directory
    itself instead of one row per file inside it: the reason is a property of the
    directory, and expanding ``.Trashes`` into a thousand identical rows would
    bury the card's real skips.
    """
    root = source_root.resolve()
    if not root.is_dir():
        raise ConfigError(
            f"source {source_root} is not a directory (resolved to {root}); "
            "point classify at the card's mount point"
        )
    pending_errors: list[Skipped] = []

    def _on_error(error: OSError) -> None:
        # os.walk swallows scandir/listdir failures unless we ask for them. A
        # directory we cannot list is an abnormal outcome: photos may be in it.
        pending_errors.append(
            Skipped(
                Path(error.filename or root),
                SkipReason.UNREADABLE,
                f"{type(error).__name__}: {error.strerror or error}",
            )
        )

    for dirpath, dirnames, filenames in os.walk(
        root, onerror=_on_error, followlinks=False
    ):
        here = Path(dirpath)
        dirnames.sort()
        filenames.sort()

        # Drain first: os.walk reports a directory it could not list through
        # `onerror` at the moment it tries to enter it, i.e. before it yields
        # anything further, so draining here keeps the stream in walk order.
        while pending_errors:
            yield pending_errors.pop(0)
        yield from _prune_directories(here, dirnames)

        for name in filenames:
            yield _judge_file(
                here / name,
                root=root,
                extensions=extensions,
                max_file_bytes=max_file_bytes,
                raw=raw,
                follow_source_symlinks=follow_source_symlinks,
            )

    while pending_errors:
        yield pending_errors.pop(0)


def _prune_directories(parent: Path, dirnames: list[str]) -> Iterator[Skipped]:
    """Remove and report the subdirectories we refuse to descend.

    Mutating ``dirnames`` in place is how ``os.walk`` is told not to descend, so
    the contents of a pruned directory are never even listed — the cheapest
    possible way to honour the read-only guarantee.

    Precedence: ``system_dir`` is tested before ``hidden`` because three of the
    four system names also start with a dot, and the specific rule is the more
    useful record. Symlinked directories are reported as ``symlink`` and are
    never descended **under any flag** (§5.1), including
    ``--follow-source-symlinks``, which governs symlinked *files* only.
    """
    keep: list[str] = []
    for name in dirnames:
        path = parent / name
        if _is_system_dir(name):
            yield Skipped(path, SkipReason.SYSTEM_DIR, f"{name} is camera/OS bookkeeping")
        elif name.startswith("."):
            yield Skipped(path, SkipReason.HIDDEN)
        elif path.is_symlink():
            yield Skipped(
                path,
                SkipReason.SYMLINK,
                "symlinked directory; never descended, in any mode",
            )
        else:
            keep.append(name)
    dirnames[:] = keep


def _is_system_dir(name: str) -> bool:
    return name in SYSTEM_DIR_NAMES or name.startswith(SYSTEM_DIR_PREFIXES)


def _judge_file(
    path: Path,
    *,
    root: Path,
    extensions: frozenset[str],
    max_file_bytes: int,
    raw: bool,
    follow_source_symlinks: bool,
) -> Candidate | Skipped:
    """Apply §5.1's rules to one file, in §5.1's order, and return the verdict.

    The order is the table's order, and it is chosen so that the cheapest and
    most specific rules run first: nothing is stat-ed until the extension says we
    might want it, and a file over the byte cap is refused from its ``st_size``
    alone, so its bytes are never read.
    """
    name = path.name
    if name.startswith("."):
        return Skipped(path, SkipReason.HIDDEN)

    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return Skipped(path, SkipReason.VIDEO)
    if suffix in RAW_EXTENSIONS and not raw:
        return Skipped(path, SkipReason.RAW_NOT_ENABLED, "pass --raw to include RAW files")
    if suffix not in RAW_EXTENSIONS:
        family = EXTENSION_FAMILY.get(suffix)
        if family is None:
            return Skipped(path, SkipReason.UNSUPPORTED_EXTENSION, f"extension {suffix!r}")
        if suffix not in extensions:
            return Skipped(
                path,
                SkipReason.FORMAT_DISABLED,
                f"family {family} is not in the requested formats",
            )

    if path.is_symlink():
        if not follow_source_symlinks:
            return Skipped(
                path,
                SkipReason.SYMLINK,
                "pass --follow-source-symlinks to ingest symlinked files",
            )
        target = Path(os.path.realpath(path))
        if not target.is_relative_to(root):
            # Ingesting this would record a `sources.path` inside the card for
            # bytes that live outside it, so provenance would lie and the
            # read-only guarantee would no longer be scoped to the card.
            return Skipped(
                path,
                SkipReason.SYMLINK_ESCAPE,
                f"target {target} is outside the source root {root}",
            )

    try:
        stat = path.stat()
    except OSError as error:
        return Skipped(
            path,
            SkipReason.UNREADABLE,
            f"{type(error).__name__}: {error.strerror or error}",
        )

    if stat.st_size == 0:
        return Skipped(path, SkipReason.ZERO_BYTES)
    if stat.st_size > max_file_bytes:
        return Skipped(
            path,
            SkipReason.TOO_LARGE,
            f"{stat.st_size} bytes exceeds max_file_bytes={max_file_bytes}",
        )

    return Candidate(path, stat.st_size, stat.st_mtime_ns)
