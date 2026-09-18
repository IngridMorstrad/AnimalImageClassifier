"""Filing an image into the output tree, atomically (DESIGN.md §5.8).

Destination: ``<output_root>/<label>/<original filename>``. Three modes — copy
(default), ``--link`` (symlink), ``--hardlink`` — and all three land the same way:
create a fresh ``.ac-tmp-*`` entry **in the destination directory**, then
``os.replace`` it into place. The temp lives in the destination directory rather
than ``/tmp`` for one reason: ``os.replace`` is only atomic within a filesystem,
so creating the temp beside its final name is what makes the rename a true atomic
swap instead of a copy-and-delete that can be interrupted half way.

**The no-write rule, which is what makes invariant I1 true in ``--hardlink``
mode.** In hardlink mode a file under ``~/animal_pics`` *is* the card's inode — E3
asserts equal ``st_ino`` — so a path-based "never write to the source" guard would
not stop an in-place write through the destination path from modifying the SD
card. That hole is closed structurally here: **no function in this module opens a
materialized file for writing, truncates one, ``copystat``s onto one, or edits one
in place. The only operations performed on an *existing* destination are
``os.replace`` and ``os.unlink``**, both of which touch the directory entry only
and leave the inode's bytes alone. New content is always a brand-new temp inode
that gets renamed in. ``shutil.copystat`` is called on the temp we just created
and own, never on a destination.

**Collisions never guess.** A name already taken is resolved in a fixed order
(:func:`resolve_destination`): identical content means the work is already done, so
nothing is written and it counts as ``already_present`` — that is what makes
re-runs cheap. Differing content goes to ``<stem>-<sha256[:8]><suffix>``. If *that*
name is also taken by different content — effectively impossible, since the suffix
is the content hash — it is a fatal :class:`MaterializeError` rather than a guessed
third name.

**Link mode never dereferences the destination.** Comparing a symlink by hashing
what it points at is exactly wrong for this tool: the common case is reviewing a
card that has since been unplugged, so the target is gone and hashing it would
raise on the one path that must not. ``os.readlink`` is compared against the
intended absolute source instead — equal means already filed, different or
dangling means the link is atomically replaced and counted ``relinked``. Only a
*regular file* at a destination is ever hashed.

**``--dry-run`` writes nothing at all.** Not the label directory, not a temp, not
the file. It computes the destination the run *would* use and hands it back for the
catalog to record with ``status='planned'`` (§5.9), so a plan can be inspected
before a single byte moves.
"""

from __future__ import annotations

import enum
import errno
import logging
import os
import shutil
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .config import Mode
from .errors import EXIT_CONFIG, ConfigError, MaterializeError
from .images import SHA256_CHUNK_BYTES, open_source, sha256_file
from .taxonomy import LABEL_RE

log = logging.getLogger(__name__)

#: Prefix for every temp entry this module creates. Also the sweep pattern: a
#: killed process leaves these behind, and §5.8 requires the next run to clear
#: them. Leading dot so §5.8's label-directory regex never matches one, which is
#: what keeps a stale temp out of label enumeration, the GUI and `verify`.
TEMP_PREFIX: Final = ".ac-tmp-"

#: Directory mode for created label directories (§5.8).
LABEL_DIR_MODE: Final = 0o755

#: How many hex characters of the content hash disambiguate a colliding name.
COLLISION_HASH_CHARS: Final = 8


class Outcome(enum.StrEnum):
    """What materializing one image actually did.

    These are the words §5.8 and §10.1 use for the run's counters, so the names
    are part of the contract rather than an internal detail.
    """

    #: A new entry was created at a name that was free.
    WRITTEN = "written"
    #: The destination already held this exact content (or, in link mode, this
    #: exact link target). Nothing was written — this is what makes a re-run cheap.
    ALREADY_PRESENT = "already_present"
    #: An existing entry was atomically replaced: a symlink that pointed somewhere
    #: else or dangled, or a link left by a previous run in a different mode.
    RELINKED = "relinked"
    #: ``--dry-run``: the destination was computed and nothing was touched.
    PLANNED = "planned"


@dataclass(frozen=True, slots=True)
class Materialized:
    """Where an image was filed, and what filing it cost."""

    dest_path: Path
    outcome: Outcome
    mode: Mode

    @property
    def wrote_something(self) -> bool:
        """Whether this call changed the output tree."""
        return self.outcome in {Outcome.WRITTEN, Outcome.RELINKED}


def require_under_output_root(path: Path, output_root: Path) -> Path:
    """Return ``path`` resolved, having proved it is under ``output_root``.

    The final component is deliberately **not** resolved: a destination that is a
    symlink must be judged as the path it is, not as whatever it points at, or a
    link into the card would pass this check and then be written through. The
    parent is resolved (so ``..`` and symlinked ancestors cannot smuggle a path
    out), and the name is re-attached afterwards.

    Fatal, not a warning: writing outside the output root is the failure this whole
    guard exists to prevent, and there is no safe way to continue past it.
    """
    root = output_root.resolve()
    resolved = path.parent.resolve() / path.name
    if not resolved.is_relative_to(root):
        raise MaterializeError(
            f"refusing to materialize to {resolved}, which is outside "
            f"output_root {root}"
        )
    return resolved


def plan_destination(source: Path, label: str, *, output_root: Path) -> Path:
    """``<output_root>/<label>/<source name>``, with the label validated.

    The label is re-checked against §5.8's ``LABEL_RE`` even though species slugs
    were validated at artifact-load time and the four reserved labels are
    constants. It costs one regex per image and it is the last line of defence
    before a label becomes a directory name: a label containing ``/`` or ``..``
    would otherwise place files outside the label tree.
    """
    if not LABEL_RE.fullmatch(label):
        raise ConfigError(
            f"label {label!r} is not a legal label directory (must match "
            f"{LABEL_RE.pattern}); labels are validated by taxonomy.slug() at "
            "artifact load, so this indicates a label that bypassed it"
        )
    return require_under_output_root(output_root / label / source.name, output_root)


def sweep_stale_temps(output_root: Path) -> int:
    """Remove ``.ac-tmp-*`` entries left by a killed run (§5.8). Returns the count.

    Scans ``output_root`` itself and one level of real subdirectories — label
    directories are flat, so there is nowhere else for a temp to be. Symlinked
    subdirectories are not descended into, because a symlink under the output root
    could point anywhere, including at the card.

    A temp that cannot be removed is logged and left; the sweep is housekeeping and
    must never be the reason a run refuses to start.
    """
    if not output_root.is_dir():
        return 0
    removed = 0
    for directory in (output_root, *_real_subdirectories(output_root)):
        for entry in _temp_entries(directory):
            try:
                entry.unlink()
            except OSError as error:
                log.warning("could not remove stale temp %s: %s", entry, error)
                continue
            log.debug("swept stale temp %s", entry)
            removed += 1
    if removed:
        log.info("swept %d stale %s file(s) from a previous run", removed, TEMP_PREFIX)
    return removed


def _real_subdirectories(root: Path) -> Iterator[Path]:
    """Immediate subdirectories of ``root``, skipping symlinked ones."""
    try:
        entries = list(os.scandir(root))
    except OSError as error:
        log.debug("cannot scan %s for stale temps: %s", root, error)
        return
    for entry in entries:
        if entry.is_dir(follow_symlinks=False):
            yield Path(entry.path)


def _temp_entries(directory: Path) -> Iterator[Path]:
    """``.ac-tmp-*`` entries directly in ``directory``, symlinks included."""
    try:
        entries = list(os.scandir(directory))
    except OSError as error:
        log.debug("cannot scan %s for stale temps: %s", directory, error)
        return
    for entry in entries:
        if entry.name.startswith(TEMP_PREFIX):
            yield Path(entry.path)


def materialize(
    source: Path,
    *,
    label: str,
    sha256: str,
    output_root: Path,
    mode: Mode,
    dry_run: bool = False,
) -> Materialized:
    """File ``source`` under ``<output_root>/<label>/`` and report what happened.

    ``sha256`` is the content hash the scanner already computed. It is passed in
    rather than recomputed because it is both the collision-suffix source and the
    identity the catalog keys on, and hashing the same file twice for two different
    answers is how those two things drift apart.
    """
    dest = plan_destination(source, label, output_root=output_root)

    if dry_run:
        # Not even mkdir: §5.8 says a dry run performs no destination writes, and a
        # created directory is a write that outlives the run.
        log.debug("dry-run: would file %s as %s", source, dest)
        return Materialized(dest_path=dest, outcome=Outcome.PLANNED, mode=mode)

    absolute_source = source.resolve()
    _ensure_label_dir(dest.parent)

    final, settled = resolve_destination(
        dest,
        sha256=sha256,
        mode=mode,
        absolute_source=absolute_source,
    )
    if settled is not None:
        log.debug("%s already filed at %s (%s)", source, final, settled)
        return Materialized(dest_path=final, outcome=settled, mode=mode)

    existed = _entry_exists(final)
    _place(source, final, mode=mode, absolute_source=absolute_source)
    outcome = Outcome.RELINKED if existed else Outcome.WRITTEN
    log.debug("%s -> %s (%s, %s)", source, final, mode, outcome)
    return Materialized(dest_path=final, outcome=outcome, mode=mode)


def resolve_destination(
    dest: Path,
    *,
    sha256: str,
    mode: Mode,
    absolute_source: Path,
) -> tuple[Path, Outcome | None]:
    """Apply §5.8's collision rules. Returns ``(path_to_use, settled_outcome)``.

    ``settled_outcome`` is non-``None`` only when nothing needs to be written — the
    content (or link target) is already there. Otherwise it is ``None`` and the
    caller writes to the returned path, which is either ``dest`` itself or the
    hash-suffixed variant.
    """
    if not _entry_exists(dest):
        return dest, None

    if _is_already_filed(dest, sha256=sha256, mode=mode, absolute_source=absolute_source):
        return dest, Outcome.ALREADY_PRESENT

    if mode is Mode.LINK and _is_symlink(dest):
        # A link that points elsewhere or dangles: replace it in place rather than
        # inventing a second name for the same image (§5.8, counted `relinked`).
        return dest, None

    if mode is not Mode.LINK and _is_symlink(dest):
        # A symlink left by an earlier `--link` run, now being re-filed as a real
        # copy or hardlink. Beyond §5.8's literal text, which assumes one mode
        # throughout; replacing converges on the mode the user asked for, and it is
        # done without ever dereferencing the old link (whose target may be gone).
        log.debug(
            "%s is a symlink from a previous --link run; replacing it in %s mode",
            dest,
            mode,
        )
        return dest, None

    suffixed = dest.with_name(
        f"{dest.stem}-{sha256[:COLLISION_HASH_CHARS]}{dest.suffix}"
    )
    if not _entry_exists(suffixed):
        log.debug("%s taken by different content; using %s", dest, suffixed.name)
        return suffixed, None
    if _is_already_filed(
        suffixed, sha256=sha256, mode=mode, absolute_source=absolute_source
    ):
        return suffixed, Outcome.ALREADY_PRESENT

    raise MaterializeError(
        f"cannot file {absolute_source}: {dest} holds different content and the "
        f"hash-suffixed name {suffixed} is also taken by content that is neither. "
        "Refusing to invent a third name; move or remove one of those files."
    )


def _is_already_filed(
    dest: Path, *, sha256: str, mode: Mode, absolute_source: Path
) -> bool:
    """Whether ``dest`` already holds exactly what we were about to put there.

    Link mode compares ``os.readlink`` with the intended absolute source and never
    dereferences, so a link whose target is gone answers ``False`` instead of
    raising. Every other mode hashes the destination, but **only when it is a
    regular file** — that is the guard that keeps a symlink from being followed off
    a disconnected card.
    """
    if mode is Mode.LINK:
        if not _is_symlink(dest):
            # A regular file where a link was expected: comparable by content.
            return _is_regular_file(dest) and _content_matches(dest, sha256)
        try:
            return Path(os.readlink(dest)) == absolute_source
        except OSError as error:
            log.debug("cannot readlink %s (%s); treating as not-already-filed", dest, error)
            return False

    if not _is_regular_file(dest):
        return False
    return _content_matches(dest, sha256)


def _content_matches(dest: Path, sha256: str) -> bool:
    """Whether the regular file at ``dest`` hashes to ``sha256``.

    Reads through :func:`animal_classifier.images.open_source`, which opens
    ``"rb"`` — a materialized file is never opened for writing, not even to
    compare it.
    """
    try:
        return sha256_file(dest) == sha256
    except OSError as error:
        log.debug("cannot hash %s (%s); treating as different content", dest, error)
        return False
    except Exception as error:  # ImageDecodeError wraps the OSError from open_source
        log.debug("cannot hash %s (%s); treating as different content", dest, error)
        return False


def _place(source: Path, dest: Path, *, mode: Mode, absolute_source: Path) -> None:
    """Create ``dest`` for ``source`` atomically, in ``mode``.

    One temp entry in ``dest``'s own directory, then ``os.replace``. The temp is
    removed in a ``finally`` on every failure path, so an interrupted run never
    leaves a partial image at a real destination — the worst it leaves is a
    ``.ac-tmp-*`` that the next run sweeps.
    """
    temp: Path | None = None
    try:
        if mode is Mode.COPY:
            temp = _copy_to_temp(source, dest.parent)
        elif mode is Mode.LINK:
            temp = _link_to_temp(absolute_source, dest.parent, hard=False)
        elif mode is Mode.HARDLINK:
            temp = _link_to_temp(source, dest.parent, hard=True)
        else:  # pragma: no cover - Mode is a closed enum
            raise MaterializeError(f"unknown materialize mode {mode!r}")
        os.replace(temp, dest)
        temp = None
    finally:
        if temp is not None:
            with _suppressed_unlink(temp):
                temp.unlink()


def _copy_to_temp(source: Path, dest_dir: Path) -> Path:
    """Stream ``source`` into a new temp in ``dest_dir``; preserve its metadata.

    ``copystat`` is applied to the **temp**, which this function just created and
    owns, so copy mode preserves mtime (§5.8) without ever touching an existing
    destination's inode.
    """
    handle_fd, temp_name = tempfile.mkstemp(dir=dest_dir, prefix=TEMP_PREFIX)
    temp = Path(temp_name)
    try:
        with open_source(source) as reader, os.fdopen(handle_fd, "wb") as writer:
            shutil.copyfileobj(reader, writer, SHA256_CHUNK_BYTES)
        shutil.copystat(source, temp)
    except OSError as error:
        with _suppressed_unlink(temp):
            temp.unlink()
        raise _materialize_os_error(error, source=source, dest_dir=dest_dir) from error
    except BaseException:
        with _suppressed_unlink(temp):
            temp.unlink()
        raise
    return temp


def _link_to_temp(target: Path, dest_dir: Path, *, hard: bool) -> Path:
    """Create a sym/hard link at a fresh temp name in ``dest_dir``.

    ``mkstemp`` cannot be used here: it creates the file, and both ``os.symlink``
    and ``os.link`` require the name to be free. A random name is generated
    instead, and a collision simply retries — the loop is bounded so a persistent
    failure surfaces rather than spinning.
    """
    for _ in range(8):
        temp = dest_dir / f"{TEMP_PREFIX}{uuid.uuid4().hex}"
        try:
            if hard:
                os.link(target, temp)
            else:
                os.symlink(target, temp)
        except FileExistsError:
            continue
        except OSError as error:
            raise _materialize_os_error(
                error, source=target, dest_dir=dest_dir, hard=hard
            ) from error
        return temp
    raise MaterializeError(
        f"could not create a unique {TEMP_PREFIX}* name in {dest_dir} after 8 tries"
    )


def _materialize_os_error(
    error: OSError, *, source: Path, dest_dir: Path, hard: bool = False
) -> MaterializeError:
    """Translate an ``OSError`` into §10.1's class for it.

    ``EXDEV`` under ``--hardlink`` is the one case that is a *configuration*
    mistake (exit 3) rather than an IO failure (exit 1), because the fix is a
    different flag, not a different disk.
    """
    if hard and error.errno == errno.EXDEV:
        return MaterializeError(
            f"SD card and {dest_dir.parent} are on different filesystems; "
            "hardlinks cannot cross filesystems — use --link for symlinks or omit "
            "the flag to copy",
            exit_code=EXIT_CONFIG,
        )
    if error.errno == errno.ENOSPC:
        return MaterializeError(
            f"out of space writing {source.name} into {dest_dir}: "
            f"{error.strerror or error}"
        )
    return MaterializeError(
        f"cannot file {source} into {dest_dir}: "
        f"{type(error).__name__}: {error.strerror or error}"
    )


def _ensure_label_dir(directory: Path) -> None:
    """Create a label directory, 0o755, parents included (§5.8)."""
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=LABEL_DIR_MODE)
    except OSError as error:
        raise MaterializeError(
            f"cannot create label directory {directory}: "
            f"{type(error).__name__}: {error.strerror or error}"
        ) from error


def _entry_exists(path: Path) -> bool:
    """Whether a directory entry exists at ``path``, dangling symlinks included.

    ``Path.exists()`` follows symlinks and so answers ``False`` for a link whose
    target is gone — precisely the destination this module must notice, because it
    still occupies the name.
    """
    return path.is_symlink() or path.exists()


def _is_symlink(path: Path) -> bool:
    return path.is_symlink()


def _is_regular_file(path: Path) -> bool:
    """A real file, not a symlink to one — the only thing safe to hash."""
    return path.is_file() and not path.is_symlink()


class _suppressed_unlink:
    """Context manager that ignores a failure to remove a temp file.

    A temp that cannot be unlinked is not worth masking the real exception for; the
    next run's :func:`sweep_stale_temps` will clear it.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        if exc_type is not None and issubclass(exc_type, OSError):  # type: ignore[arg-type]
            log.debug("could not remove temp %s: %s", self._path, exc)
            return True
        return False



def retag(
    old_dest: Path,
    *,
    new_label: str,
    sha256: str,
    output_root: Path,
    mode: Mode,
) -> Path:
    """Move an already-filed image to a new label, without opening the source (§5.8).

    A pure output-tree rename: ``os.replace`` moves the directory entry (a regular
    file, a hardlink keeping its inode, or a symlink moved *as a symlink*), so it
    works even when the card is unplugged and a ``--link`` destination dangles. The
    source is never opened — the correction of iteration 2's error where re-tag was
    defined in terms of the three materialize modes that all read the card.

    The GUI records the ``overrides`` row and the intent *before* calling this
    (I4); this performs the one rename, resolving a collision at the new
    destination by §5.8's rules. Returns the new path.
    """
    old = require_under_output_root(old_dest, output_root)
    if not _entry_exists(old):
        raise MaterializeError(
            f"cannot re-tag {sha256[:12]}: its filed path {old} is gone"
        )
    _ensure_label_dir(output_root / new_label)
    dest = plan_destination(old, new_label, output_root=output_root)

    absolute_source = old if old.is_symlink() else old.resolve()
    final, settled = resolve_destination(
        dest, sha256=sha256, mode=mode, absolute_source=absolute_source
    )
    if settled is Outcome.ALREADY_PRESENT:
        # Already filed correctly at the target; drop the old entry (§5.8 re-tag).
        if final != old:
            with _suppressed_unlink(old):
                old.unlink()
        return final
    os.replace(old, final)
    log.debug("re-tagged %s -> %s", old, final)
    return final
