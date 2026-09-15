"""Exception hierarchy and the process exit-code contract (DESIGN.md §10.1).

Exit codes, used by every entry point and asserted by the e2e suite:

===  =========================================================================
0    success
1    unexpected / IO failure
2    typer usage error (raised by typer itself, never by this module)
3    a missing or invalid *required* asset or configuration value
4    the run completed, but individual images failed or were skipped
===  =========================================================================

The rule this module exists to make structural: **a missing or unexpected
required value fails loudly with an actionable message; it is never replaced by
a default** (invariant I7). Every exception therefore carries the exit code the
process must terminate with, so no caller has to remember the mapping.
"""

from __future__ import annotations

EXIT_OK = 0
EXIT_UNEXPECTED = 1
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_PARTIAL = 4


class AnimalClassifierError(Exception):
    """Base class for every error this tool raises deliberately.

    ``exit_code`` is a class attribute so each subclass declares its default
    failure class once. A single call site may override it when DESIGN.md §10.1
    gives one specific failure a different class — for example a ``--hardlink``
    ``EXDEV`` is a configuration mistake (3) rather than an IO failure (1).
    """

    exit_code: int = EXIT_UNEXPECTED

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class ConfigError(AnimalClassifierError):
    """Invalid, unknown, out-of-range or missing configuration.

    Covers an unknown TOML key, a bad type, a value outside its documented
    range, a ``--config`` path that does not exist, an ``output_root`` nested
    with the source root, a ``formats`` entry outside ``{jpeg,png,tiff,heic}``,
    ``--device cuda`` with no CUDA device, a bird provider selected without its
    credentials, and a label that fails ``slug()``.
    """

    exit_code = EXIT_CONFIG


class AssetError(AnimalClassifierError):
    """A required on-disk asset is absent, unreadable, or not what we expect.

    Detector weights (absent / wrong size / sha256 mismatch / unpicklable), a
    model artifact (absent, ``format_version`` too new, label-space mismatch,
    missing ``temperature``), or backbone weights. The message always names the
    resolved absolute path and the command that produces the asset.
    """

    exit_code = EXIT_CONFIG


class DecodeError(AnimalClassifierError):
    """One image could not be decoded.

    Recoverable by contract: the image is recorded as skipped or failed with its
    reason and the *run* exits 4. It never aborts the run.
    """

    exit_code = EXIT_PARTIAL


class MaterializeError(AnimalClassifierError):
    """Filing an image into the output tree failed.

    Destination not writable, ``ENOSPC``, or a content collision whose
    hash-suffixed name is also taken — all fatal (exit 1) with the temp file
    cleaned up. A cross-filesystem ``--hardlink`` is raised with
    ``exit_code=EXIT_CONFIG`` because the fix is a different flag.
    """

    exit_code = EXIT_UNEXPECTED


class CatalogError(AnimalClassifierError):
    """The SQLite catalog cannot be used as-is.

    A schema version newer than this build refuses to touch the DB (exit 3).
    Lock contention is *not* this error: it is retried with backoff and, if it
    still fails, recorded against the individual image (exit 4).
    """

    exit_code = EXIT_CONFIG
