"""The SQLite catalog: schema, statuses, upserts and the re-inference transaction.

The catalog at ``<output_root>/.catalog.db`` is the durable record of what this
tool decided, and DESIGN.md §5.9 makes it — not the filesystem — the owner of
label precedence (invariant I6). Three properties of this module are load-bearing:

* **Every re-run is idempotent.** ``skipped`` and ``sources`` are keyed on ``path``
  and are therefore written *only* through ``ON CONFLICT(path) DO UPDATE``.
  A bare ``INSERT INTO skipped`` raised ``IntegrityError`` on the second run over
  the same card — a reproduced defect, and the reason
  :func:`Catalog.record_skip` is the single sanctioned way to write that table.
  Idempotency also has a caller-side half: :meth:`Catalog.ensure_image` refreshes
  ``status``/``run_id`` on a re-seen hash only under an explicit
  ``refresh_state=True``, so a scanner cannot demote a ``done`` row to ``planned``
  merely by walking the card again (follow-up F1).
* **Re-inference replaces, never appends.** :meth:`Catalog.replace_inference` is
  the whole per-image write as one ``BEGIN IMMEDIATE`` transaction in the fixed
  order of §5.9, so a re-classified hash cannot accumulate boxes. It never touches
  ``overrides`` (append-only, I6) and never rewrites ``first_seen``.
* **Lock contention is a per-image outcome, not a dead run.** :meth:`Catalog.do_write`
  retries 0.5/1/2/4/8 s and then raises :class:`CatalogLockedError` (exit 4). Only a
  schema version newer than this build is fatal (:class:`CatalogError`, exit 3).

Ordering constraint worth knowing before writing anything: ``foreign_keys=ON`` and
``sources.sha256 REFERENCES images``, so :meth:`Catalog.upsert_source` must follow
:meth:`Catalog.ensure_image` for that hash.
"""

from __future__ import annotations

import enum
import json
import logging
import sqlite3
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, TypeVar

from .errors import CatalogError, CatalogLockedError

log = logging.getLogger(__name__)

T = TypeVar("T")

#: Bumped only by a schema change that an older build could misread. A DB written
#: by a *newer* version is refused rather than migrated downwards (§5.9).
SCHEMA_VERSION: Final = 1

#: Backoff for a write that lost the lock, verbatim from §5.9. After the last
#: sleep the write is abandoned and the image is recorded ``failed``.
LOCK_BACKOFF_SECONDS: Final = (0.5, 1.0, 2.0, 4.0, 8.0)

#: ``classify`` is the long-lived writer and waits; the GUI's re-tag uses a short
#: timeout so contention surfaces as a fast 409 instead of a hung request (§5.9).
WRITER_BUSY_TIMEOUT_MS: Final = 10_000
GUI_BUSY_TIMEOUT_MS: Final = 250

#: "Newest run" is defined by construction, never by insertion order (§5.9).
NEWEST_RUN_ORDER: Final = "ORDER BY started_at DESC, run_id DESC LIMIT 1"


class Status(enum.StrEnum):
    """``images.status`` — the CHECK constraint's exact value set.

    ``PLANNED`` is also the ``--dry-run`` terminal state: the planned destination
    is recorded and nothing is written to the output tree.
    """

    PLANNED = "planned"
    MATERIALIZING = "materializing"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class LabelSource(enum.StrEnum):
    """``images.label_source`` — a human label outranks the model on later runs."""

    MODEL = "model"
    HUMAN = "human"


class RunState(enum.StrEnum):
    """``runs.state``.

    DESIGN.md §5.9 declares the column without enumerating its values; this is the
    documented value set the rest of the code uses. ``COMPLETED`` covers a run that
    finished with per-image failures too (exit 4) — ``n_failed`` carries that, so
    the state stays a statement about the *run*, and ``RUNNING`` left behind by a
    kill is exactly what tells a later ``verify`` the run was interrupted.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class Disposition(enum.StrEnum):
    """What a scanned hash costs this run — see :func:`plan_disposition`."""

    #: Submit to inference. Consumes ``--limit`` budget.
    PROCESS = "process"
    #: Already ``done``; skipped *before* the budget is consulted, so repeated
    #: ``--limit`` runs advance through the card instead of re-examining its first
    #: N files forever (§5.9).
    SKIP_DONE = "skip_done"


#: Columns :meth:`Catalog.update_image` may set. An allowlist, because the column
#: name is interpolated into SQL and because a typo'd keyword must fail loudly
#: rather than update nothing. ``sha256`` and ``first_seen`` are absent on purpose:
#: identity and discovery time are never rewritten.
UPDATABLE_IMAGE_COLUMNS: Final = frozenset(
    {
        "bytes",
        "width",
        "height",
        "exif_datetime",
        "gps_lat",
        "gps_lon",
        "blur_score",
        "blur_ref_edge",
        "label",
        "label_source",
        "confidence",
        "species_common",
        "species_scientific",
        "species_rank",
        "status",
        "dest_path",
        "mode",
        "model_id",
        "bird_provider",
        "provider_status",
        "run_id",
    }
)

#: §5.9 verbatim. ``IF NOT EXISTS`` only so opening an existing catalog is a no-op;
#: the column lists, CHECK constraints and foreign keys are unchanged from the spec.
SCHEMA_SQL: Final = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS images(
    sha256 TEXT PRIMARY KEY, bytes INT, width INT, height INT,
    exif_datetime TEXT, gps_lat REAL, gps_lon REAL,
    blur_score REAL, blur_ref_edge INT,
    label TEXT, label_source TEXT CHECK(label_source IN ('model','human')),
    confidence REAL, species_common TEXT, species_scientific TEXT, species_rank TEXT,
    status TEXT CHECK(status IN ('planned','materializing','done','skipped','failed')),
    dest_path TEXT, mode TEXT, model_id TEXT, bird_provider TEXT,
    provider_status TEXT, run_id TEXT, first_seen TEXT, last_updated TEXT);

CREATE TABLE IF NOT EXISTS sources(
    path TEXT PRIMARY KEY, sha256 TEXT, mtime_ns INT,
    FOREIGN KEY(sha256) REFERENCES images);

CREATE TABLE IF NOT EXISTS boxes(
    id INTEGER PRIMARY KEY, sha256 TEXT, idx INT, cls TEXT, conf REAL,
    x0 REAL, y0 REAL, x1 REAL, y1 REAL, area_frac REAL,
    species_common TEXT, species_scientific TEXT, species_conf REAL, species_rank TEXT,
    species_status TEXT, is_dominant INT,
    FOREIGN KEY(sha256) REFERENCES images);

CREATE TABLE IF NOT EXISTS candidates(
    box_id INT, rank INT, common TEXT, scientific TEXT, score REAL,
    PRIMARY KEY(box_id, rank), FOREIGN KEY(box_id) REFERENCES boxes);

CREATE TABLE IF NOT EXISTS overrides(
    id INTEGER PRIMARY KEY, sha256 TEXT, old_label TEXT, new_label TEXT,
    created_at TEXT, note TEXT,
    FOREIGN KEY(sha256) REFERENCES images);

CREATE TABLE IF NOT EXISTS skipped(
    path TEXT PRIMARY KEY, reason TEXT, detail TEXT, run_id TEXT, seen_at TEXT);

CREATE TABLE IF NOT EXISTS runs(
    run_id TEXT PRIMARY KEY, source_root TEXT NOT NULL, started_at TEXT, finished_at TEXT,
    argv TEXT, config_json TEXT, n_total INT, n_done INT, n_skipped INT, n_failed INT,
    state TEXT);

CREATE UNIQUE INDEX IF NOT EXISTS idx_boxes_sha_idx ON boxes(sha256, idx);
CREATE INDEX        IF NOT EXISTS idx_boxes_sha     ON boxes(sha256);
CREATE INDEX        IF NOT EXISTS idx_images_label  ON images(label);
CREATE INDEX        IF NOT EXISTS idx_sources_sha   ON sources(sha256);
"""

#: DEFECT 1's fix, as a named constant so that `rg "INSERT INTO skipped" src/`
#: lands on a statement whose name says upsert. ``skipped.path`` is the primary
#: key, so the bare INSERT this replaces raised ``IntegrityError`` the moment the
#: same card was walked twice.
SKIPPED_UPSERT_SQL: Final = (
    "INSERT INTO skipped(path, reason, detail, run_id, seen_at) VALUES(?,?,?,?,?)"
    " ON CONFLICT(path) DO UPDATE SET reason=excluded.reason, detail=excluded.detail,"
    " run_id=excluded.run_id, seen_at=excluded.seen_at"
)

#: The same shape on ``sources.path`` (§5.9): one hash has many paths, and a path
#: whose content changed in place is an update, not a conflict.
SOURCES_UPSERT_SQL: Final = (
    "INSERT INTO sources(path, sha256, mtime_ns) VALUES(?,?,?)"
    " ON CONFLICT(path) DO UPDATE SET sha256=excluded.sha256, mtime_ns=excluded.mtime_ns"
)


def utc_now() -> datetime:
    """One clock for the whole catalog, so ``ORDER BY started_at`` is meaningful."""
    return datetime.now(UTC)


def iso(moment: datetime) -> str:
    """Timestamps are stored as UTC ISO-8601 with a ``Z`` suffix, sortable as text."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_run_id(started_at: datetime) -> str:
    """``run_id`` per §5.9: sortable by construction, with a collision tail."""
    stamp = started_at.astimezone(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass(frozen=True, slots=True)
class BoxWrite:
    """One detection as it is handed to :meth:`Catalog.replace_inference`.

    Coordinates are in the single frame defined by §5.2 (EXIF-oriented pixels of
    the decoded image) and ``area_frac`` is the box's share of that frame.
    ``area_frac`` is recorded for the GUI and for the dominance audit trail; it is
    never compared against a floor — ``dominance_ratio`` is the only size gate.
    """

    idx: int
    cls: str
    conf: float
    x0: float
    y0: float
    x1: float
    y1: float
    area_frac: float
    is_dominant: bool = False
    species_common: str | None = None
    species_scientific: str | None = None
    species_conf: float | None = None
    species_rank: str | None = None
    species_status: str | None = None
    candidates: tuple[CandidateWrite, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateWrite:
    """One ranked alternative for a box. ``rank`` is 1-based and unique per box."""

    rank: int
    common: str
    scientific: str | None
    score: float


@dataclass(frozen=True, slots=True)
class ImageRow:
    """An ``images`` row, read back as a value object."""

    sha256: str
    bytes: int | None
    width: int | None
    height: int | None
    exif_datetime: str | None
    gps_lat: float | None
    gps_lon: float | None
    blur_score: float | None
    blur_ref_edge: int | None
    label: str | None
    label_source: str | None
    confidence: float | None
    species_common: str | None
    species_scientific: str | None
    species_rank: str | None
    status: str | None
    dest_path: str | None
    mode: str | None
    model_id: str | None
    bird_provider: str | None
    provider_status: str | None
    run_id: str | None
    first_seen: str | None
    last_updated: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> ImageRow:
        return cls(**{key: row[key] for key in cls.__slots__})


@dataclass(frozen=True, slots=True)
class RunRow:
    """A ``runs`` row. ``source_root`` is the only provenance of the source tree."""

    run_id: str
    source_root: str
    started_at: str | None
    finished_at: str | None
    argv: str | None
    config_json: str | None
    n_total: int | None
    n_done: int | None
    n_skipped: int | None
    n_failed: int | None
    state: str | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> RunRow:
        return cls(**{key: row[key] for key in cls.__slots__})


def plan_disposition(row: ImageRow | None, *, reclassify: bool) -> Disposition:
    """The full re-processing policy of §5.9, in one place.

    ==================  ==========================================================
    Existing status     Outcome
    ==================  ==========================================================
    no row              processed (never seen)
    ``done``            skipped unless ``--reclassify``; costs no budget
    ``planned``         re-processed — an interrupted or dry-run row is not a result
    ``materializing``   re-processed — safe, materialization is content-addressed
    ``failed``          **retried**, and the retry consumes ``--limit`` budget
    ``skipped``         re-processed; a scan-time skip is re-evaluated every run
    ==================  ==========================================================

    A path recorded in the ``skipped`` *table* is always re-walked and re-judged
    against the current format policy, which is why widening ``--formats`` picks up
    previously-skipped files without ``--reclassify``. That happens in ``scan``;
    here it appears as a row that is absent or at ``Status.SKIPPED``.
    """
    if row is None or row.status is None:
        return Disposition.PROCESS
    if row.status == Status.DONE and not reclassify:
        return Disposition.SKIP_DONE
    return Disposition.PROCESS


class Catalog:
    """A connection to one catalog DB, opened read-write or read-only.

    Not thread-safe by design: ``classify`` is the single long-lived writer (§5.9)
    and the GUI opens its own short-lived connections per request.
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        path: Path,
        *,
        read_only: bool,
        retry_writes: bool = True,
    ) -> None:
        self._conn = connection
        self._path = path
        self._read_only = read_only
        # `classify` is a long-lived writer and retries lock contention with the
        # documented backoff. The GUI must **not**: §5.9 requires its re-tag to turn
        # contention into a fast, deterministic 409 rather than a browser request
        # hanging for ten seconds, so it opens with retry_writes=False and the very
        # first SQLITE_BUSY raises CatalogLockedError.
        self._retry_writes = retry_writes

    # ------------------------------------------------------------------ opening

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        read_only: bool = False,
        busy_timeout_ms: int = WRITER_BUSY_TIMEOUT_MS,
        retry_writes: bool = True,
    ) -> Catalog:
        """Open (creating when writable) and validate the schema version.

        ``read_only=True`` uses SQLite's ``mode=ro`` URI, so every GUI ``GET``
        provably cannot contend with a running ``classify`` (§5.9). A read-only
        open of a catalog that does not exist is an error naming the path, never a
        silently empty database.

        ``retry_writes=False`` is the GUI's write mode: the first ``SQLITE_BUSY``
        raises :class:`CatalogLockedError` instead of entering ``classify``'s
        0.5/1/2/4/8 s backoff, which is what turns contention into a fast 409
        (§5.9) rather than a hung browser request.
        """
        if read_only:
            if not path.is_file():
                raise CatalogError(
                    f"no catalog at {path}. Run `animal-classifier classify SOURCE "
                    "--output <root>` first, or point --output at the root you filed into."
                )
            if not path.is_absolute():
                raise CatalogError(
                    f"a read-only catalog must be opened by absolute path, got {path}"
                )
            # as_uri() percent-encodes, so a '?' or '#' in the output root cannot be
            # misread as the start of the URI's query string.
            conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, isolation_level=None)
        else:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o755)
            conn = sqlite3.connect(path, isolation_level=None)

        conn.row_factory = sqlite3.Row
        # WAL survives a reopen, so it is set once on the writable open; asking for
        # it on a read-only connection would fail against a read-only file.
        if not read_only:
            conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")

        catalog = cls(conn, path, read_only=read_only, retry_writes=retry_writes)
        try:
            if not read_only:
                # The version gate runs before the rest of the schema: a DB written
                # by a newer build must be refused untouched, not first have this
                # build's CREATE TABLEs applied to it.
                conn.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT)")
            catalog._check_schema_version()
            if not read_only:
                conn.executescript(SCHEMA_SQL)
        except BaseException:
            conn.close()
            raise
        return catalog

    def _check_schema_version(self) -> None:
        """Refuse a DB written by a newer build rather than corrupt it (§5.9)."""
        try:
            row = self._conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
        except sqlite3.OperationalError as exc:  # read-only open of a foreign file
            raise CatalogError(
                f"{self._path} is not an animal-classifier catalog ({exc})."
            ) from exc

        if row is None:
            if self._read_only:
                raise CatalogError(
                    f"{self._path} has no schema_version row, so it was not written by "
                    "this tool (or was left half-created). Re-run `classify` against "
                    "this output root."
                )
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            return

        found = str(row["value"])
        try:
            version = int(found)
        except ValueError as exc:
            raise CatalogError(
                f"{self._path} has a non-numeric schema_version {found!r}; the file is "
                "not a catalog this build can read."
            ) from exc
        if version > SCHEMA_VERSION:
            raise CatalogError(
                f"{self._path} was written by a newer animal-classifier (schema "
                f"version {version}, this build understands {SCHEMA_VERSION}). "
                "Refusing to touch it — upgrade the tool, or use a different "
                "--output root."
            )
        if version < SCHEMA_VERSION:  # pragma: no cover - no older version exists yet
            raise CatalogError(
                f"{self._path} is at schema version {version} and no migration to "
                f"{SCHEMA_VERSION} is implemented in this build."
            )

    @property
    def path(self) -> Path:
        return self._path

    @property
    def connection(self) -> sqlite3.Connection:
        """For read queries that want plain SQL (the GUI's list endpoints)."""
        return self._conn

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------ transactions

    @contextmanager
    def write_tx(self) -> Iterator[sqlite3.Connection]:
        """One ``BEGIN IMMEDIATE`` transaction. **One attempt, no retry.**

        The write lock is taken up front, so a contended writer fails here instead
        of half-way through. Callers that must not hang (the GUI's re-tag, which
        turns contention into a 409) use this directly; ``classify`` uses
        :meth:`do_write`, which adds the documented backoff.
        """
        if self._read_only:
            raise CatalogError(
                f"{self._path} is open read-only; this code path must not write. "
                "GUI GET routes are read-only by design (DESIGN.md §5.9)."
            )
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        self._conn.execute("COMMIT")

    def do_write(self, operation: str, body: Callable[[sqlite3.Connection], T]) -> T:
        """Run ``body`` in a write transaction, retrying lock contention.

        Backoff is 0.5/1/2/4/8 s (§5.9); the whole transaction is retried, which is
        sound because ``body`` is invoked only after the lock is held and is rolled
        back entirely on failure. When the last attempt still loses, this raises
        :class:`CatalogLockedError` so the caller records **one image** as failed
        and the run exits 4 — a per-image failure never aborts the run.
        """
        attempts = len(LOCK_BACKOFF_SECONDS) + 1 if self._retry_writes else 1
        for attempt in range(attempts):
            try:
                with self.write_tx() as conn:
                    return body(conn)
            except sqlite3.OperationalError as exc:
                if not _is_locked(exc) or attempt == attempts - 1:
                    if _is_locked(exc):
                        waited = (
                            f"stayed locked for {sum(LOCK_BACKOFF_SECONDS):.1f}s"
                            if self._retry_writes
                            else "is locked (this writer does not retry, by design)"
                        )
                        raise CatalogLockedError(
                            f"{operation}: the catalog {waited} ({exc}). Another "
                            "classify run or GUI write is holding it."
                        ) from exc
                    raise
                delay = LOCK_BACKOFF_SECONDS[attempt]
                log.warning(
                    "%s: catalog locked (%s); retrying in %.1fs", operation, exc, delay
                )
                time.sleep(delay)
        raise AssertionError("unreachable: the loop either returns or raises")

    # -------------------------------------------------------------------- runs

    def start_run(
        self,
        *,
        source_root: Path | str,
        argv: Sequence[str],
        config_json: Mapping[str, Any],
        started_at: datetime | None = None,
        n_total: int | None = None,
    ) -> RunRow:
        """Insert the ``runs`` row at run start.

        ``source_root`` is ``NOT NULL`` because it is the only provenance of the
        card for every later command, so an empty value fails loudly here rather
        than leaving ``verify`` with nothing to check.
        """
        root = str(source_root).strip()
        if not root:
            raise CatalogError(
                "start_run requires a source_root: it is the only record of which "
                "card a run read, and `verify` reads it back for its nesting check."
            )
        moment = started_at or utc_now()
        run_id = new_run_id(moment)
        row = (
            run_id,
            root,
            iso(moment),
            None,
            json.dumps(list(argv)),
            json.dumps(dict(config_json), sort_keys=True),
            n_total,
            0,
            0,
            0,
            str(RunState.RUNNING),
        )

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO runs(run_id, source_root, started_at, finished_at, argv,"
                " config_json, n_total, n_done, n_skipped, n_failed, state)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                row,
            )

        self.do_write("start_run", _insert)
        return self.run(run_id)

    def update_run_counts(
        self,
        run_id: str,
        *,
        n_total: int | None = None,
        n_done: int | None = None,
        n_skipped: int | None = None,
        n_failed: int | None = None,
    ) -> None:
        """Publish live progress so the GUI's ``/api/run`` can show it mid-run."""
        updates = {
            "n_total": n_total,
            "n_done": n_done,
            "n_skipped": n_skipped,
            "n_failed": n_failed,
        }
        given = {col: val for col, val in updates.items() if val is not None}
        if not given:
            return
        assignments = ", ".join(f"{col} = ?" for col in given)
        params = (*given.values(), run_id)

        def _update(conn: sqlite3.Connection) -> None:
            conn.execute(f"UPDATE runs SET {assignments} WHERE run_id = ?", params)

        self.do_write("update_run_counts", _update)

    def finish_run(
        self,
        run_id: str,
        *,
        state: RunState,
        n_total: int,
        n_done: int,
        n_skipped: int,
        n_failed: int,
        finished_at: datetime | None = None,
    ) -> None:
        """Close the run out with its final counts and terminal state."""
        params = (
            iso(finished_at or utc_now()),
            n_total,
            n_done,
            n_skipped,
            n_failed,
            str(state),
            run_id,
        )

        def _update(conn: sqlite3.Connection) -> None:
            conn.execute(
                "UPDATE runs SET finished_at = ?, n_total = ?, n_done = ?,"
                " n_skipped = ?, n_failed = ?, state = ? WHERE run_id = ?",
                params,
            )

        self.do_write("finish_run", _update)

    def run(self, run_id: str) -> RunRow:
        row = self._conn.execute(
            "SELECT * FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            raise CatalogError(f"no runs row for run_id {run_id!r} in {self._path}")
        return RunRow.from_row(row)

    def newest_run(self) -> RunRow | None:
        """The newest run, defined as ``started_at DESC, run_id DESC`` (§5.9).

        Never "the last row inserted": that would be wrong for a resumed card and
        is exactly the ambiguity ``verify`` and ``GET /api/run`` must not inherit.
        """
        row = self._conn.execute(f"SELECT * FROM runs {NEWEST_RUN_ORDER}").fetchone()
        return None if row is None else RunRow.from_row(row)

    # ------------------------------------------------------------------ images

    def image(self, sha256: str) -> ImageRow | None:
        row = self._conn.execute(
            "SELECT * FROM images WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return None if row is None else ImageRow.from_row(row)

    def ensure_image(
        self,
        sha256: str,
        *,
        run_id: str,
        status: Status = Status.PLANNED,
        refresh_state: bool = False,
        now: datetime | None = None,
        **columns: Any,
    ) -> None:
        """Insert the row for a newly seen hash, or refresh a re-seen one.

        ``first_seen`` is written once and never rewritten — an image discovered on
        the first pass keeps its discovery time through every ``--reclassify``.
        Must be called before :meth:`upsert_source` for the same hash, because
        ``sources.sha256`` is a foreign key into this table.

        **On a re-seen hash only the columns the caller actually named are
        refreshed** (plus ``last_updated``). ``status`` and ``run_id`` are needed to
        insert a *new* row, but refreshing them on an existing row requires the
        explicit ``refresh_state=True`` opt-in (follow-up F1). Without that opt-in a
        scanner that calls this unconditionally for every hash on the card would
        reset every ``done`` row to ``planned``; :func:`plan_disposition` reads
        ``row.status`` alone, so the whole card would be re-processed and the
        second-run-is-a-clean-no-op invariant would break from the *caller* side,
        where the ``skipped`` upsert (DEFECT 1) cannot protect it. The pipeline
        passes ``refresh_state=True`` exactly when it has decided to (re-)process
        this hash in this run, i.e. after :func:`plan_disposition` returned
        :attr:`Disposition.PROCESS`.
        """
        unknown = set(columns) - UPDATABLE_IMAGE_COLUMNS
        if unknown:
            raise CatalogError(
                f"ensure_image got unknown images column(s) {sorted(unknown)}; "
                f"settable columns are {sorted(UPDATABLE_IMAGE_COLUMNS)}"
            )
        stamp = iso(now or utc_now())
        values: dict[str, Any] = {
            **columns,
            "sha256": sha256,
            "status": str(status),
            "run_id": run_id,
            "first_seen": stamp,
            "last_updated": stamp,
        }
        names = list(values)
        placeholders = ", ".join(f":{name}" for name in names)
        # sha256 is the conflict target and first_seen is write-once, so neither is
        # ever in the UPDATE list; status/run_id join it only under refresh_state.
        refreshed = [*columns, "last_updated"]
        if refresh_state:
            refreshed += ["status", "run_id"]
        assignments = ", ".join(f"{name} = excluded.{name}" for name in refreshed)

        def _upsert(conn: sqlite3.Connection) -> None:
            conn.execute(
                f"INSERT INTO images({', '.join(names)}) VALUES({placeholders})"
                f" ON CONFLICT(sha256) DO UPDATE SET {assignments}",
                values,
            )

        self.do_write("ensure_image", _upsert)

    def update_image(
        self, sha256: str, *, now: datetime | None = None, **columns: Any
    ) -> None:
        """Set named ``images`` columns, refreshing ``last_updated``.

        Unknown column names raise instead of updating nothing, so a typo in a
        keyword cannot look like a successful write.
        """
        if not columns:
            raise CatalogError("update_image was called with no columns to set")
        unknown = set(columns) - UPDATABLE_IMAGE_COLUMNS
        if unknown:
            raise CatalogError(
                f"update_image got unknown images column(s) {sorted(unknown)}; "
                f"settable columns are {sorted(UPDATABLE_IMAGE_COLUMNS)}"
            )
        values = dict(columns)
        values["last_updated"] = iso(now or utc_now())
        assignments = ", ".join(f"{name} = :{name}" for name in values)
        values["sha256"] = sha256

        def _update(conn: sqlite3.Connection) -> None:
            cursor = conn.execute(
                f"UPDATE images SET {assignments} WHERE sha256 = :sha256", values
            )
            if cursor.rowcount == 0:
                raise CatalogError(
                    f"update_image found no images row for {sha256}; the row must be "
                    "created by ensure_image before it can be updated."
                )

        self.do_write("update_image", _update)

    def hashes_by_staleness(self, limit: int | None = None) -> tuple[str, ...]:
        """Hashes oldest-classified first, ``NULL`` (never classified) first (§5.9).

        This is the ``--reclassify --limit N`` order, so repeated runs sweep the
        whole card exactly once per pass instead of re-doing the same N images.
        ``sha256`` breaks ties so the order is total and reproducible.
        """
        sql = (
            "SELECT sha256 FROM images"
            " ORDER BY (last_updated IS NULL) DESC, last_updated ASC, sha256 ASC"
        )
        params: tuple[Any, ...] = ()
        if limit is not None:
            sql += " LIMIT ?"
            params = (limit,)
        return tuple(row["sha256"] for row in self._conn.execute(sql, params))

    def label_counts(self) -> dict[str, int]:
        """Catalog-side label histogram (the GUI's per-label count, §6)."""
        rows = self._conn.execute(
            "SELECT label, COUNT(*) AS n FROM images WHERE label IS NOT NULL"
            " GROUP BY label ORDER BY label"
        )
        return {row["label"]: row["n"] for row in rows}

    def count_by_status(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT status, COUNT(*) AS n FROM images GROUP BY status"
        )
        return {row["status"]: row["n"] for row in rows}

    # ----------------------------------------------------------------- sources

    def upsert_source(self, path: Path | str, sha256: str, mtime_ns: int) -> None:
        """Record one source path for a hash — upsert, per §5.9.

        One hash legitimately has many paths (the same photo copied twice on the
        card): it is classified once and filed once. A path whose **content
        changed in place** is an update, not a conflict, so re-running over an
        edited card is not an error. Requires the ``images`` row to exist
        (``foreign_keys=ON``).
        """
        params = (str(path), sha256, int(mtime_ns))

        def _upsert(conn: sqlite3.Connection) -> None:
            conn.execute(SOURCES_UPSERT_SQL, params)

        self.do_write("upsert_source", _upsert)

    def source_paths(self, sha256: str) -> tuple[str, ...]:
        rows = self._conn.execute(
            "SELECT path FROM sources WHERE sha256 = ? ORDER BY path", (sha256,)
        )
        return tuple(row["path"] for row in rows)

    # ----------------------------------------------------------------- skipped

    def record_skip(
        self,
        path: Path | str,
        *,
        reason: str,
        detail: str | None,
        run_id: str,
        seen_at: datetime | None = None,
    ) -> None:
        """Record a skipped path. **The only write path into ``skipped``.**

        DEFECT 1: ``skipped.path`` is the primary key, so the plain
        ``INSERT INTO skipped`` this replaces raised ``IntegrityError`` the moment
        the same card was walked twice — the second run died instead of being a
        no-op. The upsert makes a re-walk overwrite the previous verdict with the
        current one (reason, detail and the run that last saw it), which is also
        what makes a widened ``--formats`` visible: the row is refreshed, not
        duplicated. There must be no other ``INSERT INTO skipped`` in the codebase.
        """
        params = (
            str(path),
            reason,
            detail,
            run_id,
            iso(seen_at or utc_now()),
        )

        def _upsert(conn: sqlite3.Connection) -> None:
            conn.execute(SKIPPED_UPSERT_SQL, params)

        self.do_write("record_skip", _upsert)

    def forget_skip(self, path: Path | str) -> None:
        """Drop a skip record because the path is now eligible.

        Skips are judged fresh every run (§5.9): a file that a widened
        ``--formats`` now accepts must not keep a stale ``skipped`` row claiming
        the tool ignored it.
        """

        def _delete(conn: sqlite3.Connection) -> None:
            conn.execute("DELETE FROM skipped WHERE path = ?", (str(path),))

        self.do_write("forget_skip", _delete)

    def skip_counts(self) -> dict[str, int]:
        rows = self._conn.execute(
            "SELECT reason, COUNT(*) AS n FROM skipped GROUP BY reason ORDER BY reason"
        )
        return {row["reason"]: row["n"] for row in rows}

    def skipped_paths(self) -> tuple[str, ...]:
        rows = self._conn.execute("SELECT path FROM skipped ORDER BY path")
        return tuple(row["path"] for row in rows)

    # --------------------------------------------------------------- inference

    def replace_inference(
        self,
        sha256: str,
        *,
        boxes: Sequence[BoxWrite],
        label: str | None,
        confidence: float | None,
        species_common: str | None = None,
        species_scientific: str | None = None,
        species_rank: str | None = None,
        model_id: str | None = None,
        bird_provider: str | None = None,
        provider_status: str | None = None,
        status: Status,
        dest_path: str | None = None,
        mode: str | None = None,
        run_id: str,
        now: datetime | None = None,
    ) -> None:
        """The §5.9 per-image write, as one transaction in the fixed order.

        ``DELETE candidates`` → ``DELETE boxes`` → insert the new boxes and their
        candidates → ``UPDATE images``. Re-inference therefore **replaces**: a
        re-classified hash cannot accumulate boxes, and ``COUNT(*)`` over ``boxes``
        and ``candidates`` is unchanged by a second run over the same card.

        Two tables are never touched here: ``overrides`` is append-only (I6), and
        ``first_seen`` is never rewritten. ``label_source`` is likewise not set by
        this method, so a human label survives ``--reclassify`` unless the caller
        explicitly demotes it via :meth:`update_image`.
        """
        stamp = iso(now or utc_now())
        ranks = [(box.idx, c.rank) for box in boxes for c in box.candidates]
        if len(set(ranks)) != len(ranks):
            raise CatalogError(
                f"replace_inference for {sha256} got duplicate candidate ranks "
                f"{ranks}; candidates are PRIMARY KEY(box_id, rank)."
            )
        indices = [box.idx for box in boxes]
        if len(set(indices)) != len(indices):
            raise CatalogError(
                f"replace_inference for {sha256} got duplicate box indices "
                f"{indices}; boxes carries a UNIQUE(sha256, idx) index."
            )

        def _write(conn: sqlite3.Connection) -> None:
            conn.execute(
                "DELETE FROM candidates WHERE box_id IN"
                " (SELECT id FROM boxes WHERE sha256 = ?)",
                (sha256,),
            )
            conn.execute("DELETE FROM boxes WHERE sha256 = ?", (sha256,))
            for box in boxes:
                cursor = conn.execute(
                    "INSERT INTO boxes(sha256, idx, cls, conf, x0, y0, x1, y1,"
                    " area_frac, species_common, species_scientific, species_conf,"
                    " species_rank, species_status, is_dominant)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        sha256,
                        box.idx,
                        box.cls,
                        box.conf,
                        box.x0,
                        box.y0,
                        box.x1,
                        box.y1,
                        box.area_frac,
                        box.species_common,
                        box.species_scientific,
                        box.species_conf,
                        box.species_rank,
                        box.species_status,
                        int(box.is_dominant),
                    ),
                )
                box_id = cursor.lastrowid
                if box.candidates:
                    conn.executemany(
                        "INSERT INTO candidates(box_id, rank, common, scientific, score)"
                        " VALUES(?,?,?,?,?)",
                        [
                            (box_id, c.rank, c.common, c.scientific, c.score)
                            for c in box.candidates
                        ],
                    )
            cursor = conn.execute(
                "UPDATE images SET label = ?, confidence = ?, species_common = ?,"
                " species_scientific = ?, species_rank = ?, model_id = ?,"
                " bird_provider = ?, provider_status = ?, status = ?, dest_path = ?,"
                " mode = ?, run_id = ?, last_updated = ? WHERE sha256 = ?",
                (
                    label,
                    confidence,
                    species_common,
                    species_scientific,
                    species_rank,
                    model_id,
                    bird_provider,
                    provider_status,
                    str(status),
                    dest_path,
                    mode,
                    run_id,
                    stamp,
                    sha256,
                ),
            )
            if cursor.rowcount == 0:
                raise CatalogError(
                    f"replace_inference found no images row for {sha256}; call "
                    "ensure_image when the hash is first seen."
                )

        self.do_write("replace_inference", _write)

    def boxes(self, sha256: str) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self._conn.execute(
                "SELECT * FROM boxes WHERE sha256 = ? ORDER BY idx", (sha256,)
            )
        )

    def candidates(self, box_id: int) -> tuple[sqlite3.Row, ...]:
        return tuple(
            self._conn.execute(
                "SELECT * FROM candidates WHERE box_id = ? ORDER BY rank", (box_id,)
            )
        )

    # --------------------------------------------------------------- overrides

    def insert_override(
        self,
        sha256: str,
        *,
        old_label: str | None,
        new_label: str,
        note: str | None = None,
        created_at: datetime | None = None,
    ) -> None:
        """Append a human correction. ``overrides`` is append-only (I6).

        Nothing in this module ever deletes or rewrites an override row, which is
        what makes ``--ignore-overrides`` a suppression for one run rather than a
        way to destroy a human decision.
        """
        params = (
            sha256,
            old_label,
            new_label,
            iso(created_at or utc_now()),
            note,
        )

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO overrides(sha256, old_label, new_label, created_at, note)"
                " VALUES(?,?,?,?,?)",
                params,
            )

        self.do_write("insert_override", _insert)

    def newest_override(self, sha256: str) -> sqlite3.Row | None:
        """The override a later run must re-apply: newest by ``created_at``, then id."""
        return self._conn.execute(
            "SELECT * FROM overrides WHERE sha256 = ?"
            " ORDER BY created_at DESC, id DESC LIMIT 1",
            (sha256,),
        ).fetchone()

    def override_count(self, sha256: str | None = None) -> int:
        if sha256 is None:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM overrides").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM overrides WHERE sha256 = ?", (sha256,)
            ).fetchone()
        return int(row["n"])


def _is_locked(exc: sqlite3.OperationalError) -> bool:
    """Distinguish lock contention from a real SQL error, by SQLite's own words."""
    message = str(exc).lower()
    return "locked" in message or "busy" in message
