"""The ``classify`` orchestration: scan → decode → detect → decide → file (§5, §9).

This module is the only place that knows the *order* of the pipeline. Every stage
it calls is a pure-ish function living in its own module, and the value of keeping
the sequence here is that the invariants which span stages — one coordinate frame,
the source is never written, a second run is a cheap no-op — are visible in one
read.

**Two threaded stages, both bounded (§9).** Hashing, decoding and blur measurement
release the GIL, so they run in a :class:`~concurrent.futures.ThreadPoolExecutor`
of ``jobs`` workers; detection, cropping and every catalog write happen on the
calling thread. Each stage keeps at most ``2 * jobs`` items in flight
(:func:`_bounded_map`), which is what stops a 64 GB card from being pulled into
RAM: the scan is lazy, so an unbounded ``executor.map`` over it would submit every
file on the card before the first result came back.

**Hashing is a stage of its own, ahead of decoding, and that ordering is the
second-run-is-cheap invariant.** ``sha256`` is the identity a row is keyed on, so
it is the only thing needed to ask :func:`~animal_classifier.catalog.plan_disposition`
whether this image is already ``done``. Folding the hash into the decode stage
would be simpler and would make every re-run pay full decode cost for a card that
is entirely filed already.

**``--limit`` is a budget on inference, not on walking.** It is decremented only
when an image is actually accepted for processing, and ``done`` rows are skipped
*before* the budget is consulted (§5.9), so repeated ``--limit`` runs advance
through the card instead of re-examining its first N files forever.

**No species model is loaded yet.** Chunk 17 wires
:mod:`animal_classifier.classify`; until then every animal box reaches §5.7 with
``species=None``, which :func:`~animal_classifier.decide.species_or_unknown`
answers ``unknown``. That is the honest outcome rather than a placeholder species,
and it leaves the dominance rule fully observable: a dominant animal files under
``unknown/`` while a frame with no dominant animal files under ``multiple/``, so
E5's boundary cases and E6's no-area-floor cases assert the *rule* today and
simply gain real species names at chunk 17. Crops are still computed, because
whether a crop is degenerate is what E26 turns on (§5.3).
"""

from __future__ import annotations

import itertools
import logging
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Final, TypeVar

from .catalog import (
    BoxWrite,
    Catalog,
    Disposition,
    RunState,
    Status,
    plan_disposition,
)
from .config import Config, DetectorKind
from .decide import Decision, ScoredBox, decide
from .detect import Box, Detector, ScriptedDetector
from .errors import AssetError
from .images import (
    SPECIES_STATUS_DEGENERATE,
    Blur,
    DecodedImage,
    ImageDecodeError,
    crop,
    decode,
    measure_blur,
    sha256_file,
)
from .materialize import Materialized, Outcome, materialize, sweep_stale_temps
from .scan import Candidate, Skipped, SkipReason, exit_code_for, scan

log = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

#: How often live progress is published to the ``runs`` row, in images. Every
#: image would be a write per image purely for the GUI's benefit; never would make
#: a long run look hung.
PROGRESS_EVERY: Final = 25


@dataclass(frozen=True, slots=True)
class Prepared:
    """Everything the main thread needs about one accepted image."""

    candidate: Candidate
    sha256: str
    decoded: DecodedImage
    blur: Blur


@dataclass
class Counters:
    """The run's tallies, and the skip reasons that set its exit class."""

    n_total: int = 0
    n_done: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    n_already_done: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    reasons: list[SkipReason] = field(default_factory=list)

    def count_outcome(self, outcome: Outcome) -> None:
        self.outcomes[str(outcome)] = self.outcomes.get(str(outcome), 0) + 1

    def count_skip(self, reason: SkipReason) -> None:
        self.n_skipped += 1
        self.reasons.append(reason)


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What one ``classify`` run did, and the exit code it implies."""

    run_id: str
    n_total: int
    n_done: int
    n_skipped: int
    n_failed: int
    n_already_done: int
    outcomes: dict[str, int]
    label_counts: dict[str, int]
    exit_code: int
    dry_run: bool


def build_detector(config: Config) -> Detector:
    """The detector ``config`` selected, or a fatal error naming what is missing.

    ``megadetector`` is not wired yet (chunk 14 owns the weights loading, the
    280 MB checkpoint's size/sha256 pinning and the real inference path). Asking
    for it therefore fails loudly with the alternative spelled out, rather than
    silently falling back to a detector that finds nothing — a silent fallback
    would file an entire safari card as ``landscape`` and look like it worked.
    """
    if config.detector is DetectorKind.SCRIPTED:
        return ScriptedDetector(confidence=config.detector_confidence)
    raise AssetError(
        f"--detector {config.detector} is not implemented yet: real MegaDetector "
        "v5a inference lands in a later build step, together with the checkpoint's "
        "pinned size and sha256. Today only `--detector scripted` can run, which "
        "reads a <image>.boxes.json sidecar beside each photo (DESIGN.md §5.4). "
        "Without a real detector every image is filed as landscape or junk, so "
        "this refuses rather than pretending the card holds no animals."
    )


def classify_run(config: Config, *, argv: Sequence[str]) -> RunSummary:
    """Run the whole ``classify`` pipeline once. Returns the summary and exit code.

    Fatal errors propagate (the caller maps them to their exit codes); per-image
    failures are recorded and the run continues, which is what turns into exit 4.
    """
    if config.source_root is None:  # pragma: no cover - scan() also guards this
        raise AssetError("classify requires a SOURCE directory")

    detector = build_detector(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    if not config.dry_run:
        sweep_stale_temps(config.output_root)

    counters = Counters()
    with Catalog.open(config.catalog_path) as catalog:
        run = catalog.start_run(
            source_root=config.source_root,
            argv=argv,
            config_json=config.to_json_dict(),
        )
        log.info(
            "run %s: %s -> %s (%s%s)",
            run.run_id,
            config.source_root,
            config.output_root,
            config.mode,
            ", dry-run" if config.dry_run else "",
        )
        try:
            _process_card(config, catalog, run.run_id, detector, counters)
        except BaseException:
            catalog.finish_run(
                run.run_id,
                state=RunState.FAILED,
                n_total=counters.n_total,
                n_done=counters.n_done,
                n_skipped=counters.n_skipped,
                n_failed=counters.n_failed,
            )
            raise
        catalog.finish_run(
            run.run_id,
            state=RunState.COMPLETED,
            n_total=counters.n_total,
            n_done=counters.n_done,
            n_skipped=counters.n_skipped,
            n_failed=counters.n_failed,
        )
        label_counts = catalog.label_counts()

    return RunSummary(
        run_id=run.run_id,
        n_total=counters.n_total,
        n_done=counters.n_done,
        n_skipped=counters.n_skipped,
        n_failed=counters.n_failed,
        n_already_done=counters.n_already_done,
        outcomes=dict(counters.outcomes),
        label_counts=label_counts,
        exit_code=exit_code_for(counters.reasons, n_failed=counters.n_failed),
        dry_run=config.dry_run,
    )


def _process_card(
    config: Config,
    catalog: Catalog,
    run_id: str,
    detector: Detector,
    counters: Counters,
) -> None:
    """Walk, prepare and process every eligible image on the card."""
    window = max(2 * config.jobs, 2)
    with ThreadPoolExecutor(max_workers=config.jobs) as pool:
        accepted = _accept(config, catalog, run_id, counters, pool, window)
        prepare = _preparer(config)
        for item, prepared, error in _bounded_map(
            pool, prepare, accepted, window=window
        ):
            if error is not None:
                _record_preparation_failure(catalog, run_id, counters, item, error)
                continue
            assert prepared is not None
            _process_image(config, catalog, run_id, detector, counters, prepared)
            if counters.n_done % PROGRESS_EVERY == 0:
                _publish_progress(catalog, run_id, counters)
    _publish_progress(catalog, run_id, counters)


def _accept(
    config: Config,
    catalog: Catalog,
    run_id: str,
    counters: Counters,
    pool: ThreadPoolExecutor,
    window: int,
) -> Iterator[tuple[Candidate, str]]:
    """Hash every candidate, record skips, and yield the images worth processing.

    Runs on the calling thread — it is a generator, so every catalog write here
    happens on the single writer thread (§5.9) even though the hashing it consumes
    was done in the pool.
    """
    budget = config.limit
    # Hashes already accepted for processing *in this run*. Content — not path — is
    # the identity (§5.9), so the same photo copied twice on the card is classified
    # once and filed once; the duplicate's source path is still recorded. Without
    # this, the second copy's row is merely `planned` (the first has not finished),
    # so it escapes the SKIP_DONE gate and would be filed a second time under its
    # own name — two destination files for one image (E14).
    seen_this_run: set[str] = set()
    for entry, digest, error in _bounded_map(
        pool, _hash_entry, scan(config), window=window
    ):
        if isinstance(entry, Skipped):
            catalog.record_skip(
                entry.path,
                reason=str(entry.reason),
                detail=entry.detail,
                run_id=run_id,
            )
            counters.count_skip(entry.reason)
            continue

        counters.n_total += 1
        if error is not None:
            # The walk stat-ed this file, so a hash failure means it changed under
            # us. One image's problem (§10.1), recorded with its reason.
            reason = _reason_for(error)
            catalog.record_skip(
                entry.path, reason=str(reason), detail=str(error), run_id=run_id
            )
            counters.count_skip(reason)
            continue
        assert digest is not None

        # This path is eligible now, so any stale `skipped` row claiming otherwise
        # must go — that is what makes a widened --formats visible (§5.9).
        catalog.forget_skip(entry.path)

        if digest in seen_this_run:
            # A duplicate of a hash already accepted this run: record the extra
            # source path (so provenance is complete) and move on. n_total counted
            # it as a walked file, but it is neither processed nor filed again.
            catalog.upsert_source(entry.path, digest, entry.mtime_ns)
            counters.n_already_done += 1
            log.debug("%s: duplicate of %s already queued this run", entry.path, digest[:12])
            continue

        row = catalog.image(digest)
        if plan_disposition(row, reclassify=config.reclassify) is Disposition.SKIP_DONE:
            # Before the budget is consulted, and without touching the row: bumping
            # last_updated here would re-order --reclassify --limit's sweep.
            catalog.upsert_source(entry.path, digest, entry.mtime_ns)
            counters.n_already_done += 1
            log.debug("%s: already done (%s)", entry.path, digest[:12])
            continue

        if budget is not None:
            if budget <= 0:
                log.info("--limit reached; stopping before %s", entry.path)
                return
            budget -= 1

        catalog.ensure_image(
            digest,
            run_id=run_id,
            status=Status.PLANNED,
            refresh_state=True,
            bytes=entry.size,
        )
        catalog.upsert_source(entry.path, digest, entry.mtime_ns)
        seen_this_run.add(digest)
        yield entry, digest


def _preparer(config: Config) -> Callable[[tuple[Candidate, str]], Prepared]:
    """Build the threaded decode+blur step, closing over ``--raw``."""

    def prepare(item: tuple[Candidate, str]) -> Prepared:
        candidate, digest = item
        decoded = decode(candidate.path, raw=config.raw)
        return Prepared(
            candidate=candidate,
            sha256=digest,
            decoded=decoded,
            blur=measure_blur(decoded.image),
        )

    return prepare


def _hash_entry(entry: Candidate | Skipped) -> str | None:
    """Threaded stage 1: the content hash of a candidate (``None`` for a skip)."""
    if isinstance(entry, Skipped):
        return None
    return sha256_file(entry.path)


def _process_image(
    config: Config,
    catalog: Catalog,
    run_id: str,
    detector: Detector,
    counters: Counters,
    prepared: Prepared,
) -> None:
    """Detect, decide, file and record one image, on the calling thread."""
    decoded = prepared.decoded
    try:
        boxes = tuple(detector.detect(decoded))
        scored = _score(boxes, decoded, crop_margin=config.crop_margin)
    except ImageDecodeError as error:
        _record_preparation_failure(
            catalog, run_id, counters, (prepared.candidate, prepared.sha256), error
        )
        return
    except Exception as error:
        # §10.1: inference raising is a per-image `failed`, exit 4, never a fatal
        # abort — one malformed frame must not cost the other 4,999.
        log.error(
            "%s: detection failed (%s: %s)",
            decoded.path,
            type(error).__name__,
            error,
            exc_info=log.isEnabledFor(logging.DEBUG),
        )
        catalog.update_image(prepared.sha256, status=Status.FAILED, run_id=run_id)
        counters.n_failed += 1
        return

    decision = decide(
        scored,
        blur_score=prepared.blur.score,
        dominance_ratio=config.dominance_ratio,
        min_species_confidence=config.min_species_confidence,
        blur_threshold=config.blur_threshold,
    )

    filed = materialize(
        prepared.candidate.path,
        label=decision.label,
        sha256=prepared.sha256,
        output_root=config.output_root,
        mode=config.mode,
        dry_run=config.dry_run,
    )
    counters.count_outcome(filed.outcome)

    _write_result(
        catalog,
        run_id,
        prepared,
        scored=scored,
        decision=decision,
        filed=filed,
        detector=detector,
        dry_run=config.dry_run,
    )
    counters.n_done += 1
    log.info(
        "%s -> %s/%s (%s)",
        prepared.candidate.path.name,
        decision.label,
        filed.dest_path.name,
        filed.outcome,
    )


def _score(
    boxes: Sequence[Box], decoded: DecodedImage, *, crop_margin: float
) -> tuple[ScoredBox, ...]:
    """Pair each box with its crop verdict (§5.3).

    The crop's *pixels* are unused until a species model exists, but whether the
    crop is degenerate is decided by geometry alone and is what E26 turns on, so
    the crop is taken now. A degenerate box keeps its place in the sequence: it
    still counts as an animal for dominance, and dropping it would be an area floor
    by the back door.
    """
    scored: list[ScoredBox] = []
    for box in boxes:
        status: str | None = None
        if box.is_animal:
            piece = crop(decoded.image, box.xyxy, crop_margin=crop_margin)
            if piece.degenerate:
                status = SPECIES_STATUS_DEGENERATE
        scored.append(ScoredBox(box=box, species=None, species_status=status))
    return tuple(scored)


def _write_result(
    catalog: Catalog,
    run_id: str,
    prepared: Prepared,
    *,
    scored: Sequence[ScoredBox],
    decision: Decision,
    filed: Materialized,
    detector: Detector,
    dry_run: bool,
) -> None:
    """One transaction per image (§5.9): boxes, candidates and the images row."""
    decoded = prepared.decoded
    catalog.update_image(
        prepared.sha256,
        width=decoded.width,
        height=decoded.height,
        exif_datetime=decoded.exif_datetime,
        gps_lat=decoded.gps_lat,
        gps_lon=decoded.gps_lon,
        blur_score=prepared.blur.score,
        blur_ref_edge=prepared.blur.ref_edge,
    )
    catalog.replace_inference(
        prepared.sha256,
        boxes=[
            BoxWrite(
                idx=idx,
                cls=str(entry.box.cls),
                conf=entry.box.conf,
                x0=entry.box.x0,
                y0=entry.box.y0,
                x1=entry.box.x1,
                y1=entry.box.y1,
                area_frac=entry.area_frac,
                is_dominant=entry is decision.dominant,
                species_status=entry.species_status,
            )
            for idx, entry in enumerate(scored)
        ],
        label=decision.label,
        confidence=decision.confidence,
        species_common=decision.species_common,
        species_scientific=decision.species_scientific,
        species_rank=decision.species_rank,
        model_id=detector.model_id,
        status=Status.PLANNED if dry_run else Status.DONE,
        dest_path=str(filed.dest_path),
        mode=str(filed.mode),
        run_id=run_id,
    )


def _record_preparation_failure(
    catalog: Catalog,
    run_id: str,
    counters: Counters,
    item: tuple[Candidate, str] | Candidate | Skipped,
    error: BaseException,
) -> None:
    """Record a decode-stage failure as the skip §10.1 requires, and continue."""
    candidate = item[0] if isinstance(item, tuple) else item
    assert isinstance(candidate, Candidate)
    digest = item[1] if isinstance(item, tuple) else None
    reason = _reason_for(error)
    log.warning("%s: %s (%s)", candidate.path, error, reason)
    catalog.record_skip(
        candidate.path, reason=str(reason), detail=str(error), run_id=run_id
    )
    if digest is not None:
        catalog.update_image(digest, status=Status.SKIPPED, run_id=run_id)
    counters.count_skip(reason)


def _reason_for(error: BaseException) -> SkipReason:
    """The ``skipped.reason`` an exception implies.

    :class:`~animal_classifier.images.ImageDecodeError` already carries the exact
    reason, which is why it exists; anything else reaching here is unexpected and
    is recorded as ``decode_error`` rather than guessed at more precisely.
    """
    if isinstance(error, ImageDecodeError):
        return error.reason
    return SkipReason.DECODE_ERROR


def _publish_progress(catalog: Catalog, run_id: str, counters: Counters) -> None:
    """Write live counts into the ``runs`` row so the GUI can show progress (§9)."""
    catalog.update_run_counts(
        run_id,
        n_total=counters.n_total,
        n_done=counters.n_done,
        n_skipped=counters.n_skipped,
        n_failed=counters.n_failed,
    )


def _bounded_map(
    pool: ThreadPoolExecutor,
    function: Callable[[T], R],
    items: Iterable[T],
    *,
    window: int,
) -> Iterator[tuple[T, R | None, BaseException | None]]:
    """``map`` with at most ``window`` items in flight, yielding in input order.

    The bound is the point (§9). ``Executor.map`` drains its input iterable
    immediately, so over a lazy scan of a 64 GB card it would submit every file
    before the first result arrived. Results are yielded in submission order so the
    catalog's per-image writes follow walk order, which keeps a run's log readable
    and its `sources` rows deterministic.

    Exceptions are returned beside their item rather than raised, because a single
    unreadable file must not tear down the pool mid-card.
    """
    pending: deque[tuple[T, Future[R]]] = deque()
    remaining = iter(items)
    for item in itertools.islice(remaining, window):
        pending.append((item, pool.submit(function, item)))
    while pending:
        item, future = pending.popleft()
        try:
            yield item, future.result(), None
        except BaseException as error:  # noqa: BLE001 - reported, not swallowed
            yield item, None, error
        nxt = next(remaining, None)
        if nxt is not None:
            pending.append((nxt, pool.submit(function, nxt)))
