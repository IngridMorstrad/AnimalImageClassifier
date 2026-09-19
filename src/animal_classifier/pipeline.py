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
from typing import Any, Final, TypeVar

from .catalog import (
    BoxWrite,
    CandidateWrite,
    Catalog,
    Disposition,
    LabelSource,
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
    if config.detector is DetectorKind.MEGADETECTOR:
        from .detect import MegaDetector

        return MegaDetector.load(
            config.detector_weights,
            confidence=config.detector_confidence,
            iou=config.detector_iou,
            image_size=config.detector_image_size,
            max_det=config.detector_max_det,
            jobs=config.jobs,
            allow_download=not config.no_download,
        )
    raise AssetError(  # pragma: no cover - DetectorKind is a closed enum
        f"unknown detector {config.detector!r}"
    )


def build_classifier(config: Config) -> Any:
    """The species classifier, or ``None`` when no model is available (§5.5, F27).

    Loading is opt-in on the artifact existing: with no ``species_model`` present
    the pipeline runs the pass-through path (every animal → ``unknown``) rather than
    refusing, which is what lets ``classify`` do useful landscape/junk triage before
    a model has been trained. Once ``SPECIES_INFERENCE_WIRED`` is true and a model is
    configured, a *missing* artifact is fatal (§10.1) — the config layer enforces
    that, so by the time we get here the path either exists or species inference is
    switched off.
    """
    from .config import SPECIES_INFERENCE_WIRED

    if config.detect_only:
        log.info(
            "detect-only: skipping species inference, so every animal reaches the "
            "review queue for you to name"
        )
        return None
    if not SPECIES_INFERENCE_WIRED:
        return None
    if not config.species_model.exists():
        # config validation only requires the artifact when SPECIES_INFERENCE_WIRED
        # is on; if it is on and the file is absent, that was already fatal. This
        # guard keeps the pass-through path available for a run that pointed at no
        # model on purpose.
        return None
    from .classify.own_model import SpeciesClassifier

    return SpeciesClassifier.from_path(config.species_model)


def classify_run(config: Config, *, argv: Sequence[str]) -> RunSummary:
    """Run the whole ``classify`` pipeline once. Returns the summary and exit code.

    Fatal errors propagate (the caller maps them to their exit codes); per-image
    failures are recorded and the run continues, which is what turns into exit 4.
    """
    if config.source_root is None:  # pragma: no cover - scan() also guards this
        raise AssetError("classify requires a SOURCE directory")

    detector = build_detector(config)
    classifier = build_classifier(config)
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
            _process_card(config, catalog, run.run_id, detector, classifier, counters)
        except KeyboardInterrupt:
            # Ctrl-C is a legitimate way to stop a long card. Close the run out with
            # the work that *did* land so `verify` and the GUI see honest counts, and
            # say plainly that re-running resumes — every finished image is already
            # `done`, so the next run skips it.
            catalog.finish_run(
                run.run_id,
                state=RunState.FAILED,
                n_total=counters.n_total,
                n_done=counters.n_done,
                n_skipped=counters.n_skipped,
                n_failed=counters.n_failed,
            )
            log.warning(
                "interrupted after %d image(s); they are saved. Re-run the same "
                "command to continue where it stopped.",
                counters.n_done,
            )
            raise
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
    classifier: Any,
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
            _process_image(config, catalog, run_id, detector, classifier, counters, prepared)
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
    classifier: Any,
    counters: Counters,
    prepared: Prepared,
) -> None:
    """Detect, classify, decide, file and record one image, on the calling thread."""
    decoded = prepared.decoded
    try:
        boxes = tuple(detector.detect(decoded))
        scored, candidates = _score(
            boxes, decoded, crop_margin=config.crop_margin, classifier=classifier
        )
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

    # §5.9: `--reclassify` keeps `label_source='human'` labels unless
    # `--ignore-overrides`. The model's inference (boxes, candidates, scores) is
    # still recorded — only the *filed label* is held to the human's correction, so
    # a re-run does not silently move a photo the user deliberately re-tagged. The
    # override is never deleted (I6); `--ignore-overrides` merely suppresses it for
    # this run and demotes the row to `model`.
    label_source = LabelSource.MODEL
    human = _human_override(catalog, prepared.sha256, config)
    if human is not None:
        label = human
        label_source = LabelSource.HUMAN
    else:
        label = decision.label

    filed = materialize(
        prepared.candidate.path,
        label=label,
        sha256=prepared.sha256,
        output_root=config.output_root,
        mode=config.mode,
        dry_run=config.dry_run,
    )
    counters.count_outcome(filed.outcome)

    model_id = detector.model_id
    if classifier is not None:
        model_id = f"{detector.model_id}+{classifier.model_id}"
    _write_result(
        catalog,
        run_id,
        prepared,
        scored=scored,
        candidates=candidates,
        decision=decision,
        filed=filed,
        label=label,
        label_source=label_source,
        model_id=model_id,
        dry_run=config.dry_run,
    )
    counters.n_done += 1
    log.info(
        "%s -> %s/%s (%s%s)",
        prepared.candidate.path.name,
        label,
        filed.dest_path.name,
        filed.outcome,
        ", human" if label_source is LabelSource.HUMAN else "",
    )


def _human_override(catalog: Catalog, sha256: str, config: Config) -> str | None:
    """The newest human label to re-apply for this hash, or ``None`` (§5.9).

    **Read from the ``overrides`` table, never from ``images.label_source``.** That
    distinction is the whole of ``--ignore-overrides``' contract: the flag suppresses
    an override for one run and demotes the row to ``label_source='model'``, so a
    later run *without* the flag must still re-apply the correction and move the file
    back. Keying off ``label_source`` would make the demotion permanent — the flag
    would silently destroy a human decision, which is precisely what §5.9 forbids
    (permanent discard is done by re-tagging, which appends a *newer* override).

    ``overrides`` is append-only (I6), so the newest row is the standing intent.
    """
    if config.ignore_overrides:
        return None
    override = catalog.newest_override(sha256)
    if override is None:
        return None
    return override["new_label"]


def _score(
    boxes: Sequence[Box],
    decoded: DecodedImage,
    *,
    crop_margin: float,
    classifier: Any,
) -> tuple[tuple[ScoredBox, ...], dict[int, list[Any]]]:
    """Crop each box, classify the non-degenerate animal crops (§5.3, §5.5).

    Whether a crop is degenerate is geometry alone and is what E26 turns on, so the
    crop is always taken. A degenerate box keeps its place — it still counts as an
    animal for dominance, and dropping it would be an area floor by the back door.

    When ``classifier`` is ``None`` (no species model) every animal reaches §5.7
    with ``species=None`` and becomes ``unknown``. When present, the classifiable
    animal crops are batched through it in one forward pass, and the per-box top-5
    ``candidates`` are returned keyed by box index for the ``candidates`` table.
    """
    scored: list[ScoredBox] = []
    classifiable: list[tuple[int, Any]] = []  # (box index, crop image)
    for index, box in enumerate(boxes):
        status: str | None = None
        crop_image = None
        if box.is_animal:
            piece = crop(decoded.image, box.xyxy, crop_margin=crop_margin)
            if piece.degenerate:
                status = SPECIES_STATUS_DEGENERATE
            else:
                crop_image = piece.image
        scored.append(ScoredBox(box=box, species=None, species_status=status))
        if crop_image is not None:
            classifiable.append((index, crop_image))

    candidates: dict[int, list[Any]] = {}
    if classifier is not None and classifiable:
        predictions = classifier.classify([image for _, image in classifiable])
        for (index, _), (prediction, box_candidates) in zip(classifiable, predictions, strict=True):
            scored[index] = ScoredBox(
                box=scored[index].box,
                species=prediction,
                species_status=None,
            )
            candidates[index] = box_candidates
    return tuple(scored), candidates


def _write_result(
    catalog: Catalog,
    run_id: str,
    prepared: Prepared,
    *,
    scored: Sequence[ScoredBox],
    candidates: dict[int, list[Any]],
    decision: Decision,
    filed: Materialized,
    label: str,
    label_source: LabelSource,
    model_id: str,
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
                species_common=None if entry.species is None else entry.species.common,
                species_scientific=None if entry.species is None else entry.species.scientific,
                species_conf=entry.species_conf,
                species_rank=None if entry.species is None else entry.species.rank,
                species_status=entry.species_status,
                candidates=tuple(
                    CandidateWrite(
                        rank=c.rank,
                        common=c.common,
                        scientific=c.scientific,
                        score=c.score,
                    )
                    for c in candidates.get(idx, [])
                ),
            )
            for idx, entry in enumerate(scored)
        ],
        label=label,
        confidence=decision.confidence,
        species_common=decision.species_common,
        species_scientific=decision.species_scientific,
        species_rank=decision.species_rank,
        model_id=model_id,
        status=Status.PLANNED if dry_run else Status.DONE,
        dest_path=str(filed.dest_path),
        mode=str(filed.mode),
        run_id=run_id,
    )
    # replace_inference does not touch label_source (a human label must survive a
    # model re-run). Set it explicitly so a re-tagged image stays `human` and a
    # fresh model decision is recorded as `model` (§5.9).
    catalog.update_image(prepared.sha256, label_source=str(label_source))


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
    try:
        for item in itertools.islice(remaining, window):
            pending.append((item, pool.submit(function, item)))
        while pending:
            item, future = pending.popleft()
            try:
                result = future.result()
            except Exception as error:
                # Deliberately `Exception`, not `BaseException`. Catching
                # GeneratorExit/KeyboardInterrupt here and then yielding again is
                # what raised "RuntimeError: generator ignored GeneratorExit" and
                # buried a clean Ctrl-C under two tracebacks.
                yield item, None, error
            else:
                yield item, result, None
            nxt = next(remaining, None)
            if nxt is not None:
                pending.append((nxt, pool.submit(function, nxt)))
    finally:
        # On any exit — normal, Ctrl-C, or the consumer closing this generator — drop
        # work that has not started. The executor's shutdown otherwise waits for every
        # queued task, so Ctrl-C stalled for a whole window of images before exiting.
        for _item, future in pending:
            future.cancel()
