"""MegaDetector v5a, loaded and run exactly as recon proved (DESIGN.md §5.4).

This is the real detector: the 280 MB yolov5 checkpoint that finds animals,
people and vehicles in a photograph. Everything here is pinned to what
``scripts/probe_md_checkpoint.py`` and ``probe_md_inference.py`` verified against
the locked environment — the alias shim, ``weights_only=False``, the ``ema``
preference, the letterbox/NMS/scale_boxes pipeline and the ``stride=64`` — so this
module is those probes turned into a `Detector`.

**Why ``weights_only=False`` is acceptable.** The checkpoint pickles a live
``models.yolo.DetectionModel``, so it cannot be loaded with the safe
``weights_only=True``. That is only tolerable because the file's provenance is
pinned: :meth:`MegaDetector.load` verifies the byte length *and* the sha256 against
the constants in ``RECON.md`` before unpickling, and refuses with the download URL
on any mismatch. An arbitrary ``.pt`` cannot be substituted.

**The alias shim.** The checkpoint was pickled from the upstream yolov5 repo, whose
packages sit at the top level (``models.yolo``, ``utils.*``). The PyPI ``yolov5``
distribution namespaces the same modules under ``yolov5.*``, so the unpickler needs
the original names aliased onto it — otherwise ``torch.load`` raises
``ModuleNotFoundError: models`` (§10.1's "re-sync with uv sync --frozen" row).

**One coordinate frame, still (I9).** ``letterbox`` resizes onto the inference
tensor, and ``scale_boxes`` maps the boxes back onto ``exif_transpose(im).size`` —
the same frame :func:`animal_classifier.images.decode` produced — never onto the raw
stored raster. ``box_from_pixels`` then normalises ``area_frac`` against that frame,
so every box this detector emits is in the one frame the rest of the system assumes.
"""

from __future__ import annotations

import hashlib
import importlib
import logging
import sys
import threading
from pathlib import Path
from typing import Any, Final

import numpy as np

from ..errors import EXIT_CONFIG, AssetError
from ..images import DecodedImage
from .base import Box, BoxClass, box_from_pixels

log = logging.getLogger(__name__)

#: The exact checkpoint this code was written against (RECON.md, §5.4). A different
#: size or hash is a different file, and since the load unpickles arbitrary objects
#: it is refused rather than trusted.
EXPECTED_BYTES: Final = 280_766_885
EXPECTED_SHA256: Final = (
    "94e88fe97c8050f2e3d0cc4cb4f64729d639d74312dcbe2f74f8eecd3b01b276"
)
DOWNLOAD_URL: Final = (
    "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
)

#: Top-level package names the pickled checkpoint refers to, aliased onto the
#: PyPI ``yolov5.*`` packages so the unpickler can resolve them.
_YOLOV5_ALIASES: Final = ("models", "utils")

#: The checkpoint's own stride (``model.stride = [8,16,32,64]``); letterbox pads to
#: a multiple of it. Pinned because a wrong stride silently shifts every box.
_STRIDE: Final = 64

#: Read the file in 1 MiB chunks to hash it without a 280 MB resident copy.
_HASH_CHUNK: Final = 1 << 20

_ALIAS_LOCK = threading.Lock()
_ALIASES_INSTALLED = False


def _install_yolov5_aliases() -> None:
    """Alias ``models``/``utils`` onto ``yolov5.*`` for the unpickler. Idempotent."""
    global _ALIASES_INSTALLED
    with _ALIAS_LOCK:
        if _ALIASES_INSTALLED:
            return
        for name in _YOLOV5_ALIASES:
            try:
                package = importlib.import_module(f"yolov5.{name}")
            except ModuleNotFoundError as error:
                raise AssetError(
                    "the yolov5 package is not importable, so the MegaDetector "
                    "checkpoint cannot be unpickled; re-sync with `uv sync --frozen`"
                ) from error
            sys.modules.setdefault(name, package)
        _ALIASES_INSTALLED = True


class MegaDetector:
    """MegaDetector v5a as a :class:`~animal_classifier.detect.base.Detector`.

    Construct with :meth:`load`, which does the verification and unpickling; the
    constructor itself just holds an already-loaded model, so a caller that has one
    (a test, a warm cache) is not forced back through the file checks.
    """

    def __init__(
        self,
        model: Any,
        *,
        model_id: str,
        confidence: float,
        iou: float,
        image_size: int,
        max_det: int,
    ) -> None:
        self._model = model
        self._model_id = model_id
        self._confidence = confidence
        self._iou = iou
        self._image_size = image_size
        self._max_det = max_det
        # Serialise inference: the pipeline runs the detector on the main thread
        # (§9), but a shared model must never be entered concurrently even so.
        self._lock = threading.Lock()

    @classmethod
    def load(
        cls,
        weights: Path,
        *,
        confidence: float,
        iou: float,
        image_size: int,
        max_det: int,
        jobs: int | None = None,
    ) -> MegaDetector:
        """Verify, unpickle and prepare the checkpoint for inference (§5.4).

        Fatal, with the download URL, on a missing file or a size/sha256 mismatch —
        the load unpickles arbitrary objects, so an unverified file is never
        trusted (§10.1). ``jobs`` sets ``torch.set_num_threads`` when given, matching
        §9's "torch parallelizes internally" note.
        """
        import torch  # noqa: PLC0415 - heavy import, only when the real detector runs

        _verify_asset(weights)
        _install_yolov5_aliases()

        if jobs is not None and jobs > 0:
            torch.set_num_threads(jobs)

        try:
            checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as error:
            raise AssetError(
                f"cannot unpickle {weights} ({error}); the yolov5 package must be "
                "installed — re-sync with `uv sync --frozen`"
            ) from error
        except Exception as error:  # noqa: BLE001 - any unpickle failure is fatal/config
            raise AssetError(
                f"cannot load MegaDetector checkpoint {weights}: "
                f"{type(error).__name__}: {error}. Re-download from {DOWNLOAD_URL}"
            ) from error

        # `ema` weights are preferred over `model` when present (standard for yolov5).
        raw_model = checkpoint.get("ema") or checkpoint.get("model")
        if raw_model is None:
            raise AssetError(
                f"MegaDetector checkpoint {weights} has neither 'ema' nor 'model' "
                f"weights; it is not the expected file — re-download from {DOWNLOAD_URL}"
            )
        model = raw_model.float().eval()
        model_id = f"megadetector:v5a:{EXPECTED_SHA256[:12]}"
        log.info("loaded MegaDetector v5a from %s (classes %s)", weights, model.names)
        return cls(
            model,
            model_id=model_id,
            confidence=confidence,
            iou=iou,
            image_size=image_size,
            max_det=max_det,
        )

    @property
    def model_id(self) -> str:
        return self._model_id

    def detect(self, decoded: DecodedImage) -> tuple[Box, ...]:
        """Boxes for one image, in the EXIF-transposed frame (§5.4).

        ``letterbox`` → CHW float32 ``/255`` → ``model(x)[0]`` →
        ``non_max_suppression`` → ``scale_boxes`` back onto the decoded frame, then
        each surviving row becomes a :class:`Box` with ``area_frac`` normalised
        against that frame. A detector failure (OOM, malformed tensor) propagates as
        the exception it is; the pipeline records that one image as ``failed`` and
        the run continues (§10.1), so this method does not swallow anything.
        """
        import torch  # noqa: PLC0415
        from yolov5.utils.augmentations import letterbox  # noqa: PLC0415
        from yolov5.utils.general import non_max_suppression, scale_boxes  # noqa: PLC0415

        # Pillow RGB -> HWC uint8 array. decode() already applied exif_transpose and
        # convert("RGB"), so this array is in the one frame.
        original = np.asarray(decoded.image)
        letterboxed, ratio, pad = letterbox(
            original, new_shape=self._image_size, stride=_STRIDE, auto=False
        )
        # HWC -> CHW, contiguous, float32 in [0, 1].
        tensor = torch.from_numpy(
            np.ascontiguousarray(letterboxed.transpose(2, 0, 1))
        ).float().div_(255.0).unsqueeze(0)

        with self._lock, torch.inference_mode():
            prediction = self._model(tensor)[0]
            detections = non_max_suppression(
                prediction,
                conf_thres=self._confidence,
                iou_thres=self._iou,
                max_det=self._max_det,
            )[0]

        if detections is None or not len(detections):
            return ()

        # Map boxes from the letterboxed tensor back onto the original frame.
        detections = detections.clone()
        detections[:, :4] = scale_boxes(
            letterboxed.shape[:2], detections[:, :4], original.shape[:2]
        ).round()

        boxes: list[Box] = []
        names = self._model.names
        for *xyxy, conf, cls_index in detections.tolist():
            boxes.append(
                box_from_pixels(
                    box_class=_class_name(names, int(cls_index)),
                    conf=float(conf),
                    x0=xyxy[0],
                    y0=xyxy[1],
                    x1=xyxy[2],
                    y1=xyxy[3],
                    frame_width=decoded.width,
                    frame_height=decoded.height,
                )
            )
        log.debug("%s: %d detection(s)", decoded.path, len(boxes))
        return tuple(boxes)


def _class_name(names: Any, index: int) -> BoxClass:
    """Map a class index to its :class:`BoxClass` via the checkpoint's own names.

    The checkpoint carries ``names = ['animal', 'person', 'vehicle']``; using it
    rather than a hard-coded order means a reordered future checkpoint cannot
    silently relabel every box. An index or name outside the three is fatal, not
    guessed.
    """
    try:
        raw = names[index]
    except (KeyError, IndexError, TypeError) as error:
        raise AssetError(
            f"MegaDetector produced class index {index}, absent from the "
            f"checkpoint's names {names}; the checkpoint is not the expected v5a file"
        ) from error
    try:
        return BoxClass(str(raw))
    except ValueError as error:
        raise AssetError(
            f"MegaDetector class {raw!r} is not one of "
            f"{[str(member) for member in BoxClass]}; unexpected checkpoint"
        ) from error


def _verify_asset(weights: Path) -> None:
    """Size + sha256 gate before any unpickling (§5.4). Fatal on any mismatch."""
    if not weights.exists():
        raise AssetError(
            f"detector weights not found at {weights}. Download MegaDetector v5a "
            f"({EXPECTED_BYTES} bytes) from {DOWNLOAD_URL}",
            exit_code=EXIT_CONFIG,
        )
    if not weights.is_file():
        raise AssetError(f"detector weights at {weights} is not a regular file")

    size = weights.stat().st_size
    if size != EXPECTED_BYTES:
        raise AssetError(
            f"detector weights at {weights} are {size} bytes, expected "
            f"{EXPECTED_BYTES}; the file is truncated or not MegaDetector v5a — "
            f"re-download from {DOWNLOAD_URL}"
        )

    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != EXPECTED_SHA256:
        raise AssetError(
            f"detector weights at {weights} have sha256 {actual}, expected "
            f"{EXPECTED_SHA256}; the file does not match MegaDetector v5a — "
            f"re-download from {DOWNLOAD_URL}"
        )
    log.debug("verified detector weights %s (%d bytes, sha256 ok)", weights, size)
