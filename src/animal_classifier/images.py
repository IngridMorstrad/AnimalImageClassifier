"""Decode, the one coordinate frame, sha256, blur and crop (DESIGN.md §5.2, §5.3).

Everything downstream of this module — the detector, ``decide.py``, the catalog,
the GUI overlay — assumes a single geometric frame, and this module is where that
frame is established, exactly once:

    ``PIL.Image.open`` → ``ImageOps.exif_transpose`` → ``convert("RGB")``

**Invariant I9.** :attr:`DecodedImage.width` / :attr:`DecodedImage.height` are the
size *after* ``exif_transpose``, and so is every coordinate and ``area_frac``
persisted anywhere. An orientation-6 file whose stored raster is 4000×3000 is
reported here as 3000×4000, and the raw stored dimensions exist nowhere in the
system — not in a variable that outlives :func:`decode`, not in the catalog, not
in an API response. One conversion at the entry point is what makes it impossible
for a later layer to be in a different frame.

**The source card is read-only.** :func:`open_source` is the only reader of source
bytes in the whole program and it opens ``"rb"``. Both :func:`decode` and
:func:`sha256_file` go through it, so the read-only guarantee is structural rather
than a convention each caller has to remember.

**No area floor, here of all places.** §5.7's ``dominance_ratio`` is the only size
gate in the system, and :func:`crop` is the most tempting place to smuggle in a
second one: a box too small to classify *looks* like a box worth discarding. It is
not discarded. A box under :data:`DEGENERATE_MIN_SIDE` px on a side after clipping
is reported as :attr:`Crop.degenerate`, which excludes it from *classification*
only — the caller still counts it as an animal for dominance, and a degenerate box
that wins dominance yields the label ``unknown`` (§5.7, E26). Dropping it would be
an area floor by the back door.

Failure contract: every per-image failure raised here is an
:class:`ImageDecodeError` carrying the :class:`~animal_classifier.scan.SkipReason`
the caller must record, so one ``except`` clause per image covers decode, and the
run's exit class follows from :func:`animal_classifier.scan.exit_code_for` without
this module knowing anything about exit codes.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Final

import cv2
import numpy as np
import pillow_heif
from PIL import Image, ImageOps

from .errors import ConfigError, DecodeError
from .scan import RAW_EXTENSIONS, SkipReason

log = logging.getLogger(__name__)

# HEIC decode *and* encode. The encoder is not incidental: it is what lets
# `make_e2e_fixtures.py` build the HEIC fixture E16 needs, so registering the
# opener at import keeps the tool and its own fixtures in one code path.
pillow_heif.register_heif_opener()

#: 400 MP (§5.2). Pillow's own bomb guard warns above this and raises above twice
#: it; we do not rely on either — :func:`decode` refuses an oversized raster from
#: the header, before any pixels are decoded. Assigning it anyway means a huge
#: image that somehow reaches ``load()`` still raises rather than allocating.
MAX_IMAGE_PIXELS: Final = 400_000_000
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS

#: sha256 streams the **raw file bytes** in 1 MiB chunks, never the decoded
#: pixels, so a hash is stable across Pillow/libheif versions and identifies
#: byte-for-byte duplicates exactly (§5.2).
SHA256_CHUNK_BYTES: Final = 1 << 20

#: The long edge the blur score is measured at, when the image is larger.
BLUR_REFERENCE_EDGE: Final = 512

#: A box with a side below this many pixels after clipping cannot be resized to a
#: model's input size meaningfully (§5.3).
DEGENERATE_MIN_SIDE: Final = 2

#: The literal §5.7 compares against and ``boxes.species_status`` stores. Exported
#: so no caller has to spell it, because a typo would silently make a degenerate
#: box classifiable.
SPECIES_STATUS_DEGENERATE: Final = "degenerate"

#: EXIF tag ids, used numerically because ``ExifTags.Base`` names differ between
#: the IFDs and the numeric ids never move.
_EXIF_IFD: Final = 0x8769
_GPS_IFD: Final = 0x8825
_TAG_DATETIME_ORIGINAL: Final = 0x9003
_TAG_GPS_LATITUDE_REF: Final = 0x0001
_TAG_GPS_LATITUDE: Final = 0x0002
_TAG_GPS_LONGITUDE_REF: Final = 0x0003
_TAG_GPS_LONGITUDE: Final = 0x0004

#: EXIF's own datetime spelling, which is *not* ISO-8601 (§5.2 stores ISO-8601).
_EXIF_DATETIME_FORMAT: Final = "%Y:%m:%d %H:%M:%S"

_RAW_EXTRA_HINT: Final = (
    "RAW decode needs the optional `raw` extra: "
    "uv pip install -e '.[raw]' (or drop --raw)"
)


class ImageDecodeError(DecodeError):
    """One image failed, and this is the ``skipped.reason`` to record for it.

    A :class:`~animal_classifier.errors.DecodeError` on purpose — recoverable, the
    run continues and exits 4 — but it carries the reason so the caller does not
    have to re-derive it from the message. ``reason`` is one of §5.2's two
    decode-time reasons (``too_large_pixels``, ``decode_error``) or ``unreadable``
    when the file disappeared or lost its permissions between the walk and the
    read.
    """

    def __init__(self, message: str, *, reason: SkipReason) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True, slots=True)
class DecodedImage:
    """A decoded image in the one frame, plus the optional EXIF facts.

    ``width``/``height`` are ``exif_transpose(im).size`` (invariant I9) and match
    ``image.size`` by construction. The three EXIF fields are optional in the
    contract: absent metadata stores ``NULL`` and is **never** fabricated, and an
    unparseable or out-of-range value is dropped with a ``WARNING`` rather than
    guessed (§5.2, §10.2).
    """

    path: Path
    image: Image.Image
    width: int
    height: int
    exif_datetime: str | None
    gps_lat: float | None
    gps_lon: float | None


@dataclass(frozen=True, slots=True)
class Blur:
    """A blur score and the long edge it was actually measured at.

    Two scores are comparable only with their reference edge in view, which is why
    ``blur_ref_edge`` is a stored column and not a constant: ``blur_threshold`` is
    calibrated at the :data:`BLUR_REFERENCE_EDGE` px reference, and images already
    smaller than that are scored **natively** (never upscaled), so their
    ``ref_edge`` is their own long edge.
    """

    score: float
    ref_edge: int


@dataclass(frozen=True, slots=True)
class Crop:
    """The pixels for one detection box, or the degenerate verdict for it.

    ``image`` is ``None`` if and only if ``degenerate`` is true, so the two can
    never disagree. ``region`` is the integer box actually cropped — margin
    applied and clipped to the frame — in the same frame as everything else.
    """

    region: tuple[int, int, int, int]
    image: Image.Image | None
    degenerate: bool

    @property
    def species_status(self) -> str | None:
        """``"degenerate"`` for a degenerate crop, else ``None``.

        ``None`` means *not classified yet*, not *fine*: the classifier fills the
        real value in. Only the degenerate case is decided here, because only the
        geometry is known here.
        """
        return SPECIES_STATUS_DEGENERATE if self.degenerate else None


@contextlib.contextmanager
def open_source(path: Path) -> Iterator[BinaryIO]:
    """Open a source file for reading — the only source reader in the program.

    ``"rb"``: no truncation, no append, no ``r+``. Every other function in this
    module reads through here, so "the card is never written to" is a property of
    one line of code instead of a rule spread across the pipeline.

    ``OSError`` is translated to :class:`ImageDecodeError` with reason
    ``unreadable``. The walk already stat-ed this file, so a failure here means it
    changed underneath us (unplugged card, permissions revoked); that is one
    image's problem, not the run's.
    """
    try:
        handle = path.open("rb")
    except OSError as error:
        raise ImageDecodeError(
            f"cannot read {path}: {type(error).__name__}: {error.strerror or error}",
            reason=SkipReason.UNREADABLE,
        ) from error
    try:
        yield handle
    finally:
        handle.close()


def sha256_file(path: Path) -> str:
    """Hex sha256 of the file's raw bytes, streamed in 1 MiB chunks.

    Streaming rather than ``read()`` because a 500 MB file is within
    ``max_file_bytes`` and must not become 500 MB of resident memory.
    """
    digest = hashlib.sha256()
    with open_source(path) as handle:
        try:
            while chunk := handle.read(SHA256_CHUNK_BYTES):
                digest.update(chunk)
        except OSError as error:
            raise ImageDecodeError(
                f"cannot read {path}: {type(error).__name__}: "
                f"{error.strerror or error}",
                reason=SkipReason.UNREADABLE,
            ) from error
    return digest.hexdigest()


def decode(path: Path, *, raw: bool = False) -> DecodedImage:
    """Decode ``path`` into the one frame and read its optional EXIF facts.

    Raises :class:`ImageDecodeError` with reason ``too_large_pixels`` for a raster
    above :data:`MAX_IMAGE_PIXELS` — judged from the header, so a decompression
    bomb is refused without ever being allocated — and ``decode_error`` for
    truncated, corrupt or unidentifiable data. Both are per-image outcomes; the
    run continues.

    ``raw`` mirrors the ``--raw`` flag. A RAW extension arriving with ``raw``
    false is an invariant break rather than a user error (the walk should have
    skipped it as ``raw_not_enabled``), so it fails loudly instead of guessing a
    decoder.
    """
    if path.suffix.lower() in RAW_EXTENSIONS:
        if not raw:
            raise ConfigError(
                f"{path} is a RAW file and RAW decode is not enabled; "
                "pass --raw to ingest RAW files"
            )
        return _decode_raw(path)
    return _decode_pillow(path)


def _decode_pillow(path: Path) -> DecodedImage:
    """The §5.2 chain, with every Pillow failure mode mapped to a skip reason."""
    with open_source(path) as handle:
        try:
            with Image.open(handle) as opened:
                _refuse_oversized(path, opened.size)
                # EXIF is read from the *original* image: exif_transpose strips
                # the orientation tag from its result, and reading before the
                # transpose keeps DateTimeOriginal/GPS independent of it.
                exif_datetime, gps_lat, gps_lon = _read_exif(path, opened)
                transposed = ImageOps.exif_transpose(opened)
                image = transposed.convert("RGB")
        except Image.DecompressionBombError as error:
            raise ImageDecodeError(
                f"{path} exceeds the {MAX_IMAGE_PIXELS} pixel cap: {error}",
                reason=SkipReason.TOO_LARGE_PIXELS,
            ) from error
        except ImageDecodeError:
            raise
        except (OSError, ValueError, SyntaxError) as error:
            # UnidentifiedImageError and truncated-file errors are both OSError
            # subclasses; ValueError/SyntaxError come out of individual plugins.
            raise ImageDecodeError(
                f"cannot decode {path}: {type(error).__name__}: {error}",
                reason=SkipReason.DECODE_ERROR,
            ) from error

    return _finish(path, image, exif_datetime, gps_lat, gps_lon)


def _decode_raw(path: Path) -> DecodedImage:
    """RAW via ``rawpy.imread(...).postprocess()``, behind ``--raw`` only.

    ``rawpy`` is an optional extra, so its absence is a configuration failure with
    the install command in the message (exit 3, §10.1) — not a per-image skip,
    because every RAW file on the card would hit it.

    RAW carries no Pillow-readable EXIF here, so the three optional fields are
    ``NULL``. ``postprocess()`` applies the camera's own flip, which puts the
    array in the same frame ``exif_transpose`` would have produced.
    """
    try:
        import rawpy
    except ModuleNotFoundError as error:
        raise ConfigError(f"--raw was given but {_RAW_EXTRA_HINT}") from error

    with open_source(path) as handle:
        try:
            with rawpy.imread(handle) as source:
                array = source.postprocess()
        except Exception as error:  # rawpy raises its own LibRawError family
            raise ImageDecodeError(
                f"cannot decode RAW {path}: {type(error).__name__}: {error}",
                reason=SkipReason.DECODE_ERROR,
            ) from error

    height, width = array.shape[:2]
    _refuse_oversized(path, (width, height))
    log.debug("%s: RAW decode carries no EXIF; captured_at and GPS are NULL", path)
    return _finish(path, Image.fromarray(array).convert("RGB"), None, None, None)


def _finish(
    path: Path,
    image: Image.Image,
    exif_datetime: str | None,
    gps_lat: float | None,
    gps_lon: float | None,
) -> DecodedImage:
    """Assemble the result, refusing a zero-pixel raster.

    A 0-px side survives some plugins and would divide by zero in ``area_frac``,
    so it is refused here where it is still one image's problem.
    """
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageDecodeError(
            f"{path} decoded to a {width}x{height} raster",
            reason=SkipReason.DECODE_ERROR,
        )
    return DecodedImage(
        path=path,
        image=image,
        width=width,
        height=height,
        exif_datetime=exif_datetime,
        gps_lat=gps_lat,
        gps_lon=gps_lon,
    )


def _refuse_oversized(path: Path, size: tuple[int, int]) -> None:
    """Refuse a raster above the 400 MP cap from its header dimensions."""
    width, height = size
    pixels = width * height
    if pixels > MAX_IMAGE_PIXELS:
        raise ImageDecodeError(
            f"{path} is {width}x{height} = {pixels} pixels, over the "
            f"{MAX_IMAGE_PIXELS} pixel cap",
            reason=SkipReason.TOO_LARGE_PIXELS,
        )


def _read_exif(
    path: Path, image: Image.Image
) -> tuple[str | None, float | None, float | None]:
    """``(exif_datetime, gps_lat, gps_lon)``, any of which may be ``None``.

    Absent EXIF is the normal case for a screenshot or an exported JPEG, so it is
    ``NULL`` and not an error. A *present but unusable* value is dropped with a
    ``WARNING``, never repaired: an invented capture time would silently corrupt
    the GUI's date filter, and an invented coordinate would send ``ebird_enrich``
    to the wrong continent.
    """
    try:
        exif = image.getexif()
    except (OSError, ValueError, SyntaxError) as error:
        log.warning("%s: EXIF block unreadable (%s); metadata is NULL", path, error)
        return None, None, None
    if not exif:
        return None, None, None

    return (
        _read_exif_datetime(path, exif),
        *_read_gps(path, exif),
    )


def _read_exif_datetime(path: Path, exif: Image.Exif) -> str | None:
    """``DateTimeOriginal`` → ISO-8601, or ``None``.

    Only ``DateTimeOriginal`` (the moment of capture) is accepted. ``DateTime``
    is the file's last-modified time, which an editing tool rewrites, so falling
    back to it would fabricate a capture time that looks authoritative.
    """
    raw = exif.get_ifd(_EXIF_IFD).get(_TAG_DATETIME_ORIGINAL)
    if raw is None:
        return None
    text = str(raw).strip().rstrip("\x00").strip()
    if not text:
        return None
    try:
        moment = datetime.strptime(text, _EXIF_DATETIME_FORMAT)
    except ValueError:
        log.warning(
            "%s: DateTimeOriginal %r is not %s; exif_datetime is NULL",
            path,
            text,
            _EXIF_DATETIME_FORMAT,
        )
        return None
    return moment.isoformat()


def _read_gps(path: Path, exif: Image.Exif) -> tuple[float | None, float | None]:
    """GPS latitude/longitude as signed decimal degrees, or ``(None, None)``.

    Both or neither: a latitude without a longitude is not a location, so a
    half-present pair is dropped rather than stored as a column the GUI would
    render as a place. Out-of-range values are dropped with a ``WARNING``
    (§10.2), because a clamped coordinate is a plausible-looking lie.
    """
    gps = exif.get_ifd(_GPS_IFD)
    if not gps:
        return None, None

    lat = _to_degrees(
        path, gps.get(_TAG_GPS_LATITUDE), gps.get(_TAG_GPS_LATITUDE_REF), "N", "S"
    )
    lon = _to_degrees(
        path, gps.get(_TAG_GPS_LONGITUDE), gps.get(_TAG_GPS_LONGITUDE_REF), "E", "W"
    )
    if lat is None or lon is None:
        if lat is not None or lon is not None:
            log.warning("%s: GPS has only one of lat/lon; both are NULL", path)
        return None, None
    if not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        log.warning(
            "%s: GPS (%s, %s) is out of range; both are NULL", path, lat, lon
        )
        return None, None
    return lat, lon


def _to_degrees(
    path: Path, value: object, ref: object, positive_ref: str, negative_ref: str
) -> float | None:
    """One EXIF ``(degrees, minutes, seconds)`` rational triple → signed float.

    The hemisphere letter decides the **sign**, so it is validated as strictly as
    the magnitude: it must decode to ``positive_ref`` or ``negative_ref``, and
    anything else — absent, empty, numeric, or unrecognised — drops the
    coordinate with a ``WARNING`` instead of falling through to the positive
    hemisphere. A defaulted sign is the worst possible EXIF lie for this tool: it
    puts a Serengeti photo (``S``) north of the equator, which is not a slightly
    wrong location but a confident wrong one, and it is what would send
    ``ebird_enrich`` to the wrong continent (§10.2's rule that a plausible-looking
    value is never invented).

    ``ref`` arrives as ``bytes`` more often than the spec suggests:
    ``GPSLatitudeRef`` is ASCII(2), but firmware that declares it UNDEFINED(7)
    makes Pillow yield ``b"S"``. That is why the letter is extracted by
    :func:`_gps_ref_letter` and not by ``str(ref)`` — ``str(b"S")`` is ``"b'S'"``,
    which starts with neither hemisphere letter and so read as North.
    """
    if value is None:
        return None
    letter = _gps_ref_letter(ref)
    if letter not in (positive_ref, negative_ref):
        log.warning(
            "%s: GPS ref %r is neither %s nor %s, so the hemisphere is unknown; "
            "coordinate is NULL",
            path,
            ref,
            positive_ref,
            negative_ref,
        )
        return None
    try:
        degrees, minutes, seconds = (float(part) for part in value)  # type: ignore[union-attr]
    except (TypeError, ValueError, ZeroDivisionError) as error:
        log.warning("%s: GPS value %r is unusable (%s); NULL", path, value, error)
        return None
    decimal = degrees + minutes / 60.0 + seconds / 3600.0
    return -decimal if letter == negative_ref else decimal


def _gps_ref_letter(ref: object) -> str | None:
    """The hemisphere letter from an EXIF ref tag, or ``None`` if there isn't one.

    Accepts the ``str`` the spec asks for and the ``bytes`` an UNDEFINED-typed tag
    produces, strips EXIF's trailing NUL, and returns a single upper-case letter.
    Returns ``None`` for every other shape so the caller drops the coordinate
    rather than guessing a hemisphere.
    """
    if isinstance(ref, bytes):
        try:
            text = ref.decode("ascii")
        except UnicodeDecodeError:
            return None
    elif isinstance(ref, str):
        text = ref
    else:
        return None
    return text.strip().rstrip("\x00").strip().upper()[:1] or None


def measure_blur(image: Image.Image) -> Blur:
    """Variance of the Laplacian on grayscale, at a bounded reference edge.

    **Downscaled only** — never upscaled. Interpolating a small image upward
    depresses Laplacian variance, which would push small *sharp* animal-free
    photos below ``blur_threshold`` and file them as ``junk`` (E11 pins exactly
    this). So an image whose long edge is already at or below
    :data:`BLUR_REFERENCE_EDGE` is scored natively and reports its own long edge
    as ``ref_edge``.

    ``blur_threshold`` (default 100.0; lower is blurrier) is calibrated at the
    512 px reference. A score carried at a smaller ``ref_edge`` is compared with
    that threshold as-is — the alternative, upscaling to 512, is the bias this
    function exists to avoid — which is why ``blur_ref_edge`` is persisted beside
    ``blur_score`` and shown in the GUI.
    """
    gray = image.convert("L")
    gray = _downscale_to_reference(gray)
    ref_edge = max(gray.size)
    laplacian = cv2.Laplacian(np.asarray(gray), cv2.CV_64F)
    return Blur(score=float(laplacian.var()), ref_edge=int(ref_edge))


def _downscale_to_reference(gray: Image.Image) -> Image.Image:
    """Scale the long edge to exactly :data:`BLUR_REFERENCE_EDGE`, or leave it.

    The long edge is set exactly, rather than derived from a rounded scale
    factor, so ``ref_edge`` is 512 for every downscaled image and two scores are
    trivially comparable.
    """
    width, height = gray.size
    long_edge = max(width, height)
    if long_edge <= BLUR_REFERENCE_EDGE:
        return gray
    scale = BLUR_REFERENCE_EDGE / long_edge
    if width >= height:
        size = (BLUR_REFERENCE_EDGE, max(1, round(height * scale)))
    else:
        size = (max(1, round(width * scale)), BLUR_REFERENCE_EDGE)
    return gray.resize(size, Image.Resampling.BILINEAR)


def crop(
    image: Image.Image,
    box: tuple[float, float, float, float],
    *,
    crop_margin: float,
) -> Crop:
    """Expand ``box`` by ``crop_margin``, clip to the frame, and cut it out.

    ``box`` is ``(x0, y0, x1, y1)`` in the frame :func:`decode` established, i.e.
    the detector's own output coordinates. The margin is a fraction of the box's
    **own** width and height, applied to each side, so a wide box gets a wide
    margin and framing is scale-invariant.

    ``crop_margin`` is a **required keyword** with no default. Training and
    inference must frame crops identically, and the value that governs is the
    one recorded in the model artifact — a default here would let a run's config
    value silently win and skew inference against the model's training framing
    (design review finding 7). Callers pass ``artifact.crop_margin``.

    A box under :data:`DEGENERATE_MIN_SIDE` px on a side after clipping comes
    back with ``degenerate=True`` and ``image=None``: unclassifiable, but the
    caller must still count it as an animal for dominance (§5.3, §5.7, E26). A box
    that misses the frame entirely is that same degenerate case with a zero-extent
    region on the frame edge — ``region`` is always ordered, never inverted.

    Raises ``ValueError`` for a non-finite coordinate, which is a detector
    malfunction rather than a per-image failure; see the comment at the check.
    """
    if not 0.0 <= crop_margin <= 0.5:
        raise ConfigError(
            f"crop_margin must be in [0.0, 0.5], got {crop_margin!r}"
        )

    coordinates = tuple(float(coordinate) for coordinate in box)
    if not all(np.isfinite(coordinates)):
        # Deliberately *not* an ImageDecodeError: a NaN or inf coordinate is a
        # detector malfunction, not one bad photo, and the per-image `except`
        # would record a single skip while every other frame in the run kept
        # receiving garbage. Left unchecked it is silent rather than loud — NaN
        # loses every comparison, so `max`/`min` clipping returns the *frame*
        # bounds and a NaN box becomes a full-frame crop that is classified as
        # an animal and contributes a meaningless `area_frac` to the dominance
        # rule (§5.7), the one decision the whole design turns on.
        raise ValueError(
            f"detection box {box!r} has a non-finite coordinate; "
            "boxes must be finite pixel coordinates in the decoded frame"
        )

    width, height = image.size
    x0, x1 = sorted((coordinates[0], coordinates[2]))
    y0, y1 = sorted((coordinates[1], coordinates[3]))
    margin_x = (x1 - x0) * crop_margin
    margin_y = (y1 - y0) * crop_margin

    # Expand, then clip, in floats — this is the extent §5.3 measures. Each edge
    # is clamped into the frame interval rather than bounded on one side only:
    # clamping is monotonic, so `left_f <= right_f` survives a box that lies
    # entirely outside the frame. Bounding one side each (`max(0, ...)` for the
    # near edge, `min(width, ...)` for the far one) inverted the region in that
    # case — a box at x=600 on a 500 px frame yielded x0=592, x1=500 — and that
    # negative-extent region was still returned and persisted to `boxes`, where
    # the GUI would try to draw it.
    left_f = _clamp(x0 - margin_x, 0.0, float(width))
    right_f = _clamp(x1 + margin_x, 0.0, float(width))
    top_f = _clamp(y0 - margin_y, 0.0, float(height))
    bottom_f = _clamp(y1 + margin_y, 0.0, float(height))

    # Outward rounding (floor the near edges, ceil the far ones) so the margin is
    # never rounded away. The degeneracy test below deliberately uses the *float*
    # extent instead: a 1 px box rounds outward to 3 px, and judging the rounded
    # region would hand the classifier three pixels of interpolation and call a
    # sub-2px detection classifiable.
    region = (
        int(np.floor(left_f)),
        int(np.floor(top_f)),
        int(np.ceil(right_f)),
        int(np.ceil(bottom_f)),
    )

    if (
        right_f - left_f < DEGENERATE_MIN_SIDE
        or bottom_f - top_f < DEGENERATE_MIN_SIDE
    ):
        log.debug(
            "box %s crops to %s (< %d px a side): degenerate, kept for dominance",
            box,
            region,
            DEGENERATE_MIN_SIDE,
        )
        return Crop(region=region, image=None, degenerate=True)

    return Crop(region=region, image=image.crop(region), degenerate=False)


def _clamp(value: float, low: float, high: float) -> float:
    """``value`` confined to ``[low, high]``.

    Monotonic, which is the property :func:`crop` relies on: clamping both edges
    of an interval into the same range cannot reorder them.
    """
    return min(max(value, low), high)
