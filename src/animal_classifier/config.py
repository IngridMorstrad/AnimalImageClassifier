"""Configuration: layered resolution, total validation, one frozen ``Config``.

Layering, highest wins (DESIGN.md §3):

    CLI flag  →  environment variable (``ANIMAL_CLASSIFIER_*``)  →  TOML file  →  built-in default

Validation is **total** and happens once, before any file is touched (§10.2). Every
failure is a :class:`~animal_classifier.errors.ConfigError` or
:class:`~animal_classifier.errors.AssetError` (exit 3) naming the offending key, the
layer it came from, and — for paths — the *resolved absolute* path.

Two properties of this module are load-bearing and must not be softened:

* **There is no ``min_box_area``, ``min_box_area_frac``, ``min_animal_area`` or
  absolute box-area floor under any name.** ``dominance_ratio`` is the only size
  gate (invariant I2). An unknown TOML key is fatal, and that check is what stops
  such a floor from ever being reintroduced by configuration.
  ``max_file_bytes`` is a whole-file *scan* cap; it never sees a detection box.
* **No required value is ever silently defaulted** (invariant I7). There is no
  ``dict.get(key, fallback)`` for a required key and no bare ``except`` here.

``detector_max_det`` is a configuration key even though DESIGN.md §5.4 quotes the
value inline (100): the detector must read every NMS parameter from ``Config``
rather than hard-coding it.
"""

from __future__ import annotations

import enum
import logging
import os
import tomllib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from .errors import AssetError, ConfigError

log = logging.getLogger(__name__)

ENV_PREFIX: Final = "ANIMAL_CLASSIFIER_"
EBIRD_API_KEY_ENV: Final = f"{ENV_PREFIX}EBIRD_API_KEY"
HOSTED_BIRD_API_ENVS: Final = (
    f"{ENV_PREFIX}HOSTED_BIRD_API_URL",
    f"{ENV_PREFIX}HOSTED_BIRD_API_KEY",
)


class Mode(enum.StrEnum):
    """How an accepted image is filed into the output tree."""

    COPY = "copy"
    LINK = "link"
    HARDLINK = "hardlink"


class DetectorKind(enum.StrEnum):
    MEGADETECTOR = "megadetector"
    SCRIPTED = "scripted"


class BirdProvider(enum.StrEnum):
    OWN_BIRD_HEAD = "own_bird_head"
    EBIRD_ENRICH = "ebird_enrich"
    HOSTED_BIRD_API = "hosted_bird_api"


class Device(enum.StrEnum):
    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class Format(enum.StrEnum):
    """An eligible *family*, not an extension. RAW is deliberately not a family."""

    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    HEIC = "heic"


class Command(enum.StrEnum):
    """Which entry point is resolving config — decides which assets are required."""

    CLASSIFY = "classify"
    GUI = "gui"
    TRAIN = "train"
    EVAL = "eval"
    EXPORT_TRAINSET = "export-trainset"
    VERIFY = "verify"


#: Family → extensions (DESIGN.md §3.1). RAW is governed solely by ``--raw`` and
#: video extensions are always skipped, so neither appears here.
FAMILY_EXTENSIONS: Final[Mapping[Format, tuple[str, ...]]] = {
    Format.JPEG: (".jpg", ".jpeg"),
    Format.PNG: (".png",),
    Format.TIFF: (".tif", ".tiff"),
    Format.HEIC: (".heic", ".heif"),
}

DEFAULT_FORMATS: Final = (Format.JPEG, Format.PNG, Format.TIFF, Format.HEIC)

#: Built-in defaults, verbatim from DESIGN.md §3.
DEFAULTS: Final[Mapping[str, Any]] = {
    "output_root": "~/animal_pics",
    "mode": Mode.COPY,
    "dominance_ratio": 1.6,
    "min_species_confidence": 0.45,
    "detector_confidence": 0.20,
    "detector_iou": 0.45,
    "detector_image_size": 1280,
    "detector_max_det": 100,
    "crop_margin": 0.08,
    "blur_threshold": 100.0,
    "max_file_bytes": 536870912,
    "species_model": "models/species.acmodel",
    "bird_model": "models/birds.acmodel",
    "detector_weights": "models/md_v5a.0.0.pt",
    "bird_provider": BirdProvider.OWN_BIRD_HEAD,
    "jobs": 7,
    "device": Device.AUTO,
    "formats": DEFAULT_FORMATS,
}

#: The exact set of keys a TOML file may contain. Anything else is fatal, which is
#: how a reintroduced box-area floor would be caught (invariant I2).
KNOWN_TOML_KEYS: Final = frozenset(DEFAULTS)

#: The remedy quoted when a species/bird artifact is missing (design review finding
#: 11b) — and it must be a command that **actually works from a fresh clone**. The
#: previous version named ``data/manifests/coco_species.jsonl``, which only exists
#: after staging ~1 GB of COCO and running a builder, so the "remedy" was guaranteed
#: to fail with `manifest not found` for anyone who followed it. The real answer for
#: a user with their own photographs is the review loop: label a card, export, train.
TRAIN_HINT_SPECIES: Final = (
    "animal-classifier label <YOUR-CARD> --output {output_root}   "
    "# name the animals in your browser, then:\n"
    "  animal-classifier export-trainset --output {output_root} --destination mine.jsonl\n"
    "  animal-classifier train --manifest mine.jsonl --out {path} --arch efficientnet_b0\n"
    "  (no species model is required to run `classify` — animals are filed as "
    "`unknown` until you train one)"
)
TRAIN_HINT_BIRDS: Final = (
    "animal-classifier label <YOUR-CARD> --output {output_root}   "
    "# name the birds in your browser, then:\n"
    "  animal-classifier export-trainset --output {output_root} --destination birds.jsonl\n"
    "  animal-classifier train --manifest birds.jsonl --out {path} --arch convnext_nano"
)
DETECTOR_WEIGHTS_URL: Final = (
    "https://github.com/agentmorris/MegaDetector/releases/download/v5.0/md_v5a.0.0.pt"
)

#: Whether ``classify`` actually loads the species and bird artifacts yet.
#:
#: §10.1 makes a missing ``species_model``/``bird_model`` a fatal exit 3 for
#: ``classify``, and that is right the moment those artifacts are *used*. Until
#: species inference is wired the pipeline runs a pass-through classifier — every
#: animal box reaches §5.7 with no prediction and is answered ``unknown`` — so
#: demanding the artifacts up front would refuse to run over an asset nothing
#: opens, and would be satisfied by an empty file anyway, since this check tests
#: existence and readability rather than contents.
#:
#: Flip to ``True`` in the same step that lands ``classify/own_model.py``; that
#: single edit restores §10.1's row, and E22 (fail-loud config) is where the
#: restored behaviour must be asserted. Deliberately a named constant rather than a
#: commented-out block so the gap is greppable and has exactly one switch.
#:
#: Flipped to ``True`` when ``classify/own_model.py`` landed (chunk 17): a
#: configured, present ``species_model`` is loaded and run per animal crop, and
#: §10.1's required-artifact rows are in force again. A run pointing at no model
#: still works (pass-through → ``unknown``), so the pipeline tolerates an absent
#: artifact gracefully rather than refusing a useful landscape/junk triage run.
SPECIES_INFERENCE_WIRED: Final = True


def config_search_paths() -> tuple[Path, ...]:
    """Default TOML locations, in order. An absent default path is normal."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    paths: list[Path] = []
    if xdg:
        paths.append(Path(xdg) / "animal-classifier" / "config.toml")
    paths.append(Path.home() / ".config" / "animal-classifier" / "config.toml")
    # Deduplicate while preserving order: XDG_CONFIG_HOME may point at ~/.config.
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return tuple(unique)


@dataclass(frozen=True, slots=True)
class Config:
    """The resolved, validated configuration for one command invocation.

    Frozen: nothing downstream may mutate a value that has been validated and
    recorded in the ``runs`` catalog row.
    """

    command: Command
    output_root: Path
    source_root: Path | None
    species_model: Path
    bird_model: Path
    detector_weights: Path
    mode: Mode
    detector: DetectorKind
    bird_provider: BirdProvider
    formats: tuple[Format, ...]
    device: Device
    device_requested: Device
    dominance_ratio: float
    min_species_confidence: float
    detector_confidence: float
    detector_iou: float
    detector_image_size: int
    detector_max_det: int
    crop_margin: float
    blur_threshold: float
    max_file_bytes: int
    jobs: int
    limit: int | None
    port: int
    dry_run: bool
    reclassify: bool
    ignore_overrides: bool
    raw: bool
    follow_source_symlinks: bool
    force_bird_head: bool
    allow_new_labels: bool
    config_path: Path | None
    #: The GUI's active-learning threshold: an animal the model scored below this is
    #: put in the review queue for a human to name. Defaults to 0.60 — deliberately
    #: *above* ``min_species_confidence`` (0.45), because the band between them is a
    #: guess the tool was willing to file but a human should still confirm.
    review_below: float = 0.60
    no_download: bool = False
    #: Skip species inference even when a model is available. What
    #: ``animal-classifier label`` runs on: you are about to name these animals
    #: yourself, so running a model that cannot name them only costs time and would
    #: fill the review queue with guesses you did not ask for.
    detect_only: bool = False
    species_model_explicit: bool = False
    ebird_api_key: str | None = field(repr=False, default=None)

    @property
    def catalog_path(self) -> Path:
        """The SQLite catalog always lives at ``<output_root>/.catalog.db``."""
        return self.output_root / ".catalog.db"

    @property
    def eligible_extensions(self) -> frozenset[str]:
        """Lower-cased extensions implied by ``formats`` (§3.1)."""
        return frozenset(
            ext for fam in self.formats for ext in FAMILY_EXTENSIONS[fam]
        )

    def to_json_dict(self) -> dict[str, Any]:
        """Serializable form for the ``runs.config_json`` column (§5.9).

        The eBird key is a credential and is reduced to a presence flag.
        """
        out: dict[str, Any] = {}
        for key, value in asdict(self).items():
            if key == "ebird_api_key":
                out["ebird_api_key_present"] = value is not None
            elif isinstance(value, Path) or isinstance(value, enum.StrEnum):
                out[key] = str(value)
            elif isinstance(value, tuple):
                out[key] = [str(v) for v in value]
            else:
                out[key] = value
        return out

    @classmethod
    def resolve(
        cls,
        *,
        command: Command,
        cli: Mapping[str, Any] | None = None,
        config_path: Path | None = None,
        env: Mapping[str, str] | None = None,
        cwd: Path | None = None,
    ) -> Config:
        """Build a validated ``Config`` from the four layers.

        ``cli`` holds only the flags the user actually passed; a key whose value is
        ``None`` is treated as "not given" so it cannot mask a lower layer.
        ``config_path`` is ``--config``: when given it **must** exist (the user
        asked for a specific file). Relative model/weights paths resolve against
        ``cwd`` (the current working directory).
        """
        cli = {k: v for k, v in (cli or {}).items() if v is not None}
        env = os.environ if env is None else env
        cwd = (Path.cwd() if cwd is None else cwd).resolve()

        toml_values, toml_origin = _load_toml(config_path, env)
        layers = _Layers(cli=cli, env=env, toml=toml_values, toml_origin=toml_origin)

        source_root = _resolve_source_root(layers, command)
        output_root = _resolve_output_root(layers)
        _guard_nesting(source_root, output_root)

        mode = _resolve_mode(layers)
        detector = _enum_value(layers, "detector", DetectorKind, DetectorKind.MEGADETECTOR)
        bird_provider = _enum_value(layers, "bird_provider", BirdProvider, DEFAULTS["bird_provider"])
        formats = _resolve_formats(layers)
        device_requested = _enum_value(layers, "device", Device, DEFAULTS["device"])
        device = _resolve_device(device_requested)

        species_model = _resolve_asset_path(layers, "species_model", cwd)
        bird_model = _resolve_asset_path(layers, "bird_model", cwd)
        detector_weights = _resolve_asset_path(layers, "detector_weights", cwd)

        config = cls(
            command=command,
            output_root=output_root,
            source_root=source_root,
            species_model=species_model,
            bird_model=bird_model,
            detector_weights=detector_weights,
            mode=mode,
            detector=detector,
            bird_provider=bird_provider,
            formats=formats,
            device=device,
            device_requested=device_requested,
            dominance_ratio=_number(
                layers, "dominance_ratio", float, minimum=1.0,
                why="a ratio below 1.0 would make the smaller box dominant",
            ),
            min_species_confidence=_number(
                layers, "min_species_confidence", float, minimum=0.0, maximum=1.0
            ),
            detector_confidence=_number(
                layers, "detector_confidence", float,
                minimum=0.0, maximum=1.0, exclusive_minimum=True,
            ),
            detector_iou=_number(
                layers, "detector_iou", float,
                minimum=0.0, maximum=1.0, exclusive_minimum=True,
            ),
            detector_image_size=_number(
                layers, "detector_image_size", int, minimum=320, maximum=2048, multiple_of=64
            ),
            detector_max_det=_number(layers, "detector_max_det", int, minimum=1, maximum=10000),
            crop_margin=_number(layers, "crop_margin", float, minimum=0.0, maximum=0.5),
            blur_threshold=_number(
                layers, "blur_threshold", float, minimum=0.0, exclusive_minimum=True
            ),
            max_file_bytes=_number(layers, "max_file_bytes", int, minimum=1024),
            jobs=_resolve_jobs(layers),
            limit=_optional_number(layers, "limit", int, minimum=1),
            port=_number_with_default(layers, "port", int, 8765, minimum=1024, maximum=65535),
            dry_run=_flag(layers, "dry_run"),
            reclassify=_flag(layers, "reclassify"),
            ignore_overrides=_flag(layers, "ignore_overrides"),
            raw=_flag(layers, "raw"),
            follow_source_symlinks=_flag(layers, "follow_source_symlinks"),
            force_bird_head=_flag(layers, "force_bird_head"),
            allow_new_labels=_flag(layers, "allow_new_labels"),
            config_path=toml_origin,
            review_below=_number_with_default(
                layers, "review_below", float, 0.60, minimum=0.0, maximum=1.0
            ),
            no_download=_flag(layers, "no_download"),
            detect_only=_flag(layers, "detect_only"),
            species_model_explicit=layers.find("species_model") is not None,
            ebird_api_key=_ebird_api_key(bird_provider, env),
        )

        _validate_bird_provider(config, env)
        _validate_required_assets(config)
        return config


# --------------------------------------------------------------------------- #
# Layered lookup
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Found:
    """A raw value plus the layer it came from, for actionable messages."""

    value: Any
    origin: str


class _Layers:
    """CLI → env → TOML lookup. Missing everywhere returns ``None``."""

    def __init__(
        self,
        *,
        cli: Mapping[str, Any],
        env: Mapping[str, str],
        toml: Mapping[str, Any],
        toml_origin: Path | None,
    ) -> None:
        self._cli = cli
        self._env = env
        self._toml = toml
        self._toml_origin = toml_origin

    def find(self, key: str) -> _Found | None:
        if key in self._cli:
            return _Found(self._cli[key], f"--{key.replace('_', '-')}")
        env_name = f"{ENV_PREFIX}{key.upper()}"
        if env_name in self._env:
            return _Found(self._env[env_name], env_name)
        if key in self._toml:
            return _Found(self._toml[key], f"{key} in {self._toml_origin}")
        return None


def _load_toml(
    config_path: Path | None, env: Mapping[str, str]
) -> tuple[Mapping[str, Any], Path | None]:
    """Read the TOML layer. An explicit ``--config`` that is absent is fatal."""
    if config_path is not None:
        resolved = config_path.expanduser().resolve()
        if not resolved.exists():
            raise ConfigError(
                f"--config file does not exist: {resolved}"
            )
        if not resolved.is_file():
            raise ConfigError(f"--config is not a regular file: {resolved}")
        return _parse_toml(resolved), resolved

    if env is os.environ:
        candidates = config_search_paths()
    else:
        xdg = env.get("XDG_CONFIG_HOME")
        candidates = tuple(
            p
            for p in (
                Path(xdg) / "animal-classifier" / "config.toml" if xdg else None,
                Path.home() / ".config" / "animal-classifier" / "config.toml",
            )
            if p is not None
        )
    for candidate in candidates:
        if candidate.is_file():
            return _parse_toml(candidate), candidate.resolve()
    return {}, None


def _parse_toml(path: Path) -> Mapping[str, Any]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"{path} is not valid TOML: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc

    unknown = sorted(set(data) - KNOWN_TOML_KEYS)
    if unknown:
        raise ConfigError(
            f"unknown key(s) in {path}: {', '.join(unknown)}. "
            f"Valid keys are: {', '.join(sorted(KNOWN_TOML_KEYS))}. "
            "Note there is deliberately no box-area floor setting; dominance_ratio "
            "is the only size gate."
        )
    return data


# --------------------------------------------------------------------------- #
# Typed, range-checked readers
# --------------------------------------------------------------------------- #


def _number(
    layers: _Layers,
    key: str,
    kind: Callable[[Any], Any],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    multiple_of: int | None = None,
    why: str | None = None,
) -> Any:
    found = layers.find(key)
    raw = DEFAULTS[key] if found is None else found.value
    origin = "built-in default" if found is None else found.origin
    return _coerce_number(
        key, raw, origin, kind,
        minimum=minimum, maximum=maximum,
        exclusive_minimum=exclusive_minimum, multiple_of=multiple_of, why=why,
    )


def _number_with_default(
    layers: _Layers,
    key: str,
    kind: Callable[[Any], Any],
    default: Any,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> Any:
    found = layers.find(key)
    raw = default if found is None else found.value
    origin = "built-in default" if found is None else found.origin
    return _coerce_number(key, raw, origin, kind, minimum=minimum, maximum=maximum)


def _optional_number(
    layers: _Layers, key: str, kind: Callable[[Any], Any], *, minimum: float | None = None
) -> Any | None:
    """A genuinely optional value: absent everywhere means "no limit"."""
    found = layers.find(key)
    if found is None:
        return None
    return _coerce_number(key, found.value, found.origin, kind, minimum=minimum)


def _coerce_number(
    key: str,
    raw: Any,
    origin: str,
    kind: Callable[[Any], Any],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
    multiple_of: int | None = None,
    why: str | None = None,
) -> Any:
    type_name = "an integer" if kind is int else "a number"
    if isinstance(raw, bool):
        raise ConfigError(f"{key} ({origin}) must be {type_name}, got a boolean: {raw!r}")
    try:
        value = kind(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} ({origin}) must be {type_name}, got {raw!r}") from exc
    if kind is float and value != value:  # NaN
        raise ConfigError(f"{key} ({origin}) must be a finite number, got {raw!r}")
    if kind is float and value in (float("inf"), float("-inf")):
        raise ConfigError(f"{key} ({origin}) must be a finite number, got {raw!r}")
    if kind is int and isinstance(raw, float) and not float(raw).is_integer():
        raise ConfigError(f"{key} ({origin}) must be {type_name}, got {raw!r}")

    reason = why or ""
    suffix = f" ({reason})" if reason else ""
    if minimum is not None:
        if exclusive_minimum and value <= minimum:
            raise ConfigError(
                f"{key} ({origin}) must be greater than {minimum}, got {value}{suffix}"
            )
        if not exclusive_minimum and value < minimum:
            raise ConfigError(
                f"{key} ({origin}) must be >= {minimum}, got {value}{suffix}"
            )
    if maximum is not None and value > maximum:
        raise ConfigError(f"{key} ({origin}) must be <= {maximum}, got {value}{suffix}")
    if multiple_of is not None and value % multiple_of != 0:
        raise ConfigError(
            f"{key} ({origin}) must be a multiple of {multiple_of}, got {value}"
        )
    return value


_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


def _flag(layers: _Layers, key: str) -> bool:
    found = layers.find(key)
    if found is None:
        return False
    raw = found.value
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ConfigError(
        f"{key} ({found.origin}) must be a boolean "
        f"({'/'.join(sorted(_TRUE))} or {'/'.join(sorted(_FALSE))}), got {raw!r}"
    )


def _enum_value(layers: _Layers, key: str, enum_cls: type[enum.StrEnum], default: Any) -> Any:
    found = layers.find(key)
    if found is None:
        return enum_cls(default)
    raw = found.value
    text = raw.value if isinstance(raw, enum.StrEnum) else str(raw).strip().lower()
    try:
        return enum_cls(text)
    except ValueError as exc:
        valid = ", ".join(m.value for m in enum_cls)
        raise ConfigError(
            f"{key} ({found.origin}) must be one of {{{valid}}}, got {raw!r}"
        ) from exc


def _resolve_mode(layers: _Layers) -> Mode:
    """``--link`` / ``--hardlink`` are booleans on the CLI; ``mode`` in TOML/env.

    The mutual exclusion of the two flags is a *usage* error (exit 2) raised by
    ``cli.py`` before we get here, so by this point at most one is set.
    """
    if _flag(layers, "link"):
        return Mode.LINK
    if _flag(layers, "hardlink"):
        return Mode.HARDLINK
    return _enum_value(layers, "mode", Mode, DEFAULTS["mode"])


def _resolve_formats(layers: _Layers) -> tuple[Format, ...]:
    """``formats`` replaces, never extends. ``raw`` is fatal here by design."""
    found = layers.find("formats")
    if found is None:
        return tuple(DEFAULT_FORMATS)
    raw = found.value
    if isinstance(raw, str):
        entries: Sequence[Any] = [part for part in raw.replace(",", " ").split() if part]
    elif isinstance(raw, Iterable):
        entries = list(raw)
    else:
        raise ConfigError(
            f"formats ({found.origin}) must be a list of families, got {raw!r}"
        )
    if not entries:
        raise ConfigError(
            f"formats ({found.origin}) must not be empty; valid families are "
            f"{{{', '.join(m.value for m in Format)}}}"
        )
    valid = ", ".join(m.value for m in Format)
    resolved: list[Format] = []
    for entry in entries:
        text = entry.value if isinstance(entry, enum.StrEnum) else str(entry).strip().lower()
        try:
            family = Format(text)
        except ValueError as exc:
            extra = ""
            if text == "raw":
                extra = (
                    " RAW is not a format family: enable it with --raw "
                    "(which needs the `raw` extra: uv pip install -e '.[raw]')."
                )
            raise ConfigError(
                f"formats ({found.origin}) contains {entry!r}, which is not one of "
                f"{{{valid}}}.{extra}"
            ) from exc
        if family not in resolved:
            resolved.append(family)
    return tuple(resolved)


def _resolve_jobs(layers: _Layers) -> int:
    jobs = _number(layers, "jobs", int, minimum=1)
    cpus = os.cpu_count()
    if cpus is not None and jobs > cpus:
        log.warning(
            "jobs=%d exceeds the %d available CPUs; oversubscription will slow the run",
            jobs,
            cpus,
        )
    return jobs


def _resolve_device(requested: Device) -> Device:
    """``auto`` resolves and is logged; explicit ``cuda`` without a device is fatal."""
    if requested is Device.CPU:
        return Device.CPU
    available = _cuda_available()
    if requested is Device.CUDA:
        if not available:
            raise ConfigError(
                "--device cuda was requested but no CUDA device is available on this "
                "host. Omit the flag to auto-select, or pass --device cpu."
            )
        return Device.CUDA
    resolved = Device.CUDA if available else Device.CPU
    log.info("device=auto resolved to %s", resolved.value)
    return resolved


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError as exc:  # torch is a hard dependency; a broken install is fatal
        raise ConfigError(
            "torch is not importable, so the device cannot be resolved. "
            f"Re-sync the environment with `uv sync --frozen`. Import error: {exc}"
        ) from exc
    return bool(torch.cuda.is_available())


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def _resolve_source_root(layers: _Layers, command: Command) -> Path | None:
    """``SOURCE`` is required for ``classify`` and meaningless elsewhere."""
    found = layers.find("source_root")
    if found is None:
        if command is Command.CLASSIFY:
            raise ConfigError("SOURCE is required: pass the card directory to classify")
        return None
    raw = Path(str(found.value)).expanduser()
    resolved = raw.resolve()
    if not resolved.exists():
        raise ConfigError(f"SOURCE does not exist: {resolved}")
    if not resolved.is_dir():
        raise ConfigError(f"SOURCE is not a directory: {resolved}")
    if not os.access(resolved, os.R_OK | os.X_OK):
        raise ConfigError(f"SOURCE is not readable: {resolved}")
    return resolved


def _resolve_output_root(layers: _Layers) -> Path:
    found = layers.find("output_root")
    raw = DEFAULTS["output_root"] if found is None else found.value
    origin = "built-in default" if found is None else found.origin
    path = Path(str(raw)).expanduser()
    if not str(path).strip():
        raise ConfigError(f"output_root ({origin}) must not be empty")
    resolved = _resolve_without_requiring_existence(path)
    if resolved.exists() and not resolved.is_dir():
        raise ConfigError(
            f"output_root ({origin}) exists but is not a directory: {resolved}"
        )
    return resolved


def _resolve_without_requiring_existence(path: Path) -> Path:
    """Absolute, normalized and **symlink-resolved**, tolerating a missing dir.

    ``os.path.abspath`` would normalize ``..`` textually without following
    symlinks, which lets an ``output_root`` that is itself a symlink *into* the
    card pass all three :func:`_guard_nesting` checks and then be written to.
    ``Path.resolve()`` is non-strict here: it resolves the existing ancestors of a
    directory that does not exist yet, so nothing is lost by using it.
    """
    return Path(path).resolve()


def _guard_nesting(source_root: Path | None, output_root: Path) -> None:
    """The card is read-only, so the two trees may not overlap in either direction."""
    if source_root is None:
        return
    if source_root == output_root:
        raise ConfigError(
            "output_root and SOURCE are the same directory, which would write to the "
            f"card: {output_root}"
        )
    if output_root.is_relative_to(source_root):
        raise ConfigError(
            f"output_root {output_root} is inside SOURCE {source_root}; the source "
            "tree is read-only. Choose an output directory outside the card."
        )
    if source_root.is_relative_to(output_root):
        raise ConfigError(
            f"SOURCE {source_root} is inside output_root {output_root}; classified "
            "files would be re-ingested on the next run. Choose separate directories."
        )


def _resolve_asset_path(layers: _Layers, key: str, cwd: Path) -> Path:
    """Relative model/weights paths resolve against the current working directory."""
    found = layers.find(key)
    raw = DEFAULTS[key] if found is None else found.value
    origin = "built-in default" if found is None else found.origin
    text = str(raw)
    if not text.strip():
        raise ConfigError(f"{key} ({origin}) must not be empty")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    return Path(os.path.abspath(path))


# --------------------------------------------------------------------------- #
# Cross-field validation
# --------------------------------------------------------------------------- #


def _ebird_api_key(provider: BirdProvider, env: Mapping[str, str]) -> str | None:
    if provider is not BirdProvider.EBIRD_ENRICH:
        return None
    key = env.get(EBIRD_API_KEY_ENV)
    return key if key else None


def _validate_bird_provider(config: Config, env: Mapping[str, str]) -> None:
    if config.bird_provider is BirdProvider.EBIRD_ENRICH:
        if config.ebird_api_key is None:
            raise ConfigError(
                f"bird_provider=ebird_enrich requires a non-empty {EBIRD_API_KEY_ENV} "
                "environment variable. Set it, or use --bird-provider own_bird_head."
            )
    elif config.bird_provider is BirdProvider.HOSTED_BIRD_API:
        raise AssetError(
            "bird_provider=hosted_bird_api is a documented stub with no shipped "
            "endpoint. It would require the environment variables "
            f"{' and '.join(HOSTED_BIRD_API_ENVS)} and an endpoint that accepts a "
            "cropped bird image. Use --bird-provider own_bird_head (our CUB-200 "
            "head) or ebird_enrich instead."
        )


def _validate_required_assets(config: Config) -> None:
    """Which assets a command genuinely needs — checked before anything is written."""
    required: list[tuple[str, Path, str]] = []
    if config.command is Command.CLASSIFY:
        # The detector checkpoint is *fetched on first use* and verified against its
        # pinned size + sha256, so its absence is not a configuration error — it is
        # a download this run will do. Only demand it up front when the user has
        # opted out of downloading (`--no-download`), where absence really is fatal.
        if config.detector is DetectorKind.MEGADETECTOR and config.no_download:
            required.append(
                (
                    "detector_weights",
                    config.detector_weights,
                    f"Download MegaDetector v5a from {DETECTOR_WEIGHTS_URL}",
                )
            )
        # Species model policy (F27, refined at chunk 17). A trained species model
        # is optional: with none present, `classify` runs the pass-through path
        # (every animal → `unknown`) and still does useful landscape/junk triage and
        # real detection. But if the user *explicitly* names a `--species-model`
        # that does not exist, that is a mistake we must not swallow — they asked for
        # a specific model, so its absence is fatal (I7). A present model at the
        # default path is loaded and used; an absent default path is the pass-through
        # path, not an error.
        if SPECIES_INFERENCE_WIRED and config.species_model_explicit and not config.detect_only:
            required.append(
                (
                    "species_model",
                    config.species_model,
                    TRAIN_HINT_SPECIES.format(
                        path=config.species_model, output_root=config.output_root
                    ),
                )
            )
    for key, path, remedy in required:
        if not path.exists():
            raise AssetError(f"{key} not found at {path}. Produce it with:\n  {remedy}")
        if not path.is_file():
            raise AssetError(f"{key} at {path} is not a regular file")
        if not os.access(path, os.R_OK):
            raise AssetError(f"{key} at {path} is not readable")
