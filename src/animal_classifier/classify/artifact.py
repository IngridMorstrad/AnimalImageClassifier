"""The ``.acmodel`` artifact: a self-describing species model (DESIGN.md §7.1).

An artifact is one ``torch.save`` dict carrying everything inference needs so it
never has to guess: the architecture, input size, normalization, the label space
(with scientific names and taxonomic rank), the flat state dict, and a
``temperature``. :func:`load` rebuilds the model and validates the whole thing up
front.

**Two things are validated at load, before any image is filed, and both are I7 in
practice:**

- **``temperature`` must be present.** ``train`` always writes ``1.0`` (an
  uncalibrated model has an *identity* temperature — a real value, not a missing
  one). A missing key is a fatal :class:`AssetError`, because
  ``artifact.get("temperature", 1.0)`` is exactly the silent default for a required
  inference value that I7 forbids.
- **Every label must slugify to a legal, non-reserved directory.** The whole label
  space is run through :func:`animal_classifier.taxonomy.slug` at load, so a model
  with one unusable class name fails immediately — not half-way through a
  5,000-image run with files already on disk. The validated slugs are cached on the
  loaded artifact, so ``decide`` and ``materialize`` reuse the same strings.

``model_id`` is immutable and is written into every ``images`` row, so any label can
be traced to the exact model that produced it. An artifact is therefore never
mutated once recorded: ``eval --calibrate`` writes a *new* file (§7.4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..errors import AssetError
from ..taxonomy import RESERVED_LABELS, slug

log = logging.getLogger(__name__)

#: The only artifact format this build understands. A newer file is refused rather
#: than misread (§7.1).
FORMAT_VERSION: Final = 1

#: Keys every artifact dict must carry. A closed set so a foreign or truncated file
#: fails at load with the missing key named, not at inference with a KeyError.
_REQUIRED_KEYS: Final = frozenset(
    {
        "format_version",
        "model_id",
        "arch",
        "input_size",
        "normalize",
        "labels",
        "state_dict",
        "temperature",
    }
)

_VALID_RANKS: Final = frozenset({"species", "genus", "family", "order", "class"})


@dataclass(frozen=True, slots=True)
class LabelEntry:
    """One class in the model's label space, with its validated slug.

    ``slug`` is the directory name this class files into; it is computed and checked
    at load so ``decide``/``materialize`` never slugify at runtime (§5.8, §7.1).
    """

    slug: str
    common: str
    scientific: str | None
    taxon_class: str | None
    rank: str


@dataclass(frozen=True, slots=True)
class Artifact:
    """A loaded, validated species model ready for inference.

    Holds the rebuilt torch module plus the metadata. ``labels`` is ordered to match
    the model's output logits, so ``labels[i]`` is the class for logit ``i``.
    """

    model_id: str
    arch: str
    input_size: int
    mean: tuple[float, ...]
    std: tuple[float, ...]
    labels: tuple[LabelEntry, ...]
    temperature: float
    module: Any  # torch.nn.Module, typed loosely to avoid importing torch here

    @property
    def label_slugs(self) -> tuple[str, ...]:
        return tuple(entry.slug for entry in self.labels)


def build_module(arch: str, num_classes: int) -> Any:
    """Rebuild the model skeleton for ``arch`` with ``num_classes`` outputs.

    ``tinycnn`` is our from-scratch smoke-test net (§7.6); everything else is a
    ``timm`` backbone with a fresh head and ``pretrained=False`` — the weights come
    from the artifact's state dict, never from a download at load time.
    """
    if arch == "tinycnn":
        from ..training.tinycnn import TinyCNN  # noqa: PLC0415

        return TinyCNN(num_classes=num_classes)
    import timm  # noqa: PLC0415

    return timm.create_model(arch, pretrained=False, num_classes=num_classes)


def load(path: Path) -> Artifact:
    """Load and fully validate an ``.acmodel`` artifact (§7.1). Fatal on any problem."""
    import torch  # noqa: PLC0415

    if not path.exists():
        raise AssetError(
            f"species model not found at {path}. Produce one with "
            "`animal-classifier train`."
        )
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:  # noqa: BLE001 - any load failure is fatal/config
        raise AssetError(
            f"cannot load artifact {path}: {type(error).__name__}: {error}"
        ) from error

    if not isinstance(blob, dict):
        raise AssetError(f"artifact {path} is not a dict; it is not an .acmodel file")

    missing = _REQUIRED_KEYS - set(blob)
    if missing:
        if "temperature" in missing:
            raise AssetError(
                f"artifact {path} has no 'temperature'; it was written by an "
                "older/foreign writer — re-run `animal-classifier train`, or fit "
                "one with `animal-classifier eval --calibrate --out <new.acmodel>`"
            )
        raise AssetError(
            f"artifact {path} is missing required key(s) {sorted(missing)}; "
            "re-run `animal-classifier train`"
        )

    version = blob["format_version"]
    if version != FORMAT_VERSION:
        raise AssetError(
            f"artifact {path} has format_version {version}, this build understands "
            f"{FORMAT_VERSION}; re-run `animal-classifier train`"
        )

    labels = _validate_labels(blob["labels"], path)
    normalize = blob["normalize"]
    try:
        mean = tuple(float(v) for v in normalize["mean"])
        std = tuple(float(v) for v in normalize["std"])
    except (KeyError, TypeError, ValueError) as error:
        raise AssetError(
            f"artifact {path} has a malformed 'normalize' block: {error}"
        ) from error

    temperature = float(blob["temperature"])
    if not temperature > 0.0:
        raise AssetError(
            f"artifact {path} has non-positive temperature {temperature}; a "
            "temperature scales logits and must be > 0"
        )

    module = build_module(str(blob["arch"]), len(labels))
    try:
        module.load_state_dict(blob["state_dict"], strict=True)
    except Exception as error:  # noqa: BLE001 - a mismatch is fatal/config
        raise AssetError(
            f"artifact {path} state_dict does not match arch {blob['arch']!r} with "
            f"{len(labels)} classes: {error}"
        ) from error
    module.eval()

    log.info(
        "loaded species artifact %s (arch=%s, %d classes, T=%.3f)",
        blob["model_id"],
        blob["arch"],
        len(labels),
        temperature,
    )
    return Artifact(
        model_id=str(blob["model_id"]),
        arch=str(blob["arch"]),
        input_size=int(blob["input_size"]),
        mean=mean,
        std=std,
        labels=labels,
        temperature=temperature,
        module=module,
    )


def _validate_labels(raw: object, path: Path) -> tuple[LabelEntry, ...]:
    """Validate and slugify the whole label space up front (§7.1)."""
    if not isinstance(raw, list) or not raw:
        raise AssetError(f"artifact {path} has an empty or malformed 'labels' list")

    entries: list[LabelEntry] = []
    seen_slugs: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "common" not in item:
            raise AssetError(
                f"artifact {path} label {index} is malformed (needs at least "
                f"'common'): {item!r}"
            )
        common = str(item["common"])
        try:
            label_slug = slug(common)
        except Exception as error:  # ConfigError from slug()
            raise AssetError(
                f"artifact {path} label {index} {common!r} does not slugify to a "
                f"legal label directory: {error}"
            ) from error
        if label_slug in RESERVED_LABELS:
            raise AssetError(
                f"artifact {path} label {index} {common!r} slugifies to reserved "
                f"label {label_slug!r} ({sorted(RESERVED_LABELS)}); a model class "
                "cannot shadow a pipeline outcome"
            )
        if label_slug in seen_slugs:
            raise AssetError(
                f"artifact {path} has two labels slugifying to {label_slug!r}; "
                "label slugs must be unique"
            )
        seen_slugs.add(label_slug)

        rank = str(item.get("rank", "species"))
        if rank not in _VALID_RANKS:
            raise AssetError(
                f"artifact {path} label {common!r} has rank {rank!r}, not one of "
                f"{sorted(_VALID_RANKS)}"
            )
        scientific = item.get("scientific")
        entries.append(
            LabelEntry(
                slug=label_slug,
                common=common,
                scientific=None if scientific in (None, "") else str(scientific),
                taxon_class=None if item.get("class") in (None, "") else str(item["class"]),
                rank=rank,
            )
        )
    return tuple(entries)


def save(
    path: Path,
    *,
    model_id: str,
    arch: str,
    input_size: int,
    mean: list[float],
    std: list[float],
    labels: list[dict[str, Any]],
    state_dict: dict[str, Any],
    temperature: float = 1.0,
    train_meta: dict[str, Any] | None = None,
) -> None:
    """Write an ``.acmodel`` artifact (§7.1). Used by ``train`` and ``eval --calibrate``."""
    import torch  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {
        "format_version": FORMAT_VERSION,
        "model_id": model_id,
        "arch": arch,
        "input_size": input_size,
        "normalize": {"mean": list(mean), "std": list(std)},
        "labels": labels,
        "state_dict": state_dict,
        "temperature": float(temperature),
        "train": train_meta or {},
    }
    torch.save(blob, path)
    log.info("wrote artifact %s (%d classes) to %s", model_id, len(labels), path)
