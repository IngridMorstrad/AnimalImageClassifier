"""The label space: ``slug()``, ``LABEL_RE``, ``RESERVED_LABELS``, static name tables.

This module is the **only** place a model class id is turned into a name that
reaches the filesystem (DESIGN.md §5.8) — which makes it the fix for a defect that
was reproduced against real data:

**DEFECT 2 — raw dataset class names leaked into output paths.** CUB-200-2011 ships
its classes as directories called ``022.Chuck_will_Widow``, and MegaDetector/COCO
ship terse lowercase ids. Feeding either straight through would have produced
``~/animal_pics/022_chuck_will_widow/``. Two distinct concepts are therefore kept
apart here, permanently:

``key``
    The stable machine identifier of a class. For CUB it is derived from the
    archive's directory name with the ordinal prefix stripped
    (``022.Chuck_will_Widow`` → ``chuck_will_widow``, see
    :func:`cub_key_from_dirname`). It is what a manifest, an ``.acmodel`` label
    entry and ``taxonomy/data/ebird_aliases.csv``'s ``cub_key`` column all agree
    on. **A key is never a directory name.**

``label``
    The directory component, and it is *always* ``slug(common)`` where ``common``
    comes from the authoritative static table in ``taxonomy/data/``. So the folder
    is ``chuck_wills_widow``, never ``022_chuck_will_widow`` and never
    ``chuck_will_widow``.

Both are computed and validated **at load time**, never at directory-creation time
(DESIGN.md §7.1, invariant I10): a table or artifact carrying one unusable class
name fails immediately with exit 3 rather than half-way through a 5,000-image run
with files already on disk.

Two authoring conventions in ``taxonomy/data/*.csv`` are load-bearing:

* **Apostrophes are written U+2019** (``Brewer’s Blackbird``, ``Chuck-will’s-widow``).
  ``slug()``'s NFKD → ``encode("ascii", "ignore")`` step *drops* U+2019, so these
  yield ``brewers_blackbird`` and ``chuck_wills_widow``. An ASCII ``'`` survives
  NFKD and collapses to ``_``, giving the legal-but-ugly ``brewer_s_blackbird``;
  that spelling is still correct behaviour for a label a user types into the GUI
  (E21), it is simply not how we author our own tables.
* **``scientific`` is empty rather than invented.** No taxonomy API is reachable
  (DESIGN.md §13.3), so it is filled only where a name could be written down
  offline with confidence. Empty means *unknown*, which is an honest absence of an
  optional field — not a substituted default (invariant I7). ``scientific`` holds
  the name at ``rank``: a binomial for ``rank=species``, a genus or family name for
  the coarser ranks, and empty for ``rank=class`` because the ``class`` column
  already carries it.

CUB's own orthography is sometimes wrong (``141.Artic_Tern``). The key keeps the
dataset's spelling, because that is what the data is keyed by; the display name and
therefore the folder are corrected (``arctic_tern``). That divergence is exactly
what the key/label split is for.
"""

from __future__ import annotations

import csv
import enum
import functools
import logging
import re
import unicodedata
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

from ..errors import ConfigError

log = logging.getLogger(__name__)

DATA_DIR: Final = Path(__file__).resolve().parent / "data"
COCO_ANIMALS_CSV: Final = DATA_DIR / "coco_animals.csv"
CUB200_CSV: Final = DATA_DIR / "cub200.csv"

# Published here beside slug(), which must satisfy it (DESIGN.md §5.8).
LABEL_RE: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
RESERVED_LABELS: Final = frozenset({"multiple", "landscape", "junk", "unknown"})
_NON_ALNUM: Final = re.compile(r"[^a-z0-9]+")

#: The class every bird rolls up to; the bird-refinement trigger keys on it (§5.6).
AVES: Final = "Aves"

#: Column order of every table under ``taxonomy/data/``. Checked exactly at load.
CSV_HEADER: Final = ("key", "common", "scientific", "class", "rank")

#: A CUB-200-2011 image directory: a three-digit ordinal, a dot, then the name.
_CUB_DIR_RE: Final = re.compile(r"^(?P<index>\d{3})\.(?P<name>\S.*)$")

#: DEFECT 2's belt-and-braces check, applied to **our own tables only**.
#:
#: :data:`LABEL_RE` deliberately permits a leading digit, because a user may type
#: one into the GUI and DESIGN.md §5.8 fixes that regex. But a leading digit in a
#: table we author ourselves has exactly one cause: a raw dataset class id such as
#: ``022.Chuck_will_Widow`` leaked through instead of being normalised. Rejecting it
#: at load turns the defect into a startup failure rather than a directory on disk.
_LEADING_DIGIT: Final = re.compile(r"^\d")


def slug(common: str) -> str:
    """Canonical directory component for a species name. Never returns an invalid name.

    Verbatim from DESIGN.md §5.8. It **raises** rather than falling back to a
    sanitized guess, because the destination directory is a required value (I7).
    """
    folded = unicodedata.normalize("NFKD", common).encode("ascii", "ignore").decode()
    s = _NON_ALNUM.sub("_", folded.lower()).strip("_")[:64].rstrip("_")
    if not LABEL_RE.fullmatch(s):
        raise ConfigError(
            f"species label {common!r} does not slugify to a valid label directory "
            f"(got {s!r}, must match {LABEL_RE.pattern}); fix the label in the model artifact "
            f"or the taxonomy table"
        )
    return s


def is_reserved(label: str) -> bool:
    """Whether ``label`` is one of the pipeline outcomes reserved in the label space."""
    return label in RESERVED_LABELS


def validate_label(label: str, *, source: str) -> str:
    """Return ``label`` unchanged, or raise :class:`ConfigError` explaining why not.

    The single gate every label crosses before it can become a directory: it must
    match :data:`LABEL_RE` and must not collide with :data:`RESERVED_LABELS`, so a
    model class can never shadow a pipeline outcome and make ``unknown/``
    ambiguous (DESIGN.md §5.8). ``source`` names where the label came from — an
    artifact path, a CSV line, an HTTP request — and appears in the message.

    Callers map the failure to their own surface: ``artifact.load()`` lets it exit
    3, the GUI's label validator turns it into HTTP 422 (§6). ``--allow-new-labels``
    widens *which* labels are known; it never widens this syntax.
    """
    if not LABEL_RE.fullmatch(label):
        raise ConfigError(
            f"{source}: label {label!r} is not a valid label directory "
            f"(must match {LABEL_RE.pattern})"
        )
    if label in RESERVED_LABELS:
        raise ConfigError(
            f"{source}: label {label!r} collides with a reserved pipeline outcome "
            f"({', '.join(sorted(RESERVED_LABELS))}); rename the class so it cannot "
            f"shadow a pipeline label on disk"
        )
    return label


def is_aves_class(taxon_class: str) -> bool:
    """Whether a taxon ``class`` value rolls up to birds.

    The bird-refinement trigger (§5.6) reads ``class`` off the *artifact's* label
    entries rather than off a static table, because the artifact is what produced
    the candidate. This helper is that comparison, named once so the string
    ``"Aves"`` is not spelled out at each call site.
    """
    return taxon_class == AVES


def cub_key_from_dirname(dirname: str) -> str:
    """``022.Chuck_will_Widow`` → ``chuck_will_widow`` — the class *key*, not a label.

    DEFECT 2's entry point. The ordinal prefix is dataset bookkeeping and is
    stripped; the remainder is lowercased and its punctuation collapsed so the key
    is a stable identifier. The result is deliberately **not** used as a directory
    name — :meth:`TaxonTable.label_for` does that, from the authoritative display
    name in ``cub200.csv``.

    Raises :class:`ConfigError` on anything that is not a CUB class directory, so a
    stray file in the archive is a loud failure rather than a silently mangled key.
    """
    match = _CUB_DIR_RE.fullmatch(dirname.strip())
    if match is None:
        raise ConfigError(
            f"{dirname!r} is not a CUB-200-2011 class directory name; expected a "
            f"three-digit ordinal, a dot, then the class name (e.g. "
            f"'022.Chuck_will_Widow')"
        )
    key = _NON_ALNUM.sub("_", match["name"].lower()).strip("_")
    if not LABEL_RE.fullmatch(key) or key in RESERVED_LABELS:
        raise ConfigError(
            f"CUB class directory {dirname!r} yields the unusable class key {key!r} "
            f"(must match {LABEL_RE.pattern} and must not be one of "
            f"{', '.join(sorted(RESERVED_LABELS))})"
        )
    return key


class Rank(enum.StrEnum):
    """Taxonomic precision of a label, stored as ``boxes.species_rank``.

    Kept explicit so a coarse identification (COCO's ``bird``, which is a whole
    class) is distinguishable from a fine one (``plains_zebra``) everywhere,
    including the GUI caption (DESIGN.md §5.5, §7.1).
    """

    SPECIES = "species"
    GENUS = "genus"
    FAMILY = "family"
    ORDER = "order"
    CLASS = "class"


@dataclass(frozen=True, slots=True)
class Taxon:
    """One row of a static table, with its label already computed and validated.

    ``label`` is stored rather than derived on access: DESIGN.md §7.1 requires the
    slugs be cached at load so ``decide.py`` and ``materialize.py`` use the very
    same validated strings, and so a bad name cannot surface mid-run.
    """

    key: str
    common: str
    scientific: str | None
    taxon_class: str
    rank: Rank
    label: str

    @property
    def is_aves(self) -> bool:
        """Whether this taxon rolls up to class ``Aves`` (the bird trigger, §5.6)."""
        return is_aves_class(self.taxon_class)


@dataclass(frozen=True, slots=True, eq=False)
class TaxonTable:
    """An immutable, fully validated static name table.

    Lookups that stand in for a *required* value (``label_for`` and friends) raise
    :class:`ConfigError` on a miss rather than returning a placeholder, per I7.
    The ``find_by_*`` methods are genuine searches and return ``None``.
    """

    name: str
    path: Path
    by_key: Mapping[str, Taxon]
    by_label: Mapping[str, Taxon]
    _by_common: Mapping[str, Taxon]
    _by_scientific: Mapping[str, Taxon]

    def __len__(self) -> int:
        return len(self.by_key)

    def __iter__(self) -> Iterator[Taxon]:
        return iter(self.by_key.values())

    def __contains__(self, key: str) -> bool:
        return key in self.by_key

    def __getitem__(self, key: str) -> Taxon:
        try:
            return self.by_key[key]
        except KeyError:
            raise ConfigError(
                f"class key {key!r} is not in the {self.name} taxonomy table "
                f"({self.path}); add a row for it, or fix the class key in the "
                f"model artifact or manifest"
            ) from None

    def keys(self) -> tuple[str, ...]:
        """Table keys in file order."""
        return tuple(self.by_key)

    def labels(self) -> tuple[str, ...]:
        """Every directory name this table can produce, in file order."""
        return tuple(taxon.label for taxon in self.by_key.values())

    def label_for(self, key: str) -> str:
        """The directory component for a class key. The only sanctioned way to name a dir."""
        return self[key].label

    def common_for(self, key: str) -> str:
        return self[key].common

    def scientific_for(self, key: str) -> str | None:
        """The scientific name at the taxon's rank, or ``None`` when unknown offline."""
        return self[key].scientific

    def rank_for(self, key: str) -> Rank:
        return self[key].rank

    def rolls_up_to_aves(self, key: str) -> bool:
        """Whether the class key is a bird — the table-driven form of the §5.6 trigger."""
        return self[key].is_aves

    def aves_keys(self) -> frozenset[str]:
        return frozenset(taxon.key for taxon in self.by_key.values() if taxon.is_aves)

    def find_by_common(self, common: str) -> Taxon | None:
        """Case-insensitive common-name lookup. A genuine search: ``None`` on a miss."""
        return self._by_common.get(common.casefold())

    def find_by_scientific(self, scientific: str) -> Taxon | None:
        """Case-insensitive scientific-name lookup. A genuine search: ``None`` on a miss."""
        return self._by_scientific.get(scientific.casefold())


def _reject_raw_dataset_id(value: str, *, kind: str, where: str) -> None:
    """Refuse a leading digit in a table we author. See :data:`_LEADING_DIGIT`."""
    if _LEADING_DIGIT.match(value):
        raise ConfigError(
            f"{where}: {kind} {value!r} starts with a digit, which means a raw dataset "
            f"class id (e.g. '022.Chuck_will_Widow') reached the table instead of a "
            f"normalised name; strip the ordinal prefix so the folder is "
            f"'chuck_wills_widow', not '022_chuck_will_widow'"
        )


def load_table(path: Path, *, name: str) -> TaxonTable:
    """Read, validate and index one ``taxonomy/data/*.csv``.

    Validation is total and every failure is a :class:`ConfigError` (exit 3) naming
    the file and the **line number**, in the style DESIGN.md §10.2 requires of the
    eBird alias table. Checked here: the exact header; a non-empty ``key`` that is
    itself a legal, non-reserved identifier and unique; a non-empty ``common`` that
    slugifies to a legal, non-reserved, table-unique label; a non-empty ``class``; a
    ``rank`` drawn from :class:`Rank`; and the right number of fields per row.

    The label-uniqueness check matters as much as the rest: two keys whose display
    names slugify onto one directory would silently merge two species into one
    folder, which no later stage could detect.
    """
    if not path.is_file():
        raise ConfigError(
            f"the {name} taxonomy table is missing at {path}; it ships inside the "
            f"package, so this build is incomplete — reinstall with `uv sync --frozen`"
        )
    by_key: dict[str, Taxon] = {}
    by_label: dict[str, Taxon] = {}
    by_common: dict[str, Taxon] = {}
    by_scientific: dict[str, Taxon] = {}

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ConfigError(
                f"{path}: the {name} taxonomy table is empty; expected the header "
                f"{','.join(CSV_HEADER)}"
            ) from None
        if tuple(field.strip() for field in header) != CSV_HEADER:
            raise ConfigError(
                f"{path}:1: unexpected header {header!r}; expected exactly "
                f"{list(CSV_HEADER)}"
            )
        for row in reader:
            line = reader.line_num
            if not row or all(not field.strip() for field in row):
                raise ConfigError(f"{path}:{line}: blank row; delete it")
            if len(row) != len(CSV_HEADER):
                raise ConfigError(
                    f"{path}:{line}: has {len(row)} fields, expected "
                    f"{len(CSV_HEADER)} ({','.join(CSV_HEADER)})"
                )
            key, common, scientific, taxon_class, rank_text = (f.strip() for f in row)
            where = f"{path}:{line}"
            if not key:
                raise ConfigError(f"{where}: 'key' is empty; it is the class identifier")
            validate_label(key, source=where)
            _reject_raw_dataset_id(key, kind="key", where=where)
            if key in by_key:
                raise ConfigError(f"{where}: duplicate key {key!r}")
            if not common:
                raise ConfigError(
                    f"{where}: 'common' is empty for key {key!r}; it is the display name "
                    f"and the source of the label directory, so it is required"
                )
            if not taxon_class:
                raise ConfigError(
                    f"{where}: 'class' is empty for key {key!r}; the bird trigger reads it, "
                    f"so write the taxonomic class (e.g. {AVES!r}, 'Mammalia')"
                )
            try:
                rank = Rank(rank_text)
            except ValueError:
                raise ConfigError(
                    f"{where}: rank {rank_text!r} for key {key!r} is not one of "
                    f"{', '.join(r.value for r in Rank)}"
                ) from None
            label = validate_label(slug(common), source=where)
            _reject_raw_dataset_id(label, kind="label", where=where)
            if label in by_label:
                raise ConfigError(
                    f"{where}: display name {common!r} slugifies to {label!r}, which key "
                    f"{by_label[label].key!r} already claims; two classes cannot share one "
                    f"label directory — change one of the display names"
                )
            taxon = Taxon(
                key=key,
                common=common,
                scientific=scientific or None,
                taxon_class=taxon_class,
                rank=rank,
                label=label,
            )
            by_key[key] = taxon
            by_label[label] = taxon
            by_common.setdefault(common.casefold(), taxon)
            if taxon.scientific is not None:
                by_scientific.setdefault(taxon.scientific.casefold(), taxon)

    if not by_key:
        raise ConfigError(f"{path}: the {name} taxonomy table has a header but no rows")
    log.debug(
        "loaded %s taxonomy table: %d classes, %d with a scientific name (%s)",
        name,
        len(by_key),
        len(by_scientific),
        path,
    )
    return TaxonTable(
        name=name,
        path=path,
        by_key=MappingProxyType(by_key),
        by_label=MappingProxyType(by_label),
        _by_common=MappingProxyType(by_common),
        _by_scientific=MappingProxyType(by_scientific),
    )


@functools.lru_cache(maxsize=1)
def coco_animals() -> TaxonTable:
    """The 10 COCO animal classes our species head is trained on (DESIGN.md §13.2)."""
    return load_table(COCO_ANIMALS_CSV, name="coco_animals")


@functools.lru_cache(maxsize=1)
def cub200() -> TaxonTable:
    """The 200 CUB-200-2011 bird classes our bird head is trained on."""
    return load_table(CUB200_CSV, name="cub200")


@functools.lru_cache(maxsize=1)
def merged() -> Mapping[str, Taxon]:
    """Every known class key → taxon, birds overriding the coarse ``bird`` class.

    COCO's ``bird`` and CUB's 200 species live in one namespace here so a caller
    holding only a class key can resolve a name without knowing which head produced
    it. The keys are disjoint apart from nothing at all — COCO's key is ``bird``,
    CUB's are species — so the merge cannot lose a class.
    """
    combined: dict[str, Taxon] = dict(coco_animals().by_key)
    combined.update(cub200().by_key)
    return MappingProxyType(combined)
