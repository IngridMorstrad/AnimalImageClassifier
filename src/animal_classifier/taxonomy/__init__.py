"""Taxonomy: the closed label space and the static offline name tables.

No species or taxonomy API is reachable from this environment (DESIGN.md §13.3), so
every common ↔ scientific name the tool prints or files comes from the CSVs in
``taxonomy/data/``. They are hand-authored, versioned with the code, and validated
in full the moment they are loaded.

Import from this package, not from :mod:`animal_classifier.taxonomy.labels`.
"""

from __future__ import annotations

from .labels import (
    AVES,
    COCO_ANIMALS_CSV,
    CSV_HEADER,
    CUB200_CSV,
    DATA_DIR,
    LABEL_RE,
    RESERVED_LABELS,
    Rank,
    Taxon,
    TaxonTable,
    coco_animals,
    cub200,
    cub_key_from_dirname,
    is_aves_class,
    is_reserved,
    load_table,
    merged,
    slug,
    validate_label,
)

__all__ = [
    "AVES",
    "COCO_ANIMALS_CSV",
    "CSV_HEADER",
    "CUB200_CSV",
    "DATA_DIR",
    "LABEL_RE",
    "RESERVED_LABELS",
    "Rank",
    "Taxon",
    "TaxonTable",
    "coco_animals",
    "cub200",
    "cub_key_from_dirname",
    "is_aves_class",
    "is_reserved",
    "load_table",
    "merged",
    "slug",
    "validate_label",
]
