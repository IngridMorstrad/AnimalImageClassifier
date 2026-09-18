"""Bird refinement providers (DESIGN.md §5.6).

Three providers behind one protocol. Two ship as real behaviour, one is an honest
stub:

- ``own_bird_head`` — our offline CUB-200 head (a species ``.acmodel`` like any
  other), run on a bird crop; ``refined`` when its top-1 clears
  ``min_species_confidence``, else ``kept_coarse``.
- ``ebird_enrich`` — **not a classifier, a re-ranker.** With EXIF GPS it fetches
  recent eBird observations near the photo and down-ranks (×0.25) aliased
  candidates that were not observed there; without GPS it only canonicalises names.
  The alias table is the whole contract, and a candidate with **no** alias row is
  exempt — absence from *our* table is a gap in our data, never evidence about the
  bird's range. Unreachable eBird is the one sanctioned graceful degradation: one
  warning, the coarse result unchanged, ``status="unreachable"``.
- ``hosted_bird_api`` — a stub that raises at config time naming the credentials it
  would need. It never silently no-ops.

This module implements the re-ranker's pure logic (alias matching, the down-rank,
re-sort, re-gate) so it is testable without a network; the network call itself is a
thin, cached wrapper it delegates to.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from typing import Protocol

log = logging.getLogger(__name__)

#: The eBird down-rank multiplier for an aliased-but-unobserved candidate (§13.5):
#: strong enough to demote an off-range species below a plausible in-range one,
#: weak enough that a correct-but-unreported species can still win from a high
#: score. A documented constant, not a fitted one — nothing in-sandbox can tune it.
EBIRD_ABSENT_MULTIPLIER = 0.25

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class Prediction:
    """A single ranked species guess (mirrors own_model's candidate shape)."""

    slug: str
    common: str
    scientific: str | None
    conf: float


@dataclass(frozen=True, slots=True)
class BirdResult:
    """The outcome of refinement — the only thing that reaches the merge rule."""

    provider: str
    status: str  # "refined" | "kept_coarse" | "unreachable" | "no_gps"
    predictions: tuple[Prediction, ...] = ()


def casefold_alnum(text: str) -> str:
    """Lowercase and strip every non-alphanumeric (§5.6's matching helper).

    ``"Brewer's Blackbird"`` and ``"brewers blackbird"`` collapse to the same key,
    which is what lets our orthography match eBird's.
    """
    return _NON_ALNUM.sub("", text.lower())


class BirdRefiner(Protocol):
    name: str

    def refine(
        self, candidates: tuple[Prediction, ...], *, gps: tuple[float, float] | None
    ) -> BirdResult: ...


@dataclass(frozen=True, slots=True)
class EBirdAlias:
    """One alias-table row: our label ↔ eBird's names."""

    cub_key: str
    ebird_com_name: str | None
    ebird_sci_name: str | None


def rerank_by_observations(
    candidates: tuple[Prediction, ...],
    *,
    aliases: dict[str, EBirdAlias],
    observed_sci: frozenset[str],
    observed_com: frozenset[str],
    min_confidence: float,
) -> tuple[tuple[Prediction, ...], bool]:
    """Apply §5.6's matching rule and re-sort. Returns (new_top5, changed).

    Pure and network-free: the caller supplies the observation sets (case-folded
    ``sciName`` and ``casefold_alnum(comName)``). A candidate with no alias row is
    exempt (score untouched). An aliased candidate absent from the observations is
    multiplied by :data:`EBIRD_ABSENT_MULTIPLIER`.
    """
    rescored: list[Prediction] = []
    for cand in candidates:
        alias = aliases.get(cand.slug)
        if alias is None:
            log.debug("%s: no eBird alias row; exempt from re-rank", cand.slug)
            rescored.append(cand)
            continue
        matched = False
        if (alias.ebird_sci_name and alias.ebird_sci_name.casefold() in observed_sci) or (alias.ebird_com_name and casefold_alnum(alias.ebird_com_name) in observed_com):
            matched = True
        if matched:
            rescored.append(cand)
        else:
            rescored.append(replace(cand, conf=cand.conf * EBIRD_ABSENT_MULTIPLIER))

    ordered = tuple(sorted(rescored, key=lambda p: p.conf, reverse=True))
    changed = [p.slug for p in ordered] != [p.slug for p in candidates]
    # A re-gate can also change the outcome even if the order held.
    gated_before = candidates[0].conf >= min_confidence if candidates else False
    gated_after = ordered[0].conf >= min_confidence if ordered else False
    return ordered, changed or (gated_before != gated_after)


def load_alias_table(rows: list[tuple[str, str, str]], *, label_space: set[str]) -> dict[str, EBirdAlias]:
    """Validate and index the alias CSV rows (§5.6). Fatal on a bad row.

    ``rows`` are ``(cub_key, ebird_com_name, ebird_sci_name)`` triples (header
    already stripped). Every ``cub_key`` must exist in the bird artifact's label
    space, and a row with both name columns empty is a typo, not data.
    """
    from ..errors import ConfigError

    table: dict[str, EBirdAlias] = {}
    for lineno, (cub_key, com, sci) in enumerate(rows, start=2):  # +1 header, +1 1-based
        key = cub_key.strip()
        com = com.strip()
        sci = sci.strip()
        if not key:
            raise ConfigError(f"ebird_aliases.csv line {lineno}: empty cub_key")
        if key not in label_space:
            raise ConfigError(
                f"ebird_aliases.csv line {lineno}: cub_key {key!r} is not in the "
                "bird artifact's label space"
            )
        if not com and not sci:
            raise ConfigError(
                f"ebird_aliases.csv line {lineno}: row for {key!r} has both name "
                "columns empty; it could never match anything"
            )
        table[key] = EBirdAlias(
            cub_key=key,
            ebird_com_name=com or None,
            ebird_sci_name=sci or None,
        )
    return table
