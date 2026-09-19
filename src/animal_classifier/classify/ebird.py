"""The ``ebird_enrich`` provider: a locality re-ranker, not a classifier (§5.6).

One cached HTTP call to eBird's recent-observations endpoint, then §5.6's matching
rule down-ranks candidates that our alias table knows about but eBird did not report
nearby. Everything that can go wrong degrades in a documented way:

============================  ==============================================
Situation                     Outcome
============================  ==============================================
no EXIF GPS                   **no network call**, ranking unchanged, ``no_gps``
host unreachable / timeout    one ``WARNING`` per run, coarse result, ``unreachable``
401 / 403                     **fatal** — a rejected credential is a real misconfiguration
observations returned         re-rank, re-gate, ``refined`` or ``kept_coarse``
============================  ==============================================

The unreachable branch is the single sanctioned graceful degradation in the design,
and it is sanctioned because the enrichment is *optional*: no required value is being
substituted, and ``provider_status`` records that enrichment did not happen.

The HTTP client is built by an injectable factory so the e2e suite can supply an
``httpx.MockTransport`` — §11.2's one sanctioned test seam at this boundary, which
exists because ``api.ebird.org`` is unreachable from the build sandbox and the
matching rule is otherwise untestable at all.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Final

from ..errors import ConfigError
from .birds import BirdResult, EBirdAlias, Prediction, casefold_alnum, rerank_by_observations

log = logging.getLogger(__name__)

ENDPOINT: Final = "https://api.ebird.org/v2/data/obs/geo/recent"
CONNECT_TIMEOUT: Final = 4.0
READ_TIMEOUT: Final = 8.0
SEARCH_RADIUS_KM: Final = 50
LOOKBACK_DAYS: Final = 30

#: Cache file under the output root, keyed by ``(round(lat,2), round(lon,2))`` (§5.6).
CACHE_NAME: Final = ".ebird-cache.json"

#: Injectable for the test seam: returns an object with ``.get(url, params, headers)``.
ClientFactory = Callable[[], Any]


def _default_client_factory() -> Any:
    import httpx  # noqa: PLC0415

    return httpx.Client(timeout=httpx.Timeout(READ_TIMEOUT, connect=CONNECT_TIMEOUT))


class EBirdEnricher:
    """Canonicalise and locality-re-rank bird candidates (§5.6)."""

    name = "ebird_enrich"

    def __init__(
        self,
        *,
        api_key: str,
        aliases: dict[str, EBirdAlias],
        min_confidence: float,
        output_root: Path,
        client_factory: ClientFactory = _default_client_factory,
    ) -> None:
        if not api_key:
            raise ConfigError(
                "ebird_enrich requires a non-empty ANIMAL_CLASSIFIER_EBIRD_API_KEY"
            )
        self._api_key = api_key
        self._aliases = aliases
        self._min_confidence = min_confidence
        self._cache_path = output_root / CACHE_NAME
        self._client_factory = client_factory
        self._cache = self._load_cache()
        self._warned = False

    # ------------------------------------------------------------------ cache

    def _load_cache(self) -> dict[str, Any]:
        if not self._cache_path.is_file():
            return {}
        try:
            return json.loads(self._cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            log.debug("ignoring unreadable eBird cache %s: %s", self._cache_path, error)
            return {}

    def _save_cache(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._cache), encoding="utf-8")
        except OSError as error:
            log.debug("could not write eBird cache: %s", error)

    @staticmethod
    def _cache_key(lat: float, lon: float) -> str:
        return f"{round(lat, 2)},{round(lon, 2)}"

    # ------------------------------------------------------------------ refine

    def refine(
        self,
        candidates: tuple[Prediction, ...],
        *,
        gps: tuple[float, float] | None,
    ) -> BirdResult:
        """Apply §5.6 to ``candidates``. Never raises for a network problem."""
        if not candidates:
            return BirdResult(provider=self.name, status="kept_coarse", predictions=())

        if gps is None:
            # No locality to reason about: canonicalise names only, ranking untouched,
            # and make no network call at all (§5.6).
            log.debug("ebird_enrich: no EXIF GPS; ranking unchanged")
            return BirdResult(
                provider=self.name,
                status="no_gps",
                predictions=self._canonicalise(candidates),
            )

        observations = self._observations(*gps)
        if observations is None:
            return BirdResult(provider=self.name, status="unreachable", predictions=candidates)

        observed_sci = frozenset(
            str(o.get("sciName", "")).casefold() for o in observations if o.get("sciName")
        )
        observed_com = frozenset(
            casefold_alnum(str(o.get("comName", ""))) for o in observations if o.get("comName")
        )
        reranked, changed = rerank_by_observations(
            self._canonicalise(candidates),
            aliases=self._aliases,
            observed_sci=observed_sci,
            observed_com=observed_com,
            min_confidence=self._min_confidence,
        )
        return BirdResult(
            provider=self.name,
            status="refined" if changed else "kept_coarse",
            predictions=reranked,
        )

    def _canonicalise(self, candidates: tuple[Prediction, ...]) -> tuple[Prediction, ...]:
        """Replace names with eBird's orthography where an alias row exists (§5.6)."""
        from dataclasses import replace  # noqa: PLC0415

        out = []
        for candidate in candidates:
            alias = self._aliases.get(candidate.slug)
            if alias is None:
                out.append(candidate)
                continue
            out.append(
                replace(
                    candidate,
                    common=alias.ebird_com_name or candidate.common,
                    scientific=alias.ebird_sci_name or candidate.scientific,
                )
            )
        return tuple(out)

    def _observations(self, lat: float, lon: float) -> list[dict[str, Any]] | None:
        """Recent nearby observations, or ``None`` when eBird is unreachable.

        A 401/403 is **fatal** (a rejected credential is a real misconfiguration);
        every other failure degrades with exactly one warning per run.
        """
        key = self._cache_key(lat, lon)
        if key in self._cache:
            log.debug("ebird_enrich: cache hit for %s", key)
            return self._cache[key]

        try:
            client = self._client_factory()
            response = client.get(
                ENDPOINT,
                params={"lat": lat, "lng": lon, "dist": SEARCH_RADIUS_KM, "back": LOOKBACK_DAYS},
                headers={"X-eBirdApiToken": self._api_key},
            )
        except Exception as error:  # noqa: BLE001 - any transport failure degrades
            self._warn_once(f"eBird unreachable ({type(error).__name__}: {error})")
            return None

        status = getattr(response, "status_code", None)
        if status in (401, 403):
            raise ConfigError(
                f"eBird rejected the API key (HTTP {status}); "
                "check ANIMAL_CLASSIFIER_EBIRD_API_KEY or use --bird-provider own_bird_head"
            )
        if status != 200:
            self._warn_once(f"eBird returned HTTP {status}")
            return None
        try:
            payload = response.json()
        except Exception as error:  # noqa: BLE001 - bad JSON degrades
            self._warn_once(f"eBird returned unparseable JSON ({error})")
            return None
        if not isinstance(payload, list):
            self._warn_once("eBird returned an unexpected payload shape")
            return None

        self._cache[key] = payload
        self._save_cache()
        return payload

    def _warn_once(self, message: str) -> None:
        """Exactly one WARNING per run for the degradation (§5.6)."""
        if not self._warned:
            log.warning(
                "%s; keeping the own_bird_head result unchanged "
                "(provider_status='unreachable')",
                message,
            )
            self._warned = True
        else:
            log.debug("%s (already warned this run)", message)
