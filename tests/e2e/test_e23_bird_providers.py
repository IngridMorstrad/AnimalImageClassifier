"""E23: ``ebird_enrich`` degrades, and the alias table governs matching (§5.6).

Three kinds of leg:

1. **CLI config legs** — ``hosted_bird_api`` is a fatal stub; ``ebird_enrich`` without
   ``ANIMAL_CLASSIFIER_EBIRD_API_KEY`` is fatal. Both through the real CLI.
2. **The degradation legs** — with a dummy key and no reachable host, the provider
   returns the coarse result with ``status='unreachable'`` and warns exactly **once**
   per run; with no GPS it makes **no network call at all** and records ``no_gps``.
3. **The stubbed-transport leg** — ``httpx.MockTransport`` injected through the
   provider's own client factory. This is §11.2's one sanctioned seam at this
   boundary: it replaces the *external HTTP service*, not any of our code, and it
   exists because ``api.ebird.org`` is unreachable from the build sandbox so the
   matching rule would otherwise be untestable at all.

The load-bearing assertion in leg 3 is the **exemption**: a candidate with no alias row
must come back with a bit-identical score. Absence from *our* table is a gap in our
data, never evidence about where the bird lives, so it must never cause a down-rank.
"""

from __future__ import annotations

import httpx
import pytest

from animal_classifier.classify.birds import EBirdAlias, Prediction
from animal_classifier.classify.ebird import EBirdEnricher

# --------------------------------------------------------------------------- #
# 1. CLI config legs
# --------------------------------------------------------------------------- #


def test_hosted_bird_api_is_a_fatal_stub(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    result = run_cli(
        ["classify", str(card), "-o", str(tmp_path / "out"), "--detector", "scripted",
         "--bird-provider", "hosted_bird_api"]
    )
    assert result.returncode == 3, result.stderr
    assert "hosted_bird_api" in result.stderr


def test_ebird_enrich_without_key_is_fatal(run_cli, tmp_path):
    card = tmp_path / "card"
    card.mkdir()
    result = run_cli(
        ["classify", str(card), "-o", str(tmp_path / "out"), "--detector", "scripted",
         "--bird-provider", "ebird_enrich"]
    )
    assert result.returncode == 3, result.stderr
    assert "EBIRD_API_KEY" in result.stderr


# --------------------------------------------------------------------------- #
# Shared fixtures for the provider legs
# --------------------------------------------------------------------------- #

#: Three candidates: one aliased-and-observed, one aliased-but-unobserved, one with
#: no alias row at all. Scores chosen so the down-rank changes the order.
CANDIDATES = (
    Prediction("brewers_blackbird", "Brewer's Blackbird", "Euphagus cyanocephalus", 0.55),
    Prediction("arctic_tern", "Arctic Tern", "Sterna paradisaea", 0.40),
    Prediction("mystery_finch", "Mystery Finch", None, 0.30),
)

ALIASES = {
    "brewers_blackbird": EBirdAlias(
        "brewers_blackbird", "Brewer's Blackbird", "Euphagus cyanocephalus"
    ),
    "arctic_tern": EBirdAlias("arctic_tern", "Arctic Tern", "Sterna paradisaea"),
    # mystery_finch deliberately absent.
}

NAIROBI = (-1.29, 36.82)


def _enricher(tmp_path, handler=None, key="dummy-key"):
    """Build the provider, optionally with a MockTransport client factory."""
    factory = None
    if handler is not None:
        def factory():
            return httpx.Client(transport=httpx.MockTransport(handler))
    return EBirdEnricher(
        api_key=key,
        aliases=ALIASES,
        min_confidence=0.45,
        output_root=tmp_path,
        **({"client_factory": factory} if factory else {}),
    )


# --------------------------------------------------------------------------- #
# 2. Degradation legs
# --------------------------------------------------------------------------- #


def test_no_gps_makes_no_network_call_and_keeps_the_ranking(tmp_path):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, json=[])

    result = _enricher(tmp_path, handler).refine(CANDIDATES, gps=None)
    assert result.status == "no_gps"
    assert calls == [], "no GPS must mean no network call at all (§5.6)"
    # Ranking untouched; names may be canonicalised.
    assert [p.slug for p in result.predictions] == [p.slug for p in CANDIDATES]
    assert [p.conf for p in result.predictions] == [p.conf for p in CANDIDATES]


def test_unreachable_host_degrades_with_one_warning(tmp_path, caplog):
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("unreachable", request=request)

    enricher = _enricher(tmp_path, handler)
    with caplog.at_level("WARNING"):
        first = enricher.refine(CANDIDATES, gps=NAIROBI)
        second = enricher.refine(CANDIDATES, gps=(10.0, 20.0))

    assert first.status == "unreachable"
    assert second.status == "unreachable"
    assert first.predictions == CANDIDATES, "the coarse result is returned unmodified"

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, (
        f"exactly one warning per run is sanctioned, got {len(warnings)}"
    )


def test_a_rejected_key_is_fatal(tmp_path):
    """§5.6/§10.1: 401/403 is a real misconfiguration, not a degradation."""
    from animal_classifier.errors import ConfigError

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "forbidden"})

    with pytest.raises(ConfigError, match="rejected the API key"):
        _enricher(tmp_path, handler).refine(CANDIDATES, gps=NAIROBI)


# --------------------------------------------------------------------------- #
# 3. The stubbed-transport matching leg
# --------------------------------------------------------------------------- #


def _observation_handler(request: httpx.Request) -> httpx.Response:
    """Two observations: the tern is present, the blackbird is not."""
    assert request.headers.get("X-eBirdApiToken") == "dummy-key", "the key is sent"
    assert "lat" in request.url.params and "lng" in request.url.params
    return httpx.Response(
        200,
        json=[
            {"comName": "Arctic Tern", "sciName": "Sterna paradisaea"},
            {"comName": "Sacred Ibis", "sciName": "Threskiornis aethiopicus"},
        ],
    )


def test_matching_rule_downranks_only_aliased_and_unobserved(tmp_path):
    result = _enricher(tmp_path, _observation_handler).refine(CANDIDATES, gps=NAIROBI)
    assert result.status == "refined", "the order changed, so this is a refinement"

    by_slug = {p.slug: p for p in result.predictions}

    # Aliased AND observed -> score untouched.
    assert by_slug["arctic_tern"].conf == pytest.approx(0.40)

    # Aliased but NOT observed -> exactly x0.25.
    assert by_slug["brewers_blackbird"].conf == pytest.approx(0.55 * 0.25)

    # No alias row -> EXEMPT, bit-identical score. This is the assertion that proves
    # absence from our table is not treated as range evidence.
    assert by_slug["mystery_finch"].conf == 0.30

    # The observed species now outranks the demoted one.
    order = [p.slug for p in result.predictions]
    assert order.index("arctic_tern") < order.index("brewers_blackbird")


def test_the_response_is_cached_by_rounded_coordinates(tmp_path):
    """§5.6: one call per (round(lat,2), round(lon,2)), cached under output_root."""
    calls: list[httpx.Request] = []

    def counting_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _observation_handler(request)

    enricher = _enricher(tmp_path, counting_handler)
    enricher.refine(CANDIDATES, gps=NAIROBI)
    enricher.refine(CANDIDATES, gps=(-1.2903, 36.8201))  # same to 2 dp
    assert len(calls) == 1, "the second lookup is served from the cache"

    from animal_classifier.classify.ebird import CACHE_NAME

    assert (tmp_path / CACHE_NAME).is_file(), "the cache is persisted under output_root"


def test_a_demoted_top1_can_legitimately_become_unknown(tmp_path):
    """The re-gate: a down-ranked winner falling below the gate is a real outcome."""
    candidates = (
        Prediction("brewers_blackbird", "Brewer's Blackbird", "Euphagus cyanocephalus", 0.50),
        Prediction("mystery_finch", "Mystery Finch", None, 0.10),
    )
    enricher = EBirdEnricher(
        api_key="dummy-key", aliases=ALIASES, min_confidence=0.45,
        output_root=tmp_path,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(_observation_handler)),
    )
    result = enricher.refine(candidates, gps=NAIROBI)
    top = result.predictions[0]
    # 0.50 x 0.25 = 0.125, below min_confidence -> the caller will file `unknown`.
    assert max(p.conf for p in result.predictions) < 0.45
    assert top.conf < 0.45
