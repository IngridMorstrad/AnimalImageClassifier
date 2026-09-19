"""E23: bird provider configuration and the eBird re-ranker (§5.6).

Three providers, three contracts:

- ``hosted_bird_api`` is a documented stub — selecting it is fatal at config time,
  naming the credentials it would need. It never silently no-ops.
- ``ebird_enrich`` without ``ANIMAL_CLASSIFIER_EBIRD_API_KEY`` is fatal at config
  time (the key *is* a required value there).
- the re-ranker's matching rule is pure and tested directly: an aliased candidate
  not observed nearby is down-ranked x0.25 and can lose the top slot, while a
  candidate with no alias row is exempt (absence from our table is a gap in our
  data, never evidence about the bird's range).
"""

from __future__ import annotations


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


def test_reranker_downranks_unobserved_aliased_candidate() -> None:
    from animal_classifier.classify.birds import (
        EBirdAlias,
        Prediction,
        rerank_by_observations,
    )

    candidates = (
        Prediction("brewers_blackbird", "Brewer's Blackbird", "Euphagus cyanocephalus", 0.55),
        Prediction("arctic_tern", "Arctic Tern", "Sterna paradisaea", 0.40),
    )
    aliases = {
        "brewers_blackbird": EBirdAlias("brewers_blackbird", "Brewer's Blackbird", "Euphagus cyanocephalus"),
        "arctic_tern": EBirdAlias("arctic_tern", "Arctic Tern", "Sterna paradisaea"),
    }
    # Only the tern was observed nearby.
    reranked, changed = rerank_by_observations(
        candidates,
        aliases=aliases,
        observed_sci=frozenset({"sterna paradisaea"}),
        observed_com=frozenset(),
        min_confidence=0.45,
    )
    assert changed
    assert reranked[0].slug == "arctic_tern", "the observed species now wins"
    assert reranked[1].conf == 0.55 * 0.25, "the unobserved one was down-ranked x0.25"


def test_reranker_exempts_candidates_with_no_alias() -> None:
    from animal_classifier.classify.birds import (
        EBirdAlias,
        Prediction,
        rerank_by_observations,
    )

    candidates = (
        Prediction("some_species", "Some Species", None, 0.9),
        Prediction("arctic_tern", "Arctic Tern", "Sterna paradisaea", 0.3),
    )
    # some_species has no alias row -> exempt, score untouched, still wins.
    reranked, _ = rerank_by_observations(
        candidates,
        aliases={"arctic_tern": EBirdAlias("arctic_tern", "Arctic Tern", "Sterna paradisaea")},
        observed_sci=frozenset(),
        observed_com=frozenset(),
        min_confidence=0.45,
    )
    assert reranked[0].slug == "some_species"
    assert reranked[0].conf == 0.9, "a candidate absent from our alias table is untouched"


def test_alias_table_rejects_unknown_key_and_empty_row() -> None:
    import pytest

    from animal_classifier.classify.birds import load_alias_table
    from animal_classifier.errors import ConfigError

    with pytest.raises(ConfigError):
        load_alias_table([("ghost", "X", "Y")], label_space={"arctic_tern"})
    with pytest.raises(ConfigError):
        load_alias_table([("arctic_tern", "", "")], label_space={"arctic_tern"})
