import pytest

from src.data_loader import load_animals
from src.matching import AnimalMatcher, HouseholdNeeds, animal_to_text
from src.schemas import AdoptionStatus, Species, TriState


@pytest.fixture(scope="module")
def matcher():
    # scope="module": load the embedding model once for all tests in this file
    return AnimalMatcher(load_animals())


def test_text_does_not_mention_compatibility(matcher):
    for animal in matcher.animals:
        text = animal_to_text(animal).lower()
        assert "good with" not in text


def test_only_available_animals_are_matched(matcher):
    results = matcher.match("any friendly animal", top_k=50)
    assert all(r.animal.status == AdoptionStatus.AVAILABLE for r in results)


def test_animals_marked_no_are_excluded(matcher):
    needs = HouseholdNeeds(has_children=True)
    results = matcher.match("a playful dog", needs, top_k=50)
    assert all(r.animal.good_with_children != TriState.NO for r in results)


def test_unknown_is_reported_not_counted_as_reason(matcher):
    needs = HouseholdNeeds(has_children=True)
    results = matcher.match("a calm dog", needs, top_k=50)
    for r in results:
        if r.animal.good_with_children == TriState.UNKNOWN:
            assert any("children" in u for u in r.unknowns)
            assert not any("good with children" in reason for reason in r.reasons)


def test_confirmed_matches_rank_before_unknowns(matcher):
    needs = HouseholdNeeds(has_children=True)
    results = matcher.match("a calm dog", needs, top_k=50)
    has_unknowns = [len(r.unknowns) > 0 for r in results]
    assert has_unknowns == sorted(has_unknowns)


def test_taro_ranks_above_bruno_for_family(matcher):
    # Near-identical descriptions; only Taro is recorded as good with children
    needs = HouseholdNeeds(species=Species.DOG, has_children=True)
    results = matcher.match("a calm large dog for slow walks", needs, top_k=50)
    ids = [r.animal.animal_id for r in results]
    assert ids.index("RP-0009") < ids.index("RP-0029")


def test_semantic_search_finds_rabbit(matcher):
    animal, _ = matcher.search("a rabbit that likes hay and tunnels", top_k=1)[0]
    assert animal.species == Species.OTHER


def test_conflicting_record_is_not_a_confirmed_match():
    conflicts = {"RP-0019": [{"field": "good_with_children", "explanation": "growled at a toddler",
                              "quote": "growled when a visiting toddler reached toward his food bowl."}]}
    matcher = AnimalMatcher(load_animals(), conflicts=conflicts)
    results = matcher.match("a calm dog", HouseholdNeeds(species=Species.DOG, has_children=True), top_k=50)
    otis = next(r for r in results if r.animal.animal_id == "RP-0019")
    assert any("Conflicting records" in u for u in otis.unknowns)
    assert not any("good with children" in reason for reason in otis.reasons)
