"""Data quality checks for the demo dataset."""
import pytest

from src.data_loader import load_animals
from src.schemas import AdoptionStatus, Species, TriState

COMPATIBILITY_FIELDS = ["good_with_dogs", "good_with_cats", "good_with_children"]


@pytest.fixture
def animals():
    return load_animals()


def test_has_dogs_and_cats(animals):
    species = {a.species for a in animals}
    assert Species.DOG in species
    assert Species.CAT in species


def test_enough_available_animals(animals):
    available = [a for a in animals if a.status == AdoptionStatus.AVAILABLE]
    assert len(available) >= 20


@pytest.mark.parametrize("field", COMPATIBILITY_FIELDS)
def test_compatibility_has_all_three_values(animals, field):
    values = {getattr(a, field) for a in animals}
    assert values == {TriState.YES, TriState.NO, TriState.UNKNOWN}


@pytest.mark.parametrize("field", COMPATIBILITY_FIELDS)
def test_known_compatibility_has_evidence(animals, field):
    missing = [
        a.animal_id for a in animals
        if getattr(a, field) != TriState.UNKNOWN and field not in a.evidence
    ]
    assert missing == [], f"{field} is yes/no without evidence: {missing}"
