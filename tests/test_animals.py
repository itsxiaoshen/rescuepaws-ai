import json

import pytest
from pydantic import ValidationError

from src.animal_search import filter_animals, get_animal_profile
from src.data_loader import load_animals
from src.schemas import AnimalProfile, Species, TriState


@pytest.fixture
def animals():
    return load_animals()


def test_load_demo_data(animals):
    assert len(animals) >= 30


def test_get_existing_animal(animals):
    animal = get_animal_profile("RP-0001", animals)
    assert animal is not None
    assert animal.name == "Mochi"


def test_get_missing_animal_returns_none(animals):
    assert get_animal_profile("RP-9999", animals) is None


def test_missing_fields_default_to_unknown():
    animal = AnimalProfile(animal_id="RP-0100", name="Test", species="dog")
    assert animal.good_with_children == TriState.UNKNOWN


def test_invalid_value_is_rejected():
    with pytest.raises(ValidationError):
        AnimalProfile(animal_id="RP-0100", name="Test", species="dragon")


def test_typo_field_is_rejected():
    with pytest.raises(ValidationError):
        AnimalProfile(animal_id="RP-0100", name="Test", species="dog", good_with_kid="yes")


def test_duplicate_ids_are_rejected(tmp_path):
    record = {"animal_id": "RP-0001", "name": "A", "species": "dog"}
    path = tmp_path / "dupes.json"
    path.write_text(json.dumps([record, record]))
    with pytest.raises(ValueError):
        load_animals(path)


def test_filter_excludes_adopted_by_default(animals):
    results = filter_animals(animals)
    assert all(a.status.value == "available" for a in results)
    assert "RP-0005" not in [a.animal_id for a in results]


def test_filter_children_excludes_unknown(animals):
    results = filter_animals(animals, species=Species.DOG, good_with_children=True)
    assert len(results) > 0
    assert all(a.good_with_children == TriState.YES for a in results)
    assert "RP-0002" not in [a.animal_id for a in results]   # Biscuit: unknown with children

def test_evidence_with_unknown_field_is_rejected():
    with pytest.raises(ValidationError):
        AnimalProfile(
            animal_id="RP-0100", name="Test", species="dog",
            evidence={"good_with_kids": "observed"},
        )
