"""Structured lookup and filtering over animal profiles."""
from src.schemas import AdoptionStatus, AnimalProfile, Size, Species, TriState


def get_animal_profile(animal_id: str, animals: list[AnimalProfile]) -> AnimalProfile | None:
    """Return the animal with this ID, or None if not found."""
    for animal in animals:
        if animal.animal_id == animal_id:
            return animal
    return None


def filter_animals(
    animals: list[AnimalProfile],
    species: Species | None = None,
    size: Size | None = None,
    status: AdoptionStatus | None = AdoptionStatus.AVAILABLE,
    good_with_dogs: bool = False,
    good_with_cats: bool = False,
    good_with_children: bool = False,
) -> list[AnimalProfile]:
    """Filter by structured attributes.

    Compatibility flags require a recorded YES. Unknown compatibility does NOT pass the filter.
    """
    results = []
    for animal in animals:
        if species is not None and animal.species != species:
            continue
        if size is not None and animal.size != size:
            continue
        if status is not None and animal.status != status:
            continue
        if good_with_dogs and animal.good_with_dogs != TriState.YES:
            continue
        if good_with_cats and animal.good_with_cats != TriState.YES:
            continue
        if good_with_children and animal.good_with_children != TriState.YES:
            continue
        results.append(animal)
    return results
