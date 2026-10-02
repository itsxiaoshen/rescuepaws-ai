"""Load and validate animal profiles from JSON."""
import json
from pathlib import Path

from src.schemas import AnimalProfile

DEFAULT_ANIMALS_PATH = Path(__file__).resolve().parent.parent / "data" / "animals_demo.json"


def load_animals(path: Path = DEFAULT_ANIMALS_PATH) -> list[AnimalProfile]:
    """Load animals from a JSON file. Raises if any record is invalid or IDs are duplicated."""
    with open(path, encoding="utf-8") as f:
        raw_records = json.load(f)

    animals = [AnimalProfile.model_validate(record) for record in raw_records]

    seen: set[str] = set()
    for animal in animals:
        if animal.animal_id in seen:
            raise ValueError(f"Duplicate animal_id: {animal.animal_id}")
        seen.add(animal.animal_id)

    return animals


DEFAULT_CONFLICTS_PATH = DEFAULT_ANIMALS_PATH.parent / "record_conflicts.json"


def load_record_conflicts(path: Path = DEFAULT_CONFLICTS_PATH) -> dict[str, list[dict]]:
    """Conflicts found by the record audit, keyed by animal ID. Empty if no audit has run."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
