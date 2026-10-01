"""Print a summary of the animal dataset. Run: python -m src.dataset_stats"""
from collections import Counter

from src.data_loader import load_animals

ENUM_FIELDS = [
    "species", "size", "status", "sex", "vaccination_status", "sterilized",
    "good_with_dogs", "good_with_cats", "good_with_children",
]


def main() -> None:
    animals = load_animals()
    print(f"Total animals: {len(animals)}\n")

    for field in ENUM_FIELDS:
        counts = Counter(getattr(animal, field).value for animal in animals)
        print(f"{field:20s} {dict(counts)}")

    missing_behavior = sum(animal.behavior_notes is None for animal in animals)
    missing_age = sum(animal.estimated_age_months is None for animal in animals)
    print(f"\nbehavior_notes not recorded: {missing_behavior}")
    print(f"estimated_age not recorded: {missing_age}")


if __name__ == "__main__":
    main()
