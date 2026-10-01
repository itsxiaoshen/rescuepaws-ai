"""Compare pure semantic search with constrained matching. Run: python -m src.demo_matching"""
from src.data_loader import load_animals
from src.matching import AnimalMatcher, HouseholdNeeds
from src.schemas import Species

REQUEST = "I live in an apartment and want a calm dog that likes slow walks."


def main() -> None:
    print("Loading model and embedding animals...")
    matcher = AnimalMatcher(load_animals())
    print(f"\nRequest: {REQUEST}\n")

    print("=== 1. Pure semantic search (no constraints) ===")
    for animal, score in matcher.search(REQUEST, top_k=5):
        print(f"{score:.2f}  {animal.animal_id} {animal.name:10s} "
              f"status={animal.status.value:12s} kids={animal.good_with_children.value}")

    print("\n=== 2. Matching for a household with a young child ===")
    needs = HouseholdNeeds(species=Species.DOG, has_children=True)
    for result in matcher.match(REQUEST, needs, top_k=5):
        print(f"\n{result.animal.animal_id} {result.animal.name} (score {result.score:.2f})")
        for reason in result.reasons:
            print(f"   + {reason}")
        for unknown in result.unknowns:
            print(f"   ? {unknown}")


if __name__ == "__main__":
    main()
