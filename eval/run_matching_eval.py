"""Evaluate adoption matching on hand-labeled cases. Run: python -m eval.run_matching_eval"""
import json
from pathlib import Path

from src.data_loader import load_animals
from src.matching import COMPATIBILITY_CHECKS, AnimalMatcher, HouseholdNeeds
from src.schemas import AdoptionStatus, AnimalProfile, Size, Species, TriState

CASES_PATH = Path(__file__).resolve().parent / "matching_cases.json"
TOP_K = 5


def build_needs(raw: dict) -> HouseholdNeeds:
    return HouseholdNeeds(
        species=Species(raw["species"]) if "species" in raw else None,
        size=Size(raw["size"]) if "size" in raw else None,
        has_children=raw.get("has_children", False),
        has_dogs=raw.get("has_dogs", False),
        has_cats=raw.get("has_cats", False),
    )


def violates_constraints(animal: AnimalProfile, needs: HouseholdNeeds) -> bool:
    """True if recommending this animal breaks a hard requirement."""
    if animal.status != AdoptionStatus.AVAILABLE:
        return True
    if needs.species is not None and animal.species != needs.species:
        return True
    if needs.size is not None and animal.size != needs.size:
        return True
    for need_attr, animal_attr, _ in COMPATIBILITY_CHECKS:
        if getattr(needs, need_attr) and getattr(animal, animal_attr) == TriState.NO:
            return True
    return False


def evaluate(cases: list[dict], get_results, verbose: bool) -> dict:
    hits, reciprocal_ranks, precisions = [], [], []
    no_match_passed, unsafe_cases, violation_cases = [], [], []

    for case in cases:
        needs = build_needs(case["needs"])
        results = get_results(case["request"], needs)   # list of (animal, has_unknowns)
        returned = [animal.animal_id for animal, _ in results]
        acceptable = set(case["acceptable_ids"])

        if acceptable:
            ranks = [i + 1 for i, animal_id in enumerate(returned) if animal_id in acceptable]
            hits.append(bool(ranks))
            reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)
            precisions.append(len(ranks) / TOP_K)
            status = f"HIT@{ranks[0]}  P@5={len(ranks) / TOP_K:.1f}" if ranks else "MISS"
        else:
            # No-match case: nothing may be presented as a fully confirmed match
            passed = all(has_unknowns for _, has_unknowns in results)
            no_match_passed.append(passed)
            status = "NO-MATCH PASS" if passed else "NO-MATCH FAIL"

        # Unsafe = an animal that must not be recommended, presented as a confirmed match
        unsafe = [
            animal.animal_id for animal, has_unknowns in results
            if animal.animal_id in case["unsafe_ids"] and not has_unknowns
        ]
        if unsafe:
            unsafe_cases.append(case["case_id"])
            status += f"  UNSAFE {unsafe}"
        if any(violates_constraints(animal, needs) for animal, _ in results):
            violation_cases.append(case["case_id"])
            status += "  VIOLATION"

        if verbose:
            print(f"{case['case_id']}  {status:40s} {returned}")

    return {
        f"Hit@{TOP_K}": sum(hits) / len(hits),
        f"MRR@{TOP_K}": sum(reciprocal_ranks) / len(reciprocal_ranks),
        f"Precision@{TOP_K}": sum(precisions) / len(precisions),
        "No-match passed": f"{sum(no_match_passed)}/{len(no_match_passed)}",
        "Unsafe shown as confirmed": f"{len(unsafe_cases)} {unsafe_cases}",
        "Cases with constraint violation": f"{len(violation_cases)} {violation_cases}",
    }


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    animals = load_animals()
    matcher = AnimalMatcher(animals)                        # uses data/record_conflicts.json
    matcher_no_audit = AnimalMatcher(animals, conflicts={})  # same, but ignores the audit

    def baseline(request, needs):
        # Pure semantic search: ignores status, species, size, and compatibility
        return [(animal, False) for animal, _ in matcher.search(request, top_k=TOP_K)]

    def hybrid_no_audit(request, needs):
        return [(r.animal, bool(r.unknowns)) for r in matcher_no_audit.match(request, needs, top_k=TOP_K)]

    def hybrid(request, needs):
        return [(r.animal, bool(r.unknowns)) for r in matcher.match(request, needs, top_k=TOP_K)]

    print("=== Hybrid matching + record audit, per case ===")
    columns = {
        "Hybrid+audit": evaluate(cases, hybrid, verbose=True),
        "Baseline": evaluate(cases, baseline, verbose=False),
        "Hybrid": evaluate(cases, hybrid_no_audit, verbose=False),
    }

    print(f"\n{'Metric':32s} {'Baseline':>10s} {'Hybrid':>10s} {'Hybrid+audit':>14s}")
    for name in columns["Hybrid+audit"]:
        cells = []
        for column in ["Baseline", "Hybrid", "Hybrid+audit"]:
            value = columns[column][name]
            cells.append(f"{value:.2f}" if isinstance(value, float) else value.split()[0])
        print(f"{name:32s} {cells[0]:>10s} {cells[1]:>10s} {cells[2]:>14s}")


if __name__ == "__main__":
    main()
