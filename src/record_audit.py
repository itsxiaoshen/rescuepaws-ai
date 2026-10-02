"""Record audit: find animals marked "good with X" whose notes describe a worrying incident with X.

Runs offline, whenever records change: python -m src.record_audit
Results are written to data/record_conflicts.json for staff to review. Matching reads
that file, so no LLM call is needed when an adopter searches.

Design: the code decides WHAT to check (only compatibility fields recorded as "yes",
because those are the claims that could put someone at risk). The LLM answers one
narrow reading question per field.
"""
import json

from pydantic import BaseModel

from src.data_loader import DEFAULT_CONFLICTS_PATH, load_animals
from src.intake import normalize_text
from src.llm import generate_structured
from src.schemas import AnimalProfile, TriState

# animal field -> how to describe it in the question
CHECKED_FIELDS = {
    "good_with_children": "children",
    "good_with_dogs": "other dogs",
    "good_with_cats": "cats",
}


class ConcernCheck(BaseModel):
    concern_found: bool
    quote: str        # copied word for word from the notes; empty if no concern
    explanation: str


AUDIT_SYSTEM_PROMPT = """You review animal shelter notes for safety concerns. Answer only from the notes provided."""


def record_text(animal: AnimalProfile) -> str:
    return "\n".join([animal.behavior_notes or "", animal.medical_notes or "", *animal.evidence.values()])


def check_field(animal: AnimalProfile, label: str, generate=generate_structured) -> ConcernCheck:
    question = (
        f"This animal's record says it is GOOD WITH {label.upper()}.\n\n"
        f"Notes:\n{record_text(animal)}\n\n"
        f"Do the notes describe any worrying behavior toward {label} (for example growling, snapping, "
        f"biting, lunging, chasing, or fear) that staff should know about before placing this animal "
        f"in a home with {label}? Only count behavior directed at {label}; general shyness, or behavior "
        f"toward other kinds of animals or people, does not count. If yes, quote the exact sentence."
    )
    return generate(AUDIT_SYSTEM_PROMPT, question, ConcernCheck)


def audit_animal(animal: AnimalProfile, generate=generate_structured) -> list[dict]:
    conflicts = []
    text = normalize_text(record_text(animal))
    for field, label in CHECKED_FIELDS.items():
        if getattr(animal, field) != TriState.YES:
            continue
        check = check_field(animal, label, generate)
        # Keep a concern only if its quote really appears in the record
        if check.concern_found and check.quote and normalize_text(check.quote) in text:
            conflicts.append({"field": field, "explanation": check.explanation, "quote": check.quote})
    return conflicts


def main() -> None:
    conflicts = {}
    for animal in load_animals():
        found = audit_animal(animal)
        if found:
            conflicts[animal.animal_id] = found
            for c in found:
                print(f"{animal.animal_id} {animal.name}: {c['field']} - {c['explanation']}")
    DEFAULT_CONFLICTS_PATH.write_text(json.dumps(conflicts, indent=2) + "\n", encoding="utf-8")
    print(f"\n{len(conflicts)} animal(s) with conflicts. Saved to {DEFAULT_CONFLICTS_PATH.name}")


if __name__ == "__main__":
    main()
