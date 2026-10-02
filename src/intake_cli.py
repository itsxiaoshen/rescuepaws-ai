"""Volunteer intake with a human review step.

Run: python -m src.intake_cli --photo data/intake_photos/black_dog.jpg --notes "Name: Shadow. Male, ..."
"""
import argparse
import json
from pathlib import Path

from src.data_loader import DEFAULT_ANIMALS_PATH, load_animals
from src.intake import IntakeDraft, create_intake_draft, next_animal_id

APPROVED_PATH = DEFAULT_ANIMALS_PATH.parent / "intake_approved.json"


def load_approved() -> list[dict]:
    if APPROVED_PATH.exists():
        return json.loads(APPROVED_PATH.read_text(encoding="utf-8"))
    return []


def print_draft(draft: IntakeDraft) -> None:
    profile = draft.profile
    print("\n=== Draft profile (NOT saved yet) ===")
    print(f"{'field':22s} {'value':45s} source")
    for name, value in profile.model_dump(mode="json", exclude={"evidence"}).items():
        source = profile.evidence.get(name, "")
        print(f"{name:22s} {str(value)[:45]:45s} {source[:70]}")
    if draft.review_flags:
        print("\nPlease check:")
        for flag in draft.review_flags:
            print(f"  ! {flag}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a draft animal profile from a photo and notes.")
    parser.add_argument("--notes", required=True, help="Volunteer intake notes")
    parser.add_argument("--photo", type=Path, help="Path to an intake photo (optional)")
    args = parser.parse_args()

    approved = load_approved()
    existing_ids = [a.animal_id for a in load_animals()] + [a["animal_id"] for a in approved]
    print("Creating draft (this calls the LLM)...")
    draft = create_intake_draft(next_animal_id(existing_ids), args.notes, args.photo)
    print_draft(draft)

    # Human review: nothing is saved unless a volunteer explicitly approves it
    if input("\nApprove and save this profile? [y/N] ").strip().lower() != "y":
        print("Not saved.")
        return
    approved.append(draft.profile.model_dump(mode="json"))
    APPROVED_PATH.write_text(json.dumps(approved, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {draft.profile.animal_id} to {APPROVED_PATH}")


if __name__ == "__main__":
    main()
