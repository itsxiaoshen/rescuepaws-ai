"""Evaluate intake extraction against human-verified expected fields (calls the LLM API).
Run: python -m eval.run_intake_eval"""
import json
from pathlib import Path

from src.intake import COMPATIBILITY_FIELDS, create_intake_draft
from src.llm import DEFAULT_MODEL

CASES_PATH = Path(__file__).resolve().parent / "intake_cases.json"
PHOTO_DIR = Path(__file__).resolve().parent.parent / "data" / "intake_photos"


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    print(f"Model: {DEFAULT_MODEL}\n")

    correct, total = 0, 0
    unsupported_claims = []  # compatibility set to yes/no when the notes don't support it

    for case in cases:
        photo_path = PHOTO_DIR / case["photo"] if case["photo"] else None
        draft = create_intake_draft("RP-0999", case["notes"], photo_path)
        profile = draft.profile.model_dump(mode="json")

        mismatches = []
        for name, expected in case["expected"].items():
            accepted = expected if isinstance(expected, list) else [expected]
            total += 1
            if profile[name] in accepted:
                correct += 1
            else:
                mismatches.append(f"{name}: expected {expected}, got {profile[name]}")
                if name in COMPATIBILITY_FIELDS and expected == "unknown":
                    unsupported_claims.append(f"{case['case_id']}.{name}")

        status = "PASS" if not mismatches else "FAIL"
        print(f"{case['case_id']} {status}  appearance={profile['appearance']!r}  flags={len(draft.review_flags)}")
        for mismatch in mismatches:
            print(f"     - {mismatch}")

    print("\n=== Summary ===")
    print(f"Field accuracy:                     {correct}/{total} = {correct / total:.2f}")
    print(f"Unsupported compatibility claims:   {len(unsupported_claims)} {unsupported_claims}")


if __name__ == "__main__":
    main()
