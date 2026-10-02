"""Check the judge itself on answers with known labels (calls the LLM API).
Run: python -m eval.run_judge_validation"""
import json
from pathlib import Path

from eval.judge import judge_groundedness
from src.rag import load_policy_chunks

CASES_PATH = Path(__file__).resolve().parent / "judge_validation.json"


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    chunks = {c.chunk_id: c for c in load_policy_chunks()}

    correct, missed_hallucinations, false_alarms = 0, [], []
    for i, case in enumerate(cases, start=1):
        chunk = chunks[case["chunk_id"]]
        evidence = f"[{chunk.chunk_id}]\n{chunk.text}"
        judgment = judge_groundedness(case["question"], case["answer"], evidence)
        ok = judgment.grounded == case["grounded"]
        correct += ok
        if not ok and not case["grounded"]:
            missed_hallucinations.append(i)
        if not ok and case["grounded"]:
            false_alarms.append(i)
        label = "grounded" if case["grounded"] else "UNGROUNDED"
        print(f"{i:2d} {'OK  ' if ok else 'MISS'} expected {label:10s} unsupported={judgment.unsupported_claims}")

    print(f"\nJudge accuracy: {correct}/{len(cases)}")
    print(f"Missed hallucinations (judge said grounded, but it wasn't): {missed_hallucinations}")
    print(f"False alarms (judge flagged a correct answer): {false_alarms}")


if __name__ == "__main__":
    main()
