"""Evaluate grounded policy answers over several runs (calls the LLM API).
Run: python -m eval.run_answer_eval [--runs 3]

Uses the dev and held-out question sets. Each answered question is also checked by an
LLM judge: is every claim in the answer supported by the retrieved evidence?
"""
import argparse
import json
from pathlib import Path

from eval.judge import judge_groundedness
from src.llm import DEFAULT_MODEL
from src.rag import PolicyRetriever, answer_policy_question, format_evidence, load_policy_chunks

EVAL_DIR = Path(__file__).resolve().parent
QUESTION_FILES = ["rag_questions.json", "rag_questions_holdout.json"]


def run_once(retriever: PolicyRetriever, questions: list[dict], verbose: bool) -> dict:
    answered, correct_citations, abstained_correctly, grounded = [], [], [], []
    problems = []

    for q in questions:
        result = answer_policy_question(q["question"], retriever)
        relevant = set(q["relevant_chunk_ids"])
        cited = {c.chunk_id for c in result.citations}

        if relevant:
            answered.append(result.answerable)
            if not result.answerable:
                problems.append(f"{q['qid']} wrongly abstained")
        else:
            abstained_correctly.append(not result.answerable)
            if result.answerable:
                problems.append(f"{q['qid']} answered an unanswerable question: {result.answer}")

        if result.answerable:
            if relevant:
                correct_citations.append(bool(relevant & cited))
            judgment = judge_groundedness(q["question"], result.answer, format_evidence(result.retrieved))
            grounded.append(judgment.grounded)
            if not judgment.grounded:
                problems.append(f"{q['qid']} ungrounded: {judgment.unsupported_claims}")

    if verbose:
        for problem in problems:
            print(f"   - {problem}")
    return {
        "Answerable answered": sum(answered) / len(answered),
        "Citation correct": sum(correct_citations) / len(correct_citations),
        "Unanswerable abstained": sum(abstained_correctly) / len(abstained_correctly),
        "Answers grounded (judge)": sum(grounded) / len(grounded),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    questions = []
    for name in QUESTION_FILES:
        questions += json.loads((EVAL_DIR / name).read_text(encoding="utf-8"))
    retriever = PolicyRetriever(load_policy_chunks())
    print(f"Model: {DEFAULT_MODEL}, {len(questions)} questions, {args.runs} runs\n")

    runs = []
    for i in range(1, args.runs + 1):
        print(f"Run {i}:")
        runs.append(run_once(retriever, questions, verbose=True))

    print(f"\n{'Metric':28s} {'mean':>6s} {'min':>6s} {'max':>6s}")
    for metric in runs[0]:
        values = [run[metric] for run in runs]
        print(f"{metric:28s} {sum(values) / len(values):6.2f} {min(values):6.2f} {max(values):6.2f}")


if __name__ == "__main__":
    main()
