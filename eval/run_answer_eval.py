"""Evaluate grounded policy answers (calls the LLM API). Run: python -m eval.run_answer_eval"""
import json
from pathlib import Path

from src.llm import DEFAULT_MODEL
from src.rag import PolicyRetriever, answer_policy_question, load_policy_chunks

QUESTIONS_PATH = Path(__file__).resolve().parent / "rag_questions.json"


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    retriever = PolicyRetriever(load_policy_chunks())
    print(f"Model: {DEFAULT_MODEL}\n")

    answered, correct_citations, abstained_correctly = [], [], []
    wrongly_abstained, wrongly_answered = [], []

    for q in questions:
        result = answer_policy_question(q["question"], retriever)
        relevant = set(q["relevant_chunk_ids"])
        cited = [c.chunk_id for c in result.citations]

        if relevant:
            answered.append(result.answerable)
            if result.answerable:
                correct_citations.append(bool(relevant & set(cited)))
            else:
                wrongly_abstained.append(q["qid"])
        else:
            abstained_correctly.append(not result.answerable)
            if result.answerable:
                wrongly_answered.append(q["qid"])

        label = "ANSWERED " if result.answerable else "ABSTAINED"
        print(f"{q['qid']} {label} {q['question']}")
        print(f"     -> {result.answer}")
        if cited:
            print(f"     cites: {cited}")

    print("\n=== Summary ===")
    print(f"Answerable questions answered:      {sum(answered)}/{len(answered)}")
    print(f"Citation correct (when answered):   {sum(correct_citations)}/{len(correct_citations)}")
    print(f"Unanswerable questions abstained:   {sum(abstained_correctly)}/{len(abstained_correctly)}")
    print(f"Wrongly abstained: {wrongly_abstained}")
    print(f"Wrongly answered:  {wrongly_answered}")


if __name__ == "__main__":
    main()
