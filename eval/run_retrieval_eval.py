"""Evaluate policy retrieval on hand-labeled questions. Run: python -m eval.run_retrieval_eval"""
import json
from pathlib import Path

from src.rag import PolicyRetriever, load_policy_chunks

QUESTIONS_PATH = Path(__file__).resolve().parent / "rag_questions.json"
TOP_K = 3


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    retriever = PolicyRetriever(load_policy_chunks())

    hits_at_1, hits_at_k, reciprocal_ranks = [], [], []
    answerable_top_scores, unanswerable_top_scores = [], []

    for q in questions:
        results = retriever.retrieve(q["question"], top_k=TOP_K)
        returned = [r.chunk.chunk_id for r in results]
        top_score = results[0].score
        relevant = set(q["relevant_chunk_ids"])

        if relevant:
            ranks = [i + 1 for i, chunk_id in enumerate(returned) if chunk_id in relevant]
            hits_at_1.append(returned[0] in relevant)
            hits_at_k.append(bool(ranks))
            reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)
            answerable_top_scores.append(top_score)
            status = f"HIT@{ranks[0]}" if ranks else "MISS"
        else:
            unanswerable_top_scores.append(top_score)
            status = "UNANSWERABLE"

        print(f"{q['qid']}  {status:13s} top={top_score:.2f}  {returned[0]}")

    n = len(hits_at_k)
    print("\n=== Retrieval (answerable questions) ===")
    print(f"Hit@1:       {sum(hits_at_1)}/{n} = {sum(hits_at_1) / n:.2f}")
    print(f"Hit@{TOP_K}:       {sum(hits_at_k)}/{n} = {sum(hits_at_k) / n:.2f}")
    print(f"MRR@{TOP_K}:       {sum(reciprocal_ranks) / n:.2f}")

    print("\n=== Top-1 similarity score: can it separate answerable from unanswerable? ===")
    print(f"Answerable:   min={min(answerable_top_scores):.2f}  "
          f"mean={sum(answerable_top_scores) / len(answerable_top_scores):.2f}")
    print(f"Unanswerable: max={max(unanswerable_top_scores):.2f}  "
          f"mean={sum(unanswerable_top_scores) / len(unanswerable_top_scores):.2f}")


if __name__ == "__main__":
    main()
