"""Compare retrieval setups on dev and held-out questions. Run: python -m eval.run_retrieval_eval

The dev set (rag_questions.json) was used to choose the setup, so its scores are optimistic.
The held-out set (rag_questions_holdout.json) was not used for any decision.
"""
import json
from pathlib import Path

from src.rag import POLICY_MODEL_NAME, PolicyRetriever, load_policy_chunks

EVAL_DIR = Path(__file__).resolve().parent
TOP_K = 3
SETUPS = [
    ("all-MiniLM-L6-v2", "embedding"),
    ("all-MiniLM-L6-v2", "hybrid"),
    (POLICY_MODEL_NAME, "embedding"),
    (POLICY_MODEL_NAME, "hybrid"),
]


def evaluate(retriever: PolicyRetriever, questions: list[dict]) -> dict:
    hits_at_1, hits_at_k, reciprocal_ranks, misses = [], [], [], []
    for q in questions:
        relevant = set(q["relevant_chunk_ids"])
        if not relevant:
            continue  # unanswerable questions are evaluated in run_answer_eval
        returned = [r.chunk.chunk_id for r in retriever.retrieve(q["question"], top_k=TOP_K)]
        ranks = [i + 1 for i, chunk_id in enumerate(returned) if chunk_id in relevant]
        hits_at_1.append(returned[0] in relevant)
        hits_at_k.append(bool(ranks))
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)
        if not ranks:
            misses.append(q["qid"])
    n = len(hits_at_k)
    return {"Hit@1": sum(hits_at_1) / n, "Hit@3": sum(hits_at_k) / n,
            "MRR@3": sum(reciprocal_ranks) / n, "misses": misses}


def main() -> None:
    chunks = load_policy_chunks()
    question_sets = {
        "dev": json.loads((EVAL_DIR / "rag_questions.json").read_text(encoding="utf-8")),
        "held-out": json.loads((EVAL_DIR / "rag_questions_holdout.json").read_text(encoding="utf-8")),
    }
    print(f"{'Model':26s} {'Mode':10s} {'Set':9s} {'Hit@1':>6s} {'Hit@3':>6s} {'MRR@3':>6s}  Misses")
    for model_name, mode in SETUPS:
        retriever = PolicyRetriever(chunks, model_name=model_name, mode=mode)
        for set_name, questions in question_sets.items():
            r = evaluate(retriever, questions)
            print(f"{model_name:26s} {mode:10s} {set_name:9s} "
                  f"{r['Hit@1']:6.2f} {r['Hit@3']:6.2f} {r['MRR@3']:6.2f}  {r['misses']}")


if __name__ == "__main__":
    main()
