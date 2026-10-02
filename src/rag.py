"""Retrieval-augmented answering over shelter policy documents."""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.llm import generate_structured

# Chosen in Phase 6 by comparing models on dev and held-out questions (see eval/eval_results.md)
POLICY_MODEL_NAME = "BAAI/bge-small-en-v1.5"

DEFAULT_POLICY_DIR = Path(__file__).resolve().parent.parent / "data" / "shelter_policies"


@dataclass
class PolicyChunk:
    chunk_id: str   # e.g. "fees#Fee schedule"
    source: str     # file name, e.g. "fees.md"
    title: str      # document title, e.g. "Adoption Fees"
    section: str    # section heading, e.g. "Fee schedule"
    text: str


@dataclass
class RetrievedChunk:
    chunk: PolicyChunk
    score: float


def chunk_markdown(path: Path) -> list[PolicyChunk]:
    """Split one markdown file into chunks, one per '## ' section."""
    text = path.read_text(encoding="utf-8")
    header, *sections = text.split("\n## ")
    title = header.strip().splitlines()[0].removeprefix("# ").strip()

    chunks = []
    for section in sections:
        heading, _, body = section.partition("\n")
        heading = heading.strip()
        chunks.append(PolicyChunk(
            chunk_id=f"{path.stem}#{heading}",
            source=path.name,
            title=title,
            section=heading,
            text=body.strip(),
        ))
    return chunks


def load_policy_chunks(policy_dir: Path = DEFAULT_POLICY_DIR) -> list[PolicyChunk]:
    chunks = []
    for path in sorted(policy_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    return chunks


STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "be",
    "i", "my", "me", "you", "your", "we", "our", "it", "can", "do", "does", "will", "what",
    "how", "if", "at", "by", "this", "that", "from", "as", "any", "there", "have", "has",
}


def tokenize(text: str) -> list[str]:
    """Lowercase words without very common words, for keyword (BM25) search."""
    return [word for word in re.findall(r"[a-z0-9]+", text.lower()) if word not in STOPWORDS]


def reciprocal_rank_fusion(rankings: list[np.ndarray], k: int = 60) -> dict[int, float]:
    """Combine several rankings: each item gets 1 / (k + rank) from every ranking it appears in."""
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            fused[int(index)] = fused.get(int(index), 0.0) + 1 / (k + rank)
    return fused


class PolicyRetriever:
    """Finds relevant policy chunks. mode: "embedding", "bm25" (keywords), or "hybrid" (both)."""

    def __init__(self, chunks: list[PolicyChunk], model_name: str = POLICY_MODEL_NAME, mode: str = "embedding"):
        self.chunks = chunks
        self.mode = mode
        self.model = SentenceTransformer(model_name)
        # Include title and heading so a short section still carries its topic
        texts = [f"{c.title} - {c.section}\n{c.text}" for c in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.bm25 = BM25Okapi([tokenize(text) for text in texts])

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks, best first."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        embedding_scores = self.embeddings @ query_embedding
        bm25_scores = self.bm25.get_scores(tokenize(query))

        if self.mode == "embedding":
            scores = {i: float(s) for i, s in enumerate(embedding_scores)}
        elif self.mode == "bm25":
            scores = {i: float(s) for i, s in enumerate(bm25_scores)}
        else:
            scores = reciprocal_rank_fusion([np.argsort(-embedding_scores), np.argsort(-bm25_scores)])

        best = sorted(scores, key=scores.get, reverse=True)[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=scores[i]) for i in best]


# ---------------------------------------------------------------------------
# Grounded answering (the "G" in RAG)
# ---------------------------------------------------------------------------

ABSTAIN_MESSAGE = (
    "I couldn't find the answer in the shelter's policy documents. "
    "Please contact adoptions@rescuepaws.example."
)

ANSWER_SYSTEM_PROMPT = """You answer adopters' questions for an animal shelter, using ONLY the policy excerpts provided.

Rules:
- Use only facts stated in the excerpts. Do not use outside knowledge, and do not guess facts the excerpts don't state.
- Applying a rule from the excerpts to the adopter's situation is expected, not guessing. For example, if a policy says "We do not accept cash", answer "No" to "Can I pay in cash?".
- If the excerpts don't contain the answer, set answerable to false and leave answer empty.
- Don't conclude that something is excluded or not allowed just because the excerpts don't mention it. For example, if a list of what the fee covers doesn't mention insurance, the excerpts don't say whether insurance is included: set answerable to false.
- If the question asks for medical advice about a specific animal (diagnosis, treatment, medication), set answerable to false.
- List the chunk_id of every excerpt you used in cited_chunk_ids.
- Keep the answer short and friendly."""


class LLMPolicyAnswer(BaseModel):
    """The structured output we ask the LLM for."""
    answerable: bool
    answer: str
    cited_chunk_ids: list[str]


@dataclass
class PolicyAnswer:
    question: str
    answer: str
    answerable: bool
    citations: list[PolicyChunk]
    retrieved: list[RetrievedChunk]


def format_evidence(retrieved: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{r.chunk.chunk_id}] ({r.chunk.title} - {r.chunk.section})\n{r.chunk.text}"
        for r in retrieved
    )


def answer_policy_question(
    question: str,
    retriever: PolicyRetriever,
    top_k: int = 3,
    generate=generate_structured,
) -> PolicyAnswer:
    """Retrieve evidence, ask the LLM to answer from it, and verify the citations."""
    retrieved = retriever.retrieve(question, top_k=top_k)
    user_message = f"Policy excerpts:\n\n{format_evidence(retrieved)}\n\nQuestion: {question}"
    llm_answer = generate(ANSWER_SYSTEM_PROMPT, user_message, LLMPolicyAnswer)

    # Keep only citations that point to chunks we actually gave the LLM
    retrieved_by_id = {r.chunk.chunk_id: r.chunk for r in retrieved}
    citations = [retrieved_by_id[cid] for cid in llm_answer.cited_chunk_ids if cid in retrieved_by_id]

    # Grounding guard: an answer without at least one valid citation is not trusted
    if not llm_answer.answerable or not citations:
        return PolicyAnswer(question, ABSTAIN_MESSAGE, False, [], retrieved)
    return PolicyAnswer(question, llm_answer.answer, True, citations, retrieved)
