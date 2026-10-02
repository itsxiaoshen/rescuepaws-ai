"""Retrieval-augmented answering over shelter policy documents."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from src.llm import generate_structured
from src.matching import MODEL_NAME

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


class PolicyRetriever:
    def __init__(self, chunks: list[PolicyChunk], model_name: str = MODEL_NAME):
        self.chunks = chunks
        self.model = SentenceTransformer(model_name)
        # Include title and heading so a short section still carries its topic
        texts = [f"{c.title} - {c.section}\n{c.text}" for c in chunks]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Return the top_k most relevant chunks, best first."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding
        best = np.argsort(-scores)[:top_k]
        return [RetrievedChunk(chunk=self.chunks[i], score=float(scores[i])) for i in best]


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
