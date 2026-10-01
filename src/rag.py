"""Retrieval over shelter policy documents (the "R" in RAG)."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

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
