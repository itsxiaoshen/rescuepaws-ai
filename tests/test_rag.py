import pytest

from src.rag import PolicyRetriever, chunk_markdown, load_policy_chunks


@pytest.fixture(scope="module")
def retriever():
    return PolicyRetriever(load_policy_chunks())


def test_chunk_markdown_splits_by_section(tmp_path):
    path = tmp_path / "sample.md"
    path.write_text("# Sample Policy\n\n## First\nAlpha text.\n\n## Second\nBeta text.\n")
    chunks = chunk_markdown(path)
    assert [c.section for c in chunks] == ["First", "Second"]
    assert chunks[0].title == "Sample Policy"
    assert chunks[0].text == "Alpha text."
    assert chunks[1].chunk_id == "sample#Second"


def test_all_policy_chunks_have_text():
    chunks = load_policy_chunks()
    assert len(chunks) >= 20
    assert all(c.text for c in chunks)


def test_chunk_ids_are_unique():
    ids = [c.chunk_id for c in load_policy_chunks()]
    assert len(ids) == len(set(ids))


def test_retrieves_fee_schedule(retriever):
    results = retriever.retrieve("How much does it cost to adopt a cat?", top_k=3)
    assert "fees#Fee schedule" in [r.chunk.chunk_id for r in results]


def test_results_sorted_by_score(retriever):
    results = retriever.retrieve("Can I return my dog?", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)

