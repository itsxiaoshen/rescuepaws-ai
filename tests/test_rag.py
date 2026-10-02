import pytest

from src.rag import (
    ABSTAIN_MESSAGE,
    LLMPolicyAnswer,
    PolicyRetriever,
    answer_policy_question,
    chunk_markdown,
    load_policy_chunks,
)


@pytest.fixture(scope="module")
def retriever():
    return PolicyRetriever(load_policy_chunks())


# --- Chunking and retrieval ---

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


# --- Grounded answering. These tests use a fake LLM: no API calls, no cost. ---

def fake_llm(answerable, answer, cited_ids):
    """Build a stand-in for generate_structured that returns a fixed answer."""
    calls = []

    def generate(system, user, output_type):
        calls.append(user)
        return LLMPolicyAnswer(answerable=answerable, answer=answer, cited_chunk_ids=cited_ids)

    generate.calls = calls
    return generate


def test_answer_with_valid_citation_is_returned(retriever):
    fake = fake_llm(True, "Cats cost $150.", ["fees#Fee schedule"])
    result = answer_policy_question("How much is a cat?", retriever, generate=fake)
    assert result.answerable
    assert result.answer == "Cats cost $150."
    assert [c.chunk_id for c in result.citations] == ["fees#Fee schedule"]


def test_llm_abstention_returns_standard_message(retriever):
    fake = fake_llm(False, "", [])
    result = answer_policy_question("Do you sell dog food?", retriever, generate=fake)
    assert not result.answerable
    assert result.answer == ABSTAIN_MESSAGE


def test_answer_citing_unretrieved_chunk_is_rejected(retriever):
    # The LLM claims a source we never gave it: treat the answer as ungrounded
    fake = fake_llm(True, "Training classes are free.", ["made_up#Training"])
    result = answer_policy_question("Do you offer training?", retriever, generate=fake)
    assert not result.answerable
    assert result.answer == ABSTAIN_MESSAGE


def test_answer_without_citations_is_rejected(retriever):
    fake = fake_llm(True, "Yes, of course.", [])
    result = answer_policy_question("Can I pay in cash?", retriever, generate=fake)
    assert not result.answerable


def test_prompt_contains_retrieved_evidence(retriever):
    fake = fake_llm(False, "", [])
    result = answer_policy_question("How much is a cat?", retriever, generate=fake)
    prompt = fake.calls[0]
    for r in result.retrieved:
        assert r.chunk.chunk_id in prompt
