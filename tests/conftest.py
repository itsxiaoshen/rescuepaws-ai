"""Fixtures shared by several test files. pytest loads this file automatically."""
import pytest

from src.rag import LLMPolicyAnswer
from src.tools import ShelterTools


def fake_policy_llm(system, user, output_type):
    return LLMPolicyAnswer(answerable=True, answer="Cats cost $150.", cited_chunk_ids=["fees#Fee schedule"])


@pytest.fixture(scope="session")
def tools():
    """Real data and embedding models, with a fake LLM for policy answers. Loaded once."""
    return ShelterTools(generate=fake_policy_llm)
