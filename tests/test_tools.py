import pytest

from src.rag import LLMPolicyAnswer
from src.tools import TOOL_SCHEMAS, ShelterTools


def fake_policy_llm(system, user, output_type):
    return LLMPolicyAnswer(answerable=True, answer="Cats cost $150.", cited_chunk_ids=["fees#Fee schedule"])


@pytest.fixture(scope="module")
def tools():
    return ShelterTools(generate=fake_policy_llm)


def test_every_schema_has_a_matching_tool(tools):
    for schema in TOOL_SCHEMAS:
        name = schema["function"]["name"]
        assert hasattr(tools, name), f"schema {name} has no implementation"


def test_get_profile_found(tools):
    result = tools.execute("get_animal_profile", {"animal_id": "RP-0009"})
    assert result["name"] == "Taro"


def test_get_profile_accepts_lowercase_id(tools):
    assert tools.execute("get_animal_profile", {"animal_id": "rp-0009"})["name"] == "Taro"


def test_get_profile_missing_returns_error(tools):
    result = tools.execute("get_animal_profile", {"animal_id": "RP-9999"})
    assert "error" in result


def test_search_by_duplicate_name_returns_all(tools):
    result = tools.execute("search_animals", {"name": "mochi"})
    assert {a["animal_id"] for a in result["animals"]} == {"RP-0001", "RP-0006"}


def test_search_excludes_unavailable_by_default(tools):
    result = tools.execute("search_animals", {"name": "Pepper"})
    assert [a["animal_id"] for a in result["animals"]] == ["RP-0010"]  # RP-0005 is adopted
    result = tools.execute("search_animals", {"name": "Pepper", "include_unavailable": True})
    assert result["count"] == 2


def test_match_returns_reasons_and_unknowns(tools):
    result = tools.execute("match_animals", {"request": "a calm dog", "species": "dog", "has_children": True})
    assert result["matches"]
    for match in result["matches"]:
        assert "reasons" in match and "unknowns" in match


def test_policy_tool_returns_answer_and_sources(tools):
    result = tools.execute("answer_policy_question", {"question": "How much is a cat?"})
    assert result["found_in_policy"]
    assert result["sources"] == ["fees.md > Fee schedule"]


def test_invalid_argument_value_returns_error(tools):
    result = tools.execute("search_animals", {"species": "dragon"})
    assert "error" in result


def test_unexpected_argument_returns_error(tools):
    result = tools.execute("get_animal_profile", {"animal_id": "RP-0001", "color": "brown"})
    assert "error" in result


def test_unknown_tool_returns_error(tools):
    assert "error" in tools.execute("adopt_animal", {"animal_id": "RP-0001"})
