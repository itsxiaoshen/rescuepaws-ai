"""API tests with fake LLMs: no API calls."""
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from src.agent import ShelterAgent
from src.llm import LLMUnavailableError
from src.rag import LLMPolicyAnswer
from src.tools import ShelterTools
from tests.test_agent import scripted_llm, text_message, tool_call_message


def fake_policy_llm(system, user, output_type):
    return LLMPolicyAnswer(answerable=True, answer="Cats cost $150.", cited_chunk_ids=["fees#Fee schedule"])


@pytest.fixture(scope="module")
def tools():
    return ShelterTools(generate=fake_policy_llm)


def make_client(tools, agent_factory=None):
    app = create_app(tools=tools, agent_factory=agent_factory or ShelterAgent)
    return TestClient(app)


def test_health(tools):
    with make_client(tools) as client:
        assert client.get("/health").json() == {"status": "ok"}


def test_get_animal_and_missing_animal(tools):
    with make_client(tools) as client:
        assert client.get("/animals/RP-0009").json()["name"] == "Taro"
        assert client.get("/animals/RP-9999").status_code == 404


def test_search_by_name(tools):
    with make_client(tools) as client:
        ids = {a["animal_id"] for a in client.get("/animals", params={"name": "Mochi"}).json()["animals"]}
        assert ids == {"RP-0001", "RP-0006"}


def test_match_validates_input(tools):
    with make_client(tools) as client:
        assert client.post("/match", json={"request": "calm dog", "species": "dragon"}).status_code == 422
        response = client.post("/match", json={"request": "calm dog", "species": "dog", "has_children": True})
        assert response.status_code == 200
        assert response.json()["matches"]


def test_policy_uses_rag_tool(tools):
    with make_client(tools) as client:
        body = client.post("/policy", json={"question": "How much is a cat?"}).json()
        assert body["answer"] == "Cats cost $150."
        assert body["sources"] == ["fees.md > Fee schedule"]


def test_chat_keeps_one_agent_per_session(tools):
    created = []

    def agent_factory(t):
        llm = scripted_llm([
            tool_call_message("call_1", "get_animal_profile", {"animal_id": "RP-0009"}),
            text_message("RP-0009 is Taro."),
            text_message("Happy to help."),
        ])
        created.append(ShelterAgent(t, llm=llm))
        return created[-1]

    with make_client(tools, agent_factory) as client:
        first = client.post("/chat", json={"message": "Tell me about RP-0009"}).json()
        assert first["reply"] == "RP-0009 is Taro."
        assert first["tool_calls"] == [{"tool": "get_animal_profile", "arguments": {"animal_id": "RP-0009"}}]

        second = client.post("/chat", json={"message": "Thanks", "session_id": first["session_id"]}).json()
        assert second["reply"] == "Happy to help."
        assert second["tool_calls"] == []
        assert len(created) == 1  # same conversation, same agent


def test_chat_returns_503_when_llm_is_down(tools):
    def broken_llm(messages, tool_schemas):
        raise LLMUnavailableError("connection refused")

    with make_client(tools, lambda t: ShelterAgent(t, llm=broken_llm)) as client:
        response = client.post("/chat", json={"message": "Hi"})
        assert response.status_code == 503
        assert "unavailable" in response.json()["detail"]


def test_chat_rejects_empty_message(tools):
    with make_client(tools) as client:
        assert client.post("/chat", json={"message": ""}).status_code == 422


def test_ui_is_served(tools):
    with make_client(tools) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert "RescuePaws AI" in response.text
