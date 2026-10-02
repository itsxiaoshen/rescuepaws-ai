"""Agent loop tests with a scripted fake LLM: no API calls."""
import json
from types import SimpleNamespace

from src.agent import MAX_STEPS, ShelterAgent


def tool_call_message(call_id, name, arguments):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(arguments)))
    return SimpleNamespace(content=None, tool_calls=[call])


def text_message(text):
    return SimpleNamespace(content=text, tool_calls=None)


class FakeTools:
    def __init__(self):
        self.calls = []

    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return {"name": "Taro"}


def scripted_llm(responses):
    """Return the scripted responses one by one, recording what the agent sent."""
    sent = []

    def llm(messages, tools):
        sent.append(list(messages))
        return responses[len(sent) - 1]

    llm.sent = sent
    return llm


def test_answers_directly_without_tools():
    tools = FakeTools()
    agent = ShelterAgent(tools, llm=scripted_llm([text_message("Hello! How can I help?")]))
    assert agent.chat("Hi") == "Hello! How can I help?"
    assert tools.calls == []


def test_runs_tool_then_answers():
    tools = FakeTools()
    llm = scripted_llm([
        tool_call_message("call_1", "get_animal_profile", {"animal_id": "RP-0009"}),
        text_message("RP-0009 is Taro."),
    ])
    agent = ShelterAgent(tools, llm=llm)

    assert agent.chat("Tell me about RP-0009") == "RP-0009 is Taro."
    assert tools.calls == [("get_animal_profile", {"animal_id": "RP-0009"})]
    assert agent.tool_log[0]["tool"] == "get_animal_profile"

    # The tool result was sent back to the LLM, linked to its call ID
    tool_message = llm.sent[1][-1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert json.loads(tool_message["content"]) == {"name": "Taro"}


def test_stops_after_max_steps():
    tools = FakeTools()
    endless = [tool_call_message(f"call_{i}", "search_animals", {}) for i in range(MAX_STEPS + 1)]
    agent = ShelterAgent(tools, llm=scripted_llm(endless))
    reply = agent.chat("Loop forever")
    assert "couldn't complete" in reply
    assert len(tools.calls) == MAX_STEPS


def test_conversation_history_is_kept():
    tools = FakeTools()
    llm = scripted_llm([text_message("First reply"), text_message("Second reply")])
    agent = ShelterAgent(tools, llm=llm)
    agent.chat("First question")
    agent.chat("Second question")
    contents = [m["content"] if isinstance(m, dict) else m.content for m in llm.sent[1]]
    assert "First question" in contents and "First reply" in contents
