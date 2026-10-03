"""Rate limit tests. The `tools` fixture is in conftest.py."""
from fastapi.testclient import TestClient

from app.api import create_app
from app.rate_limit import RateLimiter
from src.agent import ShelterAgent
from tests.test_agent import scripted_llm, text_message


class FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


def test_per_minute_limit_resets_after_a_minute():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=2, per_day=100, global_per_day=100, clock=clock)
    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert "wait a minute" in limiter.check("a")
    assert limiter.check("b") is None  # other clients are not affected
    clock.now += 61
    assert limiter.check("a") is None


def test_per_day_limit():
    clock = FakeClock()
    limiter = RateLimiter(per_minute=100, per_day=3, global_per_day=100, clock=clock)
    for _ in range(3):
        assert limiter.check("a") is None
        clock.now += 120
    assert "today's message limit" in limiter.check("a")
    clock.now += 24 * 60 * 60
    assert limiter.check("a") is None


def test_global_limit_applies_across_clients():
    limiter = RateLimiter(per_minute=100, per_day=100, global_per_day=2, clock=FakeClock())
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert "daily limit" in limiter.check("c")


def test_chat_returns_429_when_limited(tools):
    limiter = RateLimiter(per_minute=1, per_day=100, global_per_day=100)
    agent_factory = lambda t: ShelterAgent(t, llm=scripted_llm([text_message("Hi!"), text_message("Hi again!")]))
    with TestClient(create_app(tools=tools, agent_factory=agent_factory, rate_limiter=limiter)) as client:
        assert client.post("/chat", json={"message": "Hello"}).status_code == 200
        response = client.post("/chat", json={"message": "Hello again"})
        assert response.status_code == 429
        assert "wait a minute" in response.json()["detail"]


def test_lookups_are_not_rate_limited(tools):
    limiter = RateLimiter(per_minute=1, per_day=1, global_per_day=1)
    with TestClient(create_app(tools=tools, rate_limiter=limiter)) as client:
        for _ in range(3):
            assert client.get("/animals/RP-0009").status_code == 200
