"""RescuePaws HTTP API and demo UI. Run: uvicorn app.api:app --reload"""
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.rate_limit import RateLimiter
from src.agent import ShelterAgent
from src.llm import LLMUnavailableError
from src.tools import ShelterTools

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("rescuepaws.api")

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_SESSIONS = 500  # oldest conversations are dropped beyond this, so memory can't grow forever

SpeciesName = Literal["dog", "cat", "other"]
SizeName = Literal["small", "medium", "large"]


# --- Request and response bodies (FastAPI validates them with Pydantic) ---

class MatchRequest(BaseModel):
    request: str = Field(min_length=1, max_length=1000)
    species: SpeciesName | None = None
    size: SizeName | None = None
    has_children: bool = False
    has_dogs: bool = False
    has_cats: bool = False


class PolicyRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None  # omit to start a new conversation


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    tool_calls: list[dict]


def call_llm(action):
    """Run something that calls the LLM; turn provider failures into a clear 503."""
    try:
        return action()
    except LLMUnavailableError:
        logger.exception("LLM call failed")
        raise HTTPException(status_code=503, detail="The language model is unavailable. Please try again.")


def default_rate_limiter() -> RateLimiter:
    """Limits for endpoints that call the LLM. Override with environment variables."""
    return RateLimiter(
        per_minute=int(os.getenv("RATE_LIMIT_PER_MINUTE", "5")),
        per_day=int(os.getenv("RATE_LIMIT_PER_DAY", "50")),
        global_per_day=int(os.getenv("RATE_LIMIT_GLOBAL_PER_DAY", "300")),
    )


def client_id(request: Request) -> str:
    """The caller's IP. Behind a proxy (e.g. Hugging Face Spaces) it's in X-Forwarded-For."""
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def create_app(
    tools: ShelterTools | None = None,
    agent_factory=ShelterAgent,
    rate_limiter: RateLimiter | None = None,
) -> FastAPI:
    """Build the app. Tests pass in fake tools, agents, or limits; normally all are real."""
    # Conversations are kept in memory: fine for a demo, lost on restart, one process only
    sessions: dict[str, ShelterAgent] = {}
    limiter = rate_limiter or default_rate_limiter()

    def enforce_rate_limit(request: Request) -> None:
        message = limiter.check(client_id(request))
        if message:
            logger.warning("rate limited client=%s: %s", client_id(request), message)
            raise HTTPException(status_code=429, detail=message)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.tools = tools or ShelterTools()  # loads data and models once, at startup
        logger.info("Shelter tools loaded")
        yield

    app = FastAPI(title="RescuePaws AI", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/animals/{animal_id}")
    def get_animal(animal_id: str):
        result = app.state.tools.get_animal_profile(animal_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result

    @app.get("/animals")
    def search_animals(
        name: str | None = None,
        species: SpeciesName | None = None,
        size: SizeName | None = None,
        include_unavailable: bool = False,
    ):
        return app.state.tools.search_animals(name, species, size, include_unavailable)

    @app.post("/match")
    def match_animals(body: MatchRequest):
        return app.state.tools.match_animals(**body.model_dump())

    @app.post("/policy", dependencies=[Depends(enforce_rate_limit)])
    def answer_policy(body: PolicyRequest):
        return call_llm(lambda: app.state.tools.answer_policy_question(body.question))

    @app.post("/chat", response_model=ChatResponse, dependencies=[Depends(enforce_rate_limit)])
    def chat(body: ChatRequest):
        session_id = body.session_id or uuid.uuid4().hex
        agent = sessions.get(session_id)
        if agent is None:  # new conversation (or the server restarted)
            if len(sessions) >= MAX_SESSIONS:
                sessions.pop(next(iter(sessions)))  # drop the oldest conversation
            agent = sessions[session_id] = agent_factory(app.state.tools)

        log_start = len(agent.tool_log)
        reply = call_llm(lambda: agent.chat(body.message))
        tool_calls = [{"tool": c["tool"], "arguments": c["arguments"]} for c in agent.tool_log[log_start:]]
        logger.info("chat session=%s tools=%s", session_id[:8], [c["tool"] for c in tool_calls])
        return ChatResponse(session_id=session_id, reply=reply, tool_calls=tool_calls)

    @app.get("/")
    def ui():
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
