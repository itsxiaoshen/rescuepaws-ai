"""The only module that talks to the LLM provider. To switch providers, change this file."""
import base64
import mimetypes
import os
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError
from pydantic import BaseModel

load_dotenv()  # reads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(Exception):
    """The LLM provider could not be reached or returned an error (network, auth, rate limit...)."""


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create the client once, on first use."""
    return OpenAI()


def image_to_data_url(image_path: Path) -> str:
    """Encode a local image so it can be sent inside the API request."""
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def generate_structured(
    system: str,
    user: str,
    output_type: type[T],
    model: str = DEFAULT_MODEL,
    image_path: Path | None = None,
) -> T:
    """Ask the LLM for a response that matches a Pydantic model. Optionally include one image."""
    content = user
    if image_path is not None:
        content = [
            {"type": "text", "text": user},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
        ]
    try:
        response = get_client().chat.completions.parse(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            response_format=output_type,
        )
    except OpenAIError as e:
        raise LLMUnavailableError(str(e)) from e
    message = response.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"LLM did not return structured output (refusal: {message.refusal})")
    return message.parsed


def chat_with_tools(messages: list, tools: list, model: str = DEFAULT_MODEL):
    """One LLM step in an agent loop. Returns the assistant message (text and/or tool calls)."""
    try:
        response = get_client().chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
    except OpenAIError as e:
        raise LLMUnavailableError(str(e)) from e
    return response.choices[0].message
