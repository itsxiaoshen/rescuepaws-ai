"""The only module that talks to the LLM provider. To switch providers, change this file."""
import os
from functools import lru_cache
from typing import TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()  # reads OPENAI_API_KEY (and optional OPENAI_MODEL) from .env

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """Create the client once, on first use."""
    return OpenAI()


def generate_structured(system: str, user: str, output_type: type[T], model: str = DEFAULT_MODEL) -> T:
    """Ask the LLM for a response that matches a Pydantic model."""
    response = get_client().chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format=output_type,
    )
    message = response.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"LLM did not return structured output (refusal: {message.refusal})")
    return message.parsed


def chat_with_tools(messages: list, tools: list, model: str = DEFAULT_MODEL):
    """One LLM step in an agent loop. Returns the assistant message (text and/or tool calls)."""
    response = get_client().chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )
    return response.choices[0].message
