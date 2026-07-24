"""
LLM factory: the one place model choice and API keys are resolved.

Reads OPENAI_API_KEY from the environment (a .env file is loaded if present).
Agents receive the LLM by injection and never touch env vars themselves.
"""

import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

DEFAULT_MODEL = "gpt-4o-mini"


def get_llm(model: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """Build the chat model used by all agents.

    Low default temperature: financial education should be consistent,
    not creative.
    """
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to a .env file in the project "
            "root (see .env.example) or export it in your shell."
        )
    return ChatOpenAI(model=model or DEFAULT_MODEL, temperature=temperature)
