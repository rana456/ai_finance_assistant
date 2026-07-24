"""
Natural-language portfolio extraction for the chat path.

Turns a free-text message ("I own 10 Apple and 5 MSFT at $150") into a draft
list of holdings via the LLM, using a permissive schema so a single odd value
doesn't blow up — the draft is validated later against the strict PortfolioInput
model, and always echoed back for user confirmation before analysis.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = """Extract the user's stock holdings from their message. \
Respond with ONLY a JSON object of this shape:
{"holdings": [{"ticker": "AAPL", "quantity": 10, "cost_basis_per_share": 150.0}]}

Rules:
- Use the stock TICKER symbol, resolving company names yourself (Apple -> AAPL, \
Microsoft -> MSFT, Tesla -> TSLA).
- quantity is the number of shares (a number).
- cost_basis_per_share is the price paid per share if the user mentions it, \
otherwise null.
- If the message contains no holdings, return {"holdings": []}.
- Output JSON only, no prose."""

_AFFIRMATIVE = {
    "yes", "y", "yeah", "yep", "yup", "sure", "ok", "okay", "confirm", "confirmed",
    "correct", "right", "looks good", "go ahead", "analyze", "analyse", "do it",
    "that's right", "thats right", "sounds good",
}
_NEGATIVE = {
    "no", "n", "nope", "wrong", "not right", "incorrect", "change", "edit", "fix",
    "that's wrong", "thats wrong",
}


class ExtractedHolding(BaseModel):
    """Permissive draft holding (validated later via HoldingInput)."""
    ticker: str
    quantity: float
    cost_basis_per_share: Optional[float] = None


class PortfolioExtraction(BaseModel):
    holdings: list[ExtractedHolding] = Field(default_factory=list)


def extract_portfolio_holdings(llm: BaseChatModel, text: str) -> PortfolioExtraction:
    """Extract holdings from free text. Returns an empty extraction on any
    failure (parse error, bad shape) — the node treats empty as 'ask the user'."""
    try:
        resp = llm.invoke(
            [SystemMessage(content=_EXTRACT_PROMPT), HumanMessage(content=text)]
        )
        raw = re.sub(r"^```(?:json)?|```$", "", resp.content.strip(), flags=re.MULTILINE).strip()
        data = json.loads(raw)
        if not isinstance(data, dict):
            return PortfolioExtraction()
        return PortfolioExtraction(**data)
    except Exception:
        logger.warning("Portfolio extraction failed", exc_info=True)
        return PortfolioExtraction()


def is_affirmative(text: str) -> bool:
    t = text.strip().lower().rstrip("!.")
    return t in _AFFIRMATIVE or any(t.startswith(a + " ") for a in _AFFIRMATIVE)


def is_negative(text: str) -> bool:
    t = text.strip().lower().rstrip("!.")
    return t in _NEGATIVE or any(t.startswith(n + " ") for n in _NEGATIVE)
