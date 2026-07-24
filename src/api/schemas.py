"""
Request/response schemas for the API that aren't already covered by the agents'
own Pydantic models.

The structured endpoints (portfolio/market/goals) reuse the agent models
directly as their request and response bodies — these wrappers only cover the
chat and thread surface, which has no existing model.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    thread_id: Optional[str] = Field(None, description="Omit to start a new thread.")
    knowledge_level: str = "beginner"


class ChatResponse(BaseModel):
    thread_id: str
    text: str
    route: str
    data: Optional[dict] = None


class ThreadInfo(BaseModel):
    id: str
    title: str
    updated_at: str


class CreateThreadResponse(BaseModel):
    thread_id: str
    title: str


class MessageOut(BaseModel):
    """One rendered conversation turn."""
    role: str            # "user" | "assistant"
    content: str
    route: Optional[str] = None
    result_data: Optional[dict] = None


class MarketRequest(BaseModel):
    ticker: str = Field(..., min_length=1)
    analysis_type: str = "snapshot"   # snapshot | trend | metric
    period: str = "1y"                # 1d | 5d | 1mo | 6mo | 1y | 5y
