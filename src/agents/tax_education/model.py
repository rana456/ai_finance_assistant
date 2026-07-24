"""
Pydantic models for the Tax Education Agent.

Mirrors the Finance Q&A shape (it's the same RAG pattern) with tax-specific
additions: a `consult_professional` flag for the advice-guard path, and a
disclaimer that combines the "not tax advice" line with a "tax figures change
yearly — verify on IRS.gov" caution.
"""

from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from src.agents.finance_qa.model import Citation, KnowledgeLevel

# The standing disclaimer for every tax answer. Two pillars: education-not-advice,
# and the reminder that specific figures change and should be verified.
TAX_DISCLAIMER = (
    "This is general tax education, not tax advice. Tax rules and figures "
    "(limits, brackets, phase-outs) change and depend on your situation — verify "
    "current numbers at IRS.gov and consult a qualified tax professional before "
    "acting."
)


class TaxQuestion(BaseModel):
    """A user question routed to the Tax Education agent."""

    question: str = Field(..., min_length=1, description="Natural-language tax question.")
    knowledge_level: KnowledgeLevel = KnowledgeLevel.BEGINNER
    top_k: int = Field(default=4, ge=1, le=20, description="Chunks to retrieve.")

    @field_validator("question")
    @classmethod
    def non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty.")
        return v


class TaxAnswer(BaseModel):
    """The agent's response contract."""

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    is_grounded: bool = Field(
        ..., description="True if retrieval cleared the relevance threshold."
    )
    refused: bool = Field(
        default=False,
        description="True when the question sought personalized tax advice.",
    )
    refusal_reason: Optional[str] = None
    consult_professional: bool = Field(
        default=False,
        description="True when the answer explicitly points the user to a tax pro.",
    )
    disclaimer: str = TAX_DISCLAIMER

    @model_validator(mode="after")
    def refusal_needs_reason(self) -> "TaxAnswer":
        if self.refused and not self.refusal_reason:
            raise ValueError("A refused answer must include a refusal_reason.")
        return self
