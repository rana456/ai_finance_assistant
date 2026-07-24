"""
Pydantic models for the Finance Q&A Agent and its RAG knowledge base.

Two groups of models live here:
- Query/answer models: the agent's runtime input and output contract.
- Corpus models: the typed representation of knowledge-base documents and
  the chunks we embed and retrieve.

Design principles (consistent with the Portfolio agent):
- Validation at the model boundary, not scattered through agent code.
- Provenance is a first-class field: every chunk carries its source so the
  answer can always be cited (a grading requirement, and a trust requirement).
"""

from datetime import date
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class FinanceCategory(str, Enum):
    """Coarse topic buckets, aligned to the Finance Q&A agent's scope.
    Used both for corpus tagging and as an optional retrieval filter."""
    INVESTMENT_VEHICLES = "investment_vehicles"   # stocks, bonds, ETFs, funds
    CORE_CONCEPTS = "core_concepts"               # compounding, DCA, diversification
    ACCOUNT_TYPES = "account_types"               # brokerage, retirement accounts
    RISK_AND_RETURN = "risk_and_return"           # the risk/return tradeoff
    TAXES = "taxes"                               # tax concepts, account tax treatment
    GENERAL = "general"                           # catch-all / cross-cutting


class KnowledgeLevel(str, Enum):
    """Tunes how much the answer explains. Beginners get more jargon defined."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"


class LicenseType(str, Enum):
    """How we're allowed to use a document's source material.
    PUBLIC_DOMAIN: US government works — free to use.
    CC_BY_NC_SA:   Creative Commons (e.g. Bogleheads) — attribution required.
    ORIGINAL:      authored by us for this project.
    """
    PUBLIC_DOMAIN = "public_domain"
    CC_BY_NC_SA = "cc_by_nc_sa"
    ORIGINAL = "original"


# ---------------------------------------------------------------------------
# Runtime input / output
# ---------------------------------------------------------------------------


class ChatTurn(BaseModel):
    """One prior message, for follow-up questions that need conversation context."""
    role: str = Field(..., description="'user' or 'assistant'.")
    content: str

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'.")
        return v


class FinanceQuestion(BaseModel):
    """A user question routed to the Finance Q&A agent."""

    question: str = Field(..., min_length=1, description="Natural-language question.")
    category: Optional[FinanceCategory] = Field(
        None, description="Optional filter hint; None searches all categories."
    )
    knowledge_level: KnowledgeLevel = Field(
        default=KnowledgeLevel.BEGINNER,
        description="Controls explanation depth and how much jargon is defined.",
    )
    conversation_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Prior turns, for resolving follow-up questions.",
    )
    top_k: int = Field(
        default=4, ge=1, le=20,
        description="Number of chunks to retrieve before generating.",
    )

    @field_validator("question")
    @classmethod
    def non_empty_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty or whitespace.")
        return v


class Citation(BaseModel):
    """A single source backing part of an answer."""

    source: str = Field(..., description="e.g. 'SEC investor.gov', 'Bogleheads wiki'.")
    title: str = Field(..., description="Article title the chunk came from.")
    url: Optional[str] = Field(None, description="'Learn more' link when available.")
    snippet: str = Field(..., description="The retrieved chunk text.")
    relevance_score: float = Field(
        ..., ge=0, le=1, description="Fused retrieval score, normalized 0..1."
    )


class FinanceAnswer(BaseModel):
    """The agent's response contract."""

    answer: str = Field(..., description="Grounded, jargon-defined explanation.")
    citations: list[Citation] = Field(
        default_factory=list, description="Every source used to answer."
    )
    category: Optional[FinanceCategory] = None
    is_grounded: bool = Field(
        ..., description="True if retrieval cleared the relevance threshold."
    )
    refused: bool = Field(
        default=False,
        description="True if the agent declined (out of scope or advice-seeking).",
    )
    refusal_reason: Optional[str] = None
    disclaimer: str = Field(..., description="Educational disclaimer, always present.")

    @model_validator(mode="after")
    def refusal_needs_reason(self) -> "FinanceAnswer":
        if self.refused and not self.refusal_reason:
            raise ValueError("A refused answer must include a refusal_reason.")
        return self


# ---------------------------------------------------------------------------
# Corpus (knowledge base) models
# ---------------------------------------------------------------------------


class KnowledgeDocument(BaseModel):
    """One curated article, loaded from a markdown file + its manifest entry."""

    doc_id: str = Field(..., description="Stable unique id, e.g. 'core-concepts-diversification'.")
    title: str
    category: FinanceCategory
    source: str = Field(..., description="Human-readable source name for citations.")
    source_url: Optional[str] = None
    license: LicenseType
    attribution: Optional[str] = Field(
        None, description="Required for CC-licensed content; how we credit the source."
    )
    content: str = Field(..., description="Article body in markdown.")
    last_reviewed: date

    @model_validator(mode="after")
    def cc_requires_attribution(self) -> "KnowledgeDocument":
        """Enforce the license terms at the data boundary: CC content cannot
        enter the corpus without an attribution string."""
        if self.license == LicenseType.CC_BY_NC_SA and not self.attribution:
            raise ValueError(
                f"Document '{self.doc_id}' is CC-licensed and must carry an "
                "attribution string."
            )
        return self


class DocumentChunk(BaseModel):
    """A retrievable passage. Carries enough provenance to build a Citation
    without re-reading the source document."""

    chunk_id: str = Field(..., description="e.g. '<doc_id>::0002'.")
    doc_id: str
    title: str
    category: FinanceCategory
    source: str
    source_url: Optional[str] = None
    text: str = Field(..., min_length=1)
