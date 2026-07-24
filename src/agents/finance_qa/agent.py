"""
Finance Q&A Agent.

Pipeline: FinanceQuestion -> advice guard -> hybrid retrieval -> grounded
generation -> FinanceAnswer (with citations + disclaimer).

Guardrails (all enforced outside the LLM's discretion):
1. Advice guard: personalized "should I buy/sell X" questions are declined and
   redirected to education, never answered as a recommendation.
2. Grounded-only: if retrieval finds nothing relevant, the agent says so instead
   of inventing an answer.
3. Always cite + disclaim: every substantive answer carries its sources and the
   educational disclaimer, appended programmatically.
"""

import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseFinanceAgent, EDUCATIONAL_DISCLAIMER
from src.agents.finance_qa.model import (
    Citation,
    FinanceAnswer,
    FinanceQuestion,
    KnowledgeLevel,
)
from src.rag.retriever import HybridRetriever, ScoredChunk

logger = logging.getLogger(__name__)

# Personalized-advice patterns. Deliberately about *actions on specific
# holdings* — general "what is / how does" questions pass through to education.
_ADVICE_PATTERNS = [
    r"\bshould i (buy|sell|invest|put|move|hold)\b",
    r"\b(is|are)\s+\w+\s+a (good|bad|smart|safe) (buy|investment|stock|bet)\b",
    r"\bwhat should i (buy|invest|do with)\b",
    r"\bwhich (stock|fund|etf|share)s? should i\b",
    r"\b(recommend|suggest)\b(\s+\w+){0,4}\s+(stock|fund|etf|investment|bond|share|securit(y|ies))s?\b",
    r"\bhow much (of my|should i)\b.*\b(invest|put)\b",
]

_ADVICE_REDIRECT = (
    "I can't give personalized investment recommendations — that's financial "
    "advice, and I'm here for education. But I'm happy to explain the concepts "
    "that would help you make that decision yourself. For example, I can walk "
    "through how the investment type works, what diversification means, or how "
    "to think about risk and return. What would you like to understand?"
)

_UNGROUNDED_REPLY = (
    "I don't have information about that in my knowledge base yet, so I'd rather "
    "not guess. I can help with investing basics like stocks, bonds, ETFs and "
    "mutual funds, account types, diversification, compound interest, and the "
    "risk/return tradeoff. Try rephrasing, or ask me about one of those topics."
)

_SYSTEM_PROMPT = """You are a financial education assistant for beginners. \
Answer the user's question using ONLY the numbered sources provided. \

Rules:
- Use only the information in the sources. If they don't fully cover the \
question, answer what they support and say what's missing — never invent facts.
- Define any financial term you use in plain language.
- Never recommend buying or selling a specific security. Explain concepts and \
general principles only.
- Be warm, clear, and concise (under 250 words).
- Do not write your own disclaimer or list sources; those are added \
automatically."""


class FinanceQAAgent(BaseFinanceAgent):
    name = "finance_qa"
    description = (
        "Answers general financial-education questions (investment types, "
        "accounts, core concepts, risk/return) from a curated, cited knowledge "
        "base. Does not give personalized investment advice."
    )

    def __init__(self, llm: BaseChatModel, retriever: HybridRetriever):
        super().__init__(llm)
        self.retriever = retriever

    def run(self, question: FinanceQuestion) -> FinanceAnswer:
        # --- Guard 1: personalized advice ---
        if self._is_advice_seeking(question.question):
            return FinanceAnswer(
                answer=_ADVICE_REDIRECT,
                citations=[],
                category=question.category,
                is_grounded=False,
                refused=True,
                refusal_reason="Personalized investment advice is out of scope.",
                disclaimer=EDUCATIONAL_DISCLAIMER,
            )

        # --- Retrieve ---
        retrieval = self.retriever.retrieve(
            question.question, top_k=question.top_k, category=question.category
        )

        # --- Guard 2: nothing relevant -> don't hallucinate ---
        if not retrieval.is_grounded:
            return FinanceAnswer(
                answer=_UNGROUNDED_REPLY,
                citations=[],
                category=question.category,
                is_grounded=False,
                refused=False,
                disclaimer=EDUCATIONAL_DISCLAIMER,
            )

        answer_text = self._generate(question, retrieval.results)
        citations = self._build_citations(retrieval.results)
        return FinanceAnswer(
            answer=answer_text,
            citations=citations,
            category=question.category,
            is_grounded=True,
            refused=False,
            disclaimer=EDUCATIONAL_DISCLAIMER,
        )

    @staticmethod
    def _is_advice_seeking(question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in _ADVICE_PATTERNS)

    def _generate(self, question: FinanceQuestion, results: list[ScoredChunk]) -> str:
        """Generate a grounded answer, with a template fallback if the LLM fails."""
        sources_block = "\n\n".join(
            f"[Source {i}] {sc.chunk.title} ({sc.chunk.source})\n{sc.chunk.text}"
            for i, sc in enumerate(results, start=1)
        )
        level_note = (
            "The reader is a beginner; explain gently and define all terms."
            if question.knowledge_level == KnowledgeLevel.BEGINNER
            else "The reader has some background; you can be more concise."
        )
        user_content = (
            f"{level_note}\n\nQuestion: {question.question}\n\n"
            f"Sources:\n{sources_block}"
        )
        try:
            response = self.llm.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT),
                 HumanMessage(content=user_content)]
            )
            return response.content
        except Exception:
            logger.warning("LLM generation failed; using template", exc_info=True)
            return self._template_answer(results)

    @staticmethod
    def _template_answer(results: list[ScoredChunk]) -> str:
        """Deterministic fallback: surface the most relevant source text so the
        user still gets grounded information when the LLM is unavailable."""
        top = results[0].chunk
        return (
            "Here's the most relevant information from my knowledge base "
            f"(AI explanation is temporarily unavailable):\n\n{top.text}"
        )

    @staticmethod
    def _build_citations(results: list[ScoredChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for sc in results:
            snippet = sc.chunk.text
            if len(snippet) > 300:
                snippet = snippet[:297].rstrip() + "..."
            citations.append(
                Citation(
                    source=sc.chunk.source,
                    title=sc.chunk.title,
                    url=sc.chunk.source_url,
                    snippet=snippet,
                    relevance_score=round(sc.cosine, 4),
                )
            )
        return citations
