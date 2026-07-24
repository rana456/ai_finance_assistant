"""
Tax Education Agent.

A domain-focused RAG agent: it reuses the Finance Q&A retrieval stack over a
tax-focused slice of the knowledge base, and adds two tax-specific guardrails:

1. Tax-advice guard: personalized tax questions ("how much will I owe?",
   "should I do Roth or traditional?", "is X deductible for me?") are refused
   and redirected to a tax professional — the education/advice line is a real
   legal boundary here.
2. Number caution: the standing disclaimer reminds users that specific figures
   change yearly and must be verified at IRS.gov. The corpus itself is
   number-light by design, so the agent never asserts a stale limit or bracket.

Everything else — grounded-only answering, citations, template fallback — is
the same safety model as the Finance Q&A agent.
"""

import logging
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.base import BaseFinanceAgent
from src.agents.finance_qa.model import Citation
from src.agents.tax_education.model import TAX_DISCLAIMER, TaxAnswer, TaxQuestion
from src.rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Personalized-advice patterns: questions about the user's *own* tax outcome,
# reporting, or a definitive should-I decision. General "how does X work"
# questions pass through to education.
_ADVICE_PATTERNS = [
    r"\bhow much (tax|will i (owe|pay)|do i owe)\b",
    r"\bwhat('?s| is) my (tax |marginal |effective )+(rate|bracket|bill|liability)\b",
    r"\bwhat (tax )?bracket am i in\b",
    r"\bshould i (do|choose|pick|contribute to|convert|use) (a |the )?(roth|traditional|ira|401|hsa|529)\b",
    r"\bis\s+.*\bdeductible for me\b",
    r"\bhow (do|should) i (report|file|claim|deduct)\b",
    r"\bcan i (deduct|claim|write off)\b.*\b(my|our)\b",
    r"\bwhat (should|can) i (deduct|claim|do) (on|for) my (taxes|return)\b",
]

_ADVICE_REDIRECT = (
    "That's a question about your specific tax situation, which is personalized "
    "tax advice — I can't give that, and getting it wrong has real consequences. "
    "A qualified tax professional (a CPA or enrolled agent) can answer it for "
    "your circumstances. What I can do is explain the general concepts so you're "
    "ready for that conversation — for example, how Roth and traditional accounts "
    "differ, or how capital gains are taxed. Want me to walk through the concept?"
)

_UNGROUNDED_REPLY = (
    "I don't have that tax topic in my knowledge base yet, so I'd rather not "
    "guess — tax details matter too much to wing. I can explain account types "
    "(401(k), IRA, Roth, HSA, 529, brokerage), how tax-deferred vs. tax-free vs. "
    "taxable accounts work, and how capital gains and dividends are taxed. Try "
    "one of those, or rephrase your question."
)

_SYSTEM_PROMPT = """You are a financial education assistant explaining tax \
concepts to a beginner, using ONLY the numbered sources provided.

Rules:
- Use only the information in the sources. If they don't fully cover the \
question, answer what they support and say what's missing — never invent facts.
- This is general education, NOT tax advice. Never tell the user what THEY \
specifically should do, owe, report, or deduct.
- Do NOT state specific current dollar figures, contribution limits, or tax \
brackets as fact — those change yearly. If a number matters, say it varies by \
year and point the user to IRS.gov to verify.
- Define tax terms in plain language. Be warm, clear, and concise (under 250 words).
- Do not write your own disclaimer or list sources; those are added automatically."""


class TaxEducationAgent(BaseFinanceAgent):
    name = "tax_education"
    description = (
        "Explains tax concepts and account types (401(k), IRA, Roth, HSA, 529, "
        "brokerage; capital gains, dividends) from a curated, cited knowledge "
        "base. General education only — not personalized tax advice."
    )

    def __init__(self, llm: BaseChatModel, retriever: HybridRetriever):
        super().__init__(llm)
        self.retriever = retriever

    def run(self, question: TaxQuestion) -> TaxAnswer:
        # --- Guard: personalized tax advice ---
        if self._is_advice_seeking(question.question):
            return TaxAnswer(
                answer=_ADVICE_REDIRECT,
                citations=[],
                is_grounded=False,
                refused=True,
                refusal_reason="Personalized tax advice is out of scope.",
                consult_professional=True,
                disclaimer=TAX_DISCLAIMER,
            )

        # --- Retrieve (tax-focused corpus) ---
        retrieval = self.retriever.retrieve(question.question, top_k=question.top_k)

        # --- Guard: nothing relevant -> don't guess ---
        if not retrieval.is_grounded:
            return TaxAnswer(
                answer=_UNGROUNDED_REPLY,
                citations=[],
                is_grounded=False,
                disclaimer=TAX_DISCLAIMER,
            )

        answer_text = self._generate(question, retrieval.results)
        return TaxAnswer(
            answer=answer_text,
            citations=self._build_citations(retrieval.results),
            is_grounded=True,
            disclaimer=TAX_DISCLAIMER,
        )

    @staticmethod
    def _is_advice_seeking(question: str) -> bool:
        q = question.lower()
        return any(re.search(p, q) for p in _ADVICE_PATTERNS)

    def _generate(self, question: TaxQuestion, results) -> str:
        sources_block = "\n\n".join(
            f"[Source {i}] {sc.chunk.title} ({sc.chunk.source})\n{sc.chunk.text}"
            for i, sc in enumerate(results, start=1)
        )
        user_content = f"Question: {question.question}\n\nSources:\n{sources_block}"
        try:
            response = self.llm.invoke(
                [SystemMessage(content=_SYSTEM_PROMPT),
                 HumanMessage(content=user_content)]
            )
            return response.content
        except Exception:
            logger.warning("Tax generation failed; using template", exc_info=True)
            return (
                "Here's the most relevant information from my knowledge base "
                f"(AI explanation is temporarily unavailable):\n\n{results[0].chunk.text}"
            )

    @staticmethod
    def _build_citations(results) -> list[Citation]:
        citations = []
        for sc in results:
            snippet = sc.chunk.text
            if len(snippet) > 300:
                snippet = snippet[:297].rstrip() + "..."
            citations.append(Citation(
                source=sc.chunk.source, title=sc.chunk.title, url=sc.chunk.source_url,
                snippet=snippet, relevance_score=round(sc.cosine, 4),
            ))
        return citations
