"""
Hybrid retriever: dense (FAISS cosine) + sparse (BM25) fused with Reciprocal
Rank Fusion.

Why hybrid: financial questions mix precise jargon ("expense ratio", "401(k)")
with plain-language paraphrases ("the yearly fee funds charge"). Dense search
handles paraphrase; BM25 handles exact terms. RRF merges their rankings without
score-scale tuning — a chunk ranked highly by either method rises.

Grounding: the agent must be able to say "I don't know" rather than hallucinate.
We expose that via `is_grounded`, driven by the best chunk's cosine similarity
(a semantic floor), not by RRF agreement (which always returns *something*).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.agents.finance_qa.model import DocumentChunk, FinanceCategory
from src.rag.embedder import Embedder
from src.rag.vector_store import VectorStore

# RRF constant. 60 is the standard default; it dampens the influence of exact
# rank so that "top few in either list" matters more than "#1 vs #2".
RRF_K = 60

# Minimum top-chunk cosine similarity to consider the corpus relevant.
# Tuned for OpenAI text-embedding-3-small; below this we treat as ungrounded.
DEFAULT_GROUNDING_THRESHOLD = 0.28

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for BM25."""
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ScoredChunk:
    """A retrieved chunk with its relevance signals."""

    chunk: DocumentChunk
    cosine: float   # semantic similarity to the query, 0..1 (grounding signal)
    rrf: float      # fused rank score (ordering signal)


@dataclass
class RetrievalResult:
    results: list[ScoredChunk]
    is_grounded: bool


class HybridRetriever:
    """Dense + sparse retrieval with RRF fusion over a fixed chunk corpus."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: Embedder,
        grounding_threshold: float = DEFAULT_GROUNDING_THRESHOLD,
    ):
        self.store = vector_store
        self.embedder = embedder
        self.grounding_threshold = grounding_threshold
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in vector_store.chunks])

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        category: FinanceCategory | None = None,
    ) -> RetrievalResult:
        """Retrieve the top_k most relevant chunks for a query.

        `category`, when given, restricts results to that topic bucket.
        """
        chunks = self.store.chunks
        # Pull a wider candidate pool than top_k so fusion has room to work.
        pool = max(top_k * 3, 10)

        # --- Dense ---
        query_embedding = self.embedder.embed_query(query)
        dense = self.store.search(query_embedding, pool)  # [(idx, cosine)]
        dense_rank = {idx: rank for rank, (idx, _) in enumerate(dense)}

        # --- Sparse (BM25) ---
        scores = self._bm25.get_scores(_tokenize(query))
        sparse_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Only keep positive-score matches, capped at the pool size.
        sparse_idxs = [i for i in sparse_idxs if scores[i] > 0][:pool]
        sparse_rank = {idx: rank for rank, idx in enumerate(sparse_idxs)}

        # --- Reciprocal Rank Fusion ---
        candidates = set(dense_rank) | set(sparse_rank)
        fused: list[ScoredChunk] = []
        for idx in candidates:
            rrf = 0.0
            if idx in dense_rank:
                rrf += 1.0 / (RRF_K + dense_rank[idx])
            if idx in sparse_rank:
                rrf += 1.0 / (RRF_K + sparse_rank[idx])
            chunk = chunks[idx]
            if category is not None and chunk.category != category:
                continue
            cosine = self.store.cosine_for(idx, query_embedding)
            fused.append(ScoredChunk(chunk=chunk, cosine=max(0.0, cosine), rrf=rrf))

        fused.sort(key=lambda sc: sc.rrf, reverse=True)
        top = fused[:top_k]

        is_grounded = bool(top) and max(sc.cosine for sc in top) >= self.grounding_threshold
        return RetrievalResult(results=top, is_grounded=is_grounded)
