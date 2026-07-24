"""
FAISS-backed dense vector store.

Stores L2-normalized chunk embeddings in a flat inner-product index, so inner
product == cosine similarity. Flat (exact) search is the right choice for a
small curated corpus: no recall loss, no tuning, instant at this scale.

The store persists to disk as two files (the FAISS index + a JSON sidecar of
chunk metadata) so the index is built once and loaded cheaply thereafter.
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from src.agents.finance_qa.model import DocumentChunk


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows so inner product equals cosine similarity."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # avoid divide-by-zero on any zero vector
    return vectors / norms


class VectorStore:
    """A flat cosine-similarity index over document chunks."""

    def __init__(
        self,
        index: faiss.Index,
        chunks: list[DocumentChunk],
        embeddings: np.ndarray,
    ):
        self.index = index
        self.chunks = chunks
        self.embeddings = embeddings  # normalized, aligned with `chunks`

    @classmethod
    def build(
        cls, chunks: list[DocumentChunk], embeddings: list[list[float]]
    ) -> "VectorStore":
        """Build an index from chunks and their (unnormalized) embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length.")
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks.")
        mat = _normalize(np.asarray(embeddings, dtype="float32"))
        index = faiss.IndexFlatIP(mat.shape[1])
        index.add(mat)
        return cls(index=index, chunks=chunks, embeddings=mat)

    def search(
        self, query_embedding: list[float], top_n: int
    ) -> list[tuple[int, float]]:
        """Return (chunk_index, cosine_similarity) for the top_n nearest chunks."""
        q = _normalize(np.asarray([query_embedding], dtype="float32"))
        top_n = min(top_n, len(self.chunks))
        scores, idxs = self.index.search(q, top_n)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0]) if i != -1]

    def cosine_for(self, chunk_index: int, query_embedding: list[float]) -> float:
        """Cosine similarity of a specific chunk to the query (used to score
        candidates that only the sparse retriever surfaced)."""
        q = _normalize(np.asarray([query_embedding], dtype="float32"))[0]
        return float(np.dot(self.embeddings[chunk_index], q))

    # --- persistence ---

    def save(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        meta = [c.model_dump(mode="json") for c in self.chunks]
        (directory / "chunks.json").write_text(json.dumps(meta), encoding="utf-8")
        np.save(directory / "embeddings.npy", self.embeddings)

    @classmethod
    def load(cls, directory: Path) -> "VectorStore":
        directory = Path(directory)
        index = faiss.read_index(str(directory / "index.faiss"))
        meta = json.loads((directory / "chunks.json").read_text(encoding="utf-8"))
        chunks = [DocumentChunk(**m) for m in meta]
        embeddings = np.load(directory / "embeddings.npy")
        return cls(index=index, chunks=chunks, embeddings=embeddings)
