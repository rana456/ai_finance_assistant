"""
Embedder: turns text into vectors.

`Embedder` is a small Protocol so the retriever depends on an interface, not
on OpenAI. Production uses OpenAIEmbedder; tests inject a deterministic fake
(see tests/conftest for the hashing fake) and never touch the network.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# text-embedding-3-small: 1536 dims, cheap (~$0.02 / 1M tokens).
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


@runtime_checkable
class Embedder(Protocol):
    """Minimal embedding interface used across the RAG pipeline."""

    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents/chunks."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        ...


class OpenAIEmbedder:
    """Embedder backed by OpenAI's embeddings API.

    Kept import-light: the OpenAI client is created lazily so that importing
    this module (and running offline tests) never requires the SDK or a key.
    """

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, dimension: int = 1536):
        self.model = model
        self.dimension = dimension
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import OpenAI  # lazy import
            self._client = OpenAI()
        return self._client

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._get_client().embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
