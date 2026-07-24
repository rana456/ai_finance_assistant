"""
Build and persist the Finance Q&A knowledge-base index.

Run once after adding or editing articles:
    .venv/bin/python scripts/build_index.py

Requires OPENAI_API_KEY (embeddings are computed via OpenAI). The resulting
index is loaded cheaply at query time by the retriever.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.rag.chunker import chunk_documents
from src.rag.embedder import OpenAIEmbedder
from src.rag.loader import load_documents
from src.rag.vector_store import VectorStore

INDEX_DIR = Path(__file__).resolve().parent.parent / "src" / "data" / "index"


def main() -> None:
    load_dotenv()
    docs = load_documents()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    embedder = OpenAIEmbedder()
    print(f"Embedding with {embedder.model} ...")
    embeddings = embedder.embed_documents([c.text for c in chunks])

    store = VectorStore.build(chunks, embeddings)
    store.save(INDEX_DIR)
    print(f"Index saved to {INDEX_DIR}")


if __name__ == "__main__":
    main()
