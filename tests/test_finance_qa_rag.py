"""Tests for the RAG pipeline: loader, chunker, vector store, hybrid retriever.
All offline via the FakeEmbedder — no OpenAI, no network."""

import pytest

from src.agents.finance_qa.model import (
    DocumentChunk,
    FinanceCategory,
    KnowledgeDocument,
    LicenseType,
)
from src.rag.chunker import chunk_document, chunk_documents
from src.rag.loader import load_documents, parse_article
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore

from datetime import date


# --- helpers ---

def make_doc(doc_id="doc-x", title="Test Title", content="# Test Title\n\n## A\nalpha body\n\n## B\nbeta body"):
    return KnowledgeDocument(
        doc_id=doc_id, title=title, category=FinanceCategory.CORE_CONCEPTS,
        source="Test", source_url=None, license=LicenseType.ORIGINAL,
        attribution=None, content=content, last_reviewed=date(2026, 7, 21),
    )


def build_retriever(embedder, chunks=None):
    docs = load_documents() if chunks is None else None
    chunks = chunks if chunks is not None else chunk_documents(docs)
    embeddings = embedder.embed_documents([c.text for c in chunks])
    store = VectorStore.build(chunks, embeddings)
    return HybridRetriever(store, embedder, grounding_threshold=0.05)


# --- loader ---

class TestLoader:
    def test_loads_sample_corpus(self):
        docs = load_documents()
        assert len(docs) >= 3
        assert {d.doc_id for d in docs} >= {
            "investment-vehicles-what-is-a-stock",
            "core-concepts-diversification",
            "core-concepts-compound-interest",
        }

    def test_cc_article_has_attribution(self):
        docs = {d.doc_id: d for d in load_documents()}
        div = docs["core-concepts-diversification"]
        assert div.license == LicenseType.CC_BY_NC_SA
        assert div.attribution  # enforced by the model, checked here too

    def test_missing_frontmatter_rejected(self):
        with pytest.raises(ValueError):
            parse_article("# Just a heading\n\nno frontmatter here")


# --- chunker ---

class TestChunker:
    def test_splits_on_headings(self):
        chunks = chunk_document(make_doc())
        assert len(chunks) == 2
        assert "alpha body" in chunks[0].text
        assert "beta body" in chunks[1].text

    def test_title_section_not_emitted(self):
        # The H1 that duplicates the title should not become its own chunk.
        chunks = chunk_document(make_doc(content="# Test Title\n\nintro\n\n## A\nbody"))
        assert all("body" in c.text or "intro" in c.text for c in chunks)
        assert not any(c.text.strip() == "Test Title" for c in chunks)

    def test_chunk_carries_provenance(self):
        chunks = chunk_document(make_doc())
        c = chunks[0]
        assert c.doc_id == "doc-x"
        assert c.source == "Test"
        assert c.chunk_id.startswith("doc-x::")

    def test_chunk_prefixed_with_title_and_heading(self):
        chunks = chunk_document(make_doc())
        assert chunks[0].text.startswith("Test Title — A")

    def test_oversized_section_split(self):
        big = "# T\n\n## Big\n" + "\n\n".join(["para " * 40] * 6)
        chunks = chunk_document(make_doc(title="T", content=big))
        assert len(chunks) > 1


# --- vector store ---

class TestVectorStore:
    def test_build_and_search(self, fake_embedder):
        chunks = [
            DocumentChunk(chunk_id="c1", doc_id="d", title="t", category=FinanceCategory.GENERAL, source="s", text="apple banana cherry"),
            DocumentChunk(chunk_id="c2", doc_id="d", title="t", category=FinanceCategory.GENERAL, source="s", text="dog cat fish"),
        ]
        store = VectorStore.build(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
        hits = store.search(fake_embedder.embed_query("apple banana"), top_n=2)
        assert hits[0][0] == 0  # apple/banana chunk ranks first
        assert hits[0][1] > hits[1][1]

    def test_save_and_load_roundtrip(self, fake_embedder, tmp_path):
        chunks = chunk_documents(load_documents())
        store = VectorStore.build(chunks, fake_embedder.embed_documents([c.text for c in chunks]))
        store.save(tmp_path / "idx")
        loaded = VectorStore.load(tmp_path / "idx")
        assert len(loaded.chunks) == len(chunks)
        assert loaded.chunks[0].chunk_id == chunks[0].chunk_id

    def test_build_rejects_length_mismatch(self, fake_embedder):
        chunks = [DocumentChunk(chunk_id="c1", doc_id="d", title="t", category=FinanceCategory.GENERAL, source="s", text="x")]
        with pytest.raises(ValueError):
            VectorStore.build(chunks, [])


# --- hybrid retriever ---

class TestHybridRetriever:
    def test_retrieves_relevant_chunk(self, fake_embedder):
        r = build_retriever(fake_embedder)
        result = r.retrieve("what is compound interest", top_k=3)
        assert result.is_grounded
        assert any("compound" in sc.chunk.doc_id for sc in result.results)

    def test_exact_term_found_via_sparse(self, fake_embedder):
        # "diversification" is a precise term BM25 should catch.
        r = build_retriever(fake_embedder)
        result = r.retrieve("diversification", top_k=3)
        assert any(sc.chunk.doc_id == "core-concepts-diversification" for sc in result.results)

    def test_category_filter_restricts_results(self, fake_embedder):
        r = build_retriever(fake_embedder)
        result = r.retrieve("stock", top_k=5, category=FinanceCategory.CORE_CONCEPTS)
        assert all(sc.chunk.category == FinanceCategory.CORE_CONCEPTS for sc in result.results)

    def test_ungrounded_when_no_overlap(self, fake_embedder):
        # A query sharing no tokens with the corpus -> near-zero cosine.
        r = build_retriever(fake_embedder)
        r.grounding_threshold = 0.5  # strict floor
        result = r.retrieve("zxqw plkm qwerty", top_k=3)
        assert not result.is_grounded

    def test_respects_top_k(self, fake_embedder):
        r = build_retriever(fake_embedder)
        result = r.retrieve("investing basics money", top_k=2)
        assert len(result.results) <= 2
