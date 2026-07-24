"""
Chunker: KnowledgeDocument -> list[DocumentChunk].

Our articles are short and heading-structured, so we chunk on markdown
headings. Each chunk is one section, prefixed with the article title and
section heading for context (so a retrieved chunk is self-describing even
out of its document). Oversized sections are split on paragraph boundaries
to keep chunks within a rough token budget.
"""

import re

from src.agents.finance_qa.model import DocumentChunk, KnowledgeDocument

# Rough character budget per chunk (~ a few hundred tokens). Section-based
# chunking means most sections land well under this; the split is a safety net.
MAX_CHUNK_CHARS = 1200

_HEADING_RE = re.compile(r"^#{1,6}\s+(.*)$")


def _split_sections(markdown: str) -> list[tuple[str, str]]:
    """Split markdown into (heading, body) sections by heading lines.
    Content before the first heading (if any) is returned under ''."""
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush():
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_heading, body))

    for line in markdown.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            flush()
            current_heading = m.group(1).strip()
            current_lines = []
        else:
            current_lines.append(line)
    flush()
    return sections


def _split_oversized(text: str, limit: int) -> list[str]:
    """Split an over-limit section on blank lines (paragraphs), greedily
    packing paragraphs up to the limit."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    buf = ""
    for p in paragraphs:
        candidate = f"{buf}\n\n{p}".strip() if buf else p
        if len(candidate) <= limit or not buf:
            buf = candidate
        else:
            parts.append(buf)
            buf = p
    if buf:
        parts.append(buf)
    return parts


def chunk_document(
    doc: KnowledgeDocument, max_chars: int = MAX_CHUNK_CHARS
) -> list[DocumentChunk]:
    """Chunk one document into retrievable passages.

    The article title (H1) is treated as document-level context rather than a
    section, so it isn't emitted as its own chunk.
    """
    chunks: list[DocumentChunk] = []
    index = 0
    for heading, body in _split_sections(doc.content):
        # Skip the H1 title section (it duplicates doc.title and has no body
        # of its own worth retrieving).
        if heading and heading.strip().lower() == doc.title.strip().lower():
            continue
        pieces = _split_oversized(body, max_chars) if len(body) > max_chars else [body]
        for piece in pieces:
            # Prefix with title + section so the chunk stands on its own.
            prefix = f"{doc.title}"
            if heading:
                prefix += f" — {heading}"
            text = f"{prefix}\n{piece}".strip()
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc.doc_id}::{index:04d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    category=doc.category,
                    source=doc.source,
                    source_url=doc.source_url,
                    text=text,
                )
            )
            index += 1
    return chunks


def chunk_documents(docs: list[KnowledgeDocument]) -> list[DocumentChunk]:
    """Chunk a whole corpus."""
    chunks: list[DocumentChunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks
