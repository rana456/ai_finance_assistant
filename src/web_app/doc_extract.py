"""
Extract text from an uploaded document so the portfolio flow can read holdings.

Handles the common cases (CSV, TXT, PDF). The extracted text is handed to the
existing LLM portfolio extractor via the chat flow — which echoes the parsed
holdings back for confirmation before any analysis, exactly like typed input.

Privacy note: we only pass text through to holdings extraction; we don't log or
persist raw document contents (statements contain PII).
"""

from __future__ import annotations

import io

# Cap how much document text we forward to the LLM (keeps token cost bounded).
MAX_CHARS = 6000


def extract_text(filename: str, data: bytes) -> str:
    """Best-effort text extraction from an uploaded file's bytes."""
    name = filename.lower()
    try:
        if name.endswith(".pdf"):
            text = _pdf_text(data)
        elif name.endswith((".csv", ".txt", ".tsv")):
            text = data.decode("utf-8", errors="replace")
        else:
            # Fall back to a best-effort decode for unknown types.
            text = data.decode("utf-8", errors="replace")
    except Exception:
        return ""
    return text[:MAX_CHARS].strip()


def _pdf_text(data: bytes) -> str:
    import pdfplumber

    parts: list[str] = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return "\n".join(parts)


def build_upload_message(filename: str, text: str) -> str:
    """Wrap extracted document text as a chat message that routes to the
    portfolio flow (which will extract and confirm the holdings)."""
    return (
        "Please analyze the portfolio holdings in this uploaded document "
        f"'{filename}':\n\n{text}"
    )
