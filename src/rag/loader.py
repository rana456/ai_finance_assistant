"""
Knowledge-base loader: markdown files with YAML frontmatter -> KnowledgeDocument.

The frontmatter is the per-article source of truth; the manifest is a
reviewable inventory. Loading validates through the pydantic model, so a
mislicensed or malformed article fails loudly at load time rather than
silently polluting the index.
"""

from pathlib import Path

import yaml

from src.agents.finance_qa.model import KnowledgeDocument

# Default location of the curated corpus.
KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "knowledge_base"

_FRONTMATTER_DELIM = "---"


def parse_article(raw: str) -> KnowledgeDocument:
    """Parse one markdown article (frontmatter + body) into a KnowledgeDocument."""
    if not raw.lstrip().startswith(_FRONTMATTER_DELIM):
        raise ValueError("Article is missing its YAML frontmatter block.")
    # split into ['', frontmatter, body]
    _, frontmatter, body = raw.split(_FRONTMATTER_DELIM, 2)
    meta = yaml.safe_load(frontmatter) or {}
    return KnowledgeDocument(**meta, content=body.strip())


def load_documents(kb_dir: Path = KNOWLEDGE_BASE_DIR) -> list[KnowledgeDocument]:
    """Load and validate every *.md article in the knowledge base directory.

    The manifest.yaml is intentionally skipped (it's an inventory, not an
    article). Returns documents sorted by doc_id for deterministic indexing.
    """
    docs: list[KnowledgeDocument] = []
    for path in sorted(kb_dir.glob("*.md")):
        try:
            docs.append(parse_article(path.read_text(encoding="utf-8")))
        except Exception as e:
            raise ValueError(f"Failed to load article '{path.name}': {e}") from e
    if not docs:
        raise ValueError(f"No articles found in {kb_dir}.")
    return docs
