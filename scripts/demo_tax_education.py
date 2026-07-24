"""Manual demo of the Tax Education Agent with the real index + a real LLM.

Requires OPENAI_API_KEY and a prebuilt index (scripts/build_index.py).

    .venv/bin/python scripts/demo_tax_education.py "How does a Roth IRA work?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.agents.tax_education.agent import TaxEducationAgent
from src.agents.tax_education.model import TaxQuestion
from src.core.llm import get_llm
from src.rag.embedder import OpenAIEmbedder
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore

INDEX_DIR = Path(__file__).resolve().parent.parent / "src" / "data" / "index"


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]) or "What's the difference between a Roth and traditional IRA?"

    retriever = HybridRetriever(VectorStore.load(INDEX_DIR), OpenAIEmbedder())
    agent = TaxEducationAgent(llm=get_llm(), retriever=retriever)
    result = agent.run(TaxQuestion(question=question))

    print(f"\nQ: {question}\n{'=' * 62}")
    print(result.answer)
    if result.refused:
        print("\n[refused → referred to a tax professional]")
    if result.citations:
        print("\n--- Sources ---")
        for c in result.citations:
            print(f"  • {c.title} ({c.source})")
    print(f"\n{result.disclaimer}")


if __name__ == "__main__":
    main()
