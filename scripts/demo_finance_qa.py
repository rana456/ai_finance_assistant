"""Manual demo of the Finance Q&A Agent with the real index and a real LLM.

Requires OPENAI_API_KEY and a prebuilt index (run scripts/build_index.py first).

    .venv/bin/python scripts/demo_finance_qa.py "what is compound interest?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.agents.finance_qa.agent import FinanceQAAgent
from src.agents.finance_qa.model import FinanceQuestion
from src.core.llm import get_llm
from src.rag.embedder import OpenAIEmbedder
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore

INDEX_DIR = Path(__file__).resolve().parent.parent / "src" / "data" / "index"


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]) or "What is compound interest?"

    store = VectorStore.load(INDEX_DIR)
    retriever = HybridRetriever(store, OpenAIEmbedder())
    agent = FinanceQAAgent(llm=get_llm(), retriever=retriever)

    result = agent.run(FinanceQuestion(question=question))

    print(f"\nQ: {question}\n{'=' * 60}")
    print(result.answer)
    if result.citations:
        print(f"\n--- Sources ---")
        for c in result.citations:
            print(f"  • {c.title} ({c.source})  [relevance {c.relevance_score:.2f}]")
    print(f"\n{result.disclaimer}")


if __name__ == "__main__":
    main()
