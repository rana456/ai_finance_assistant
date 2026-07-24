"""Manual demo of the Market Analysis Agent with live data + a real LLM.

Requires OPENAI_API_KEY, a prebuilt index (scripts/build_index.py), and
optionally ALPHAVANTAGE_API_KEY for the news feature.

    .venv/bin/python scripts/demo_market_analysis.py "How has Apple done this year?"
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.agents.market_analysis.agent import MarketAnalysisAgent
from src.agents.market_analysis.model import MarketQuery
from src.core.llm import get_llm
from src.data.market_analysis_service import MarketAnalysisService
from src.data.news_client import AlphaVantageNewsClient
from src.rag.embedder import OpenAIEmbedder
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import VectorStore

INDEX_DIR = Path(__file__).resolve().parent.parent / "src" / "data" / "index"


def main() -> None:
    load_dotenv()
    question = " ".join(sys.argv[1:]) or "How has Cigna performed this year?"

    retriever = None
    if INDEX_DIR.exists():
        retriever = HybridRetriever(VectorStore.load(INDEX_DIR), OpenAIEmbedder())

    agent = MarketAnalysisAgent(
        llm=get_llm(),
        market_data=MarketAnalysisService(),
        retriever=retriever,
        news_client=AlphaVantageNewsClient(),
    )

    result = agent.run(MarketQuery(question=question))

    print(f"\nQ: {question}\n{'=' * 60}")
    print(f"[intent: {result.intent.value}]  data as of {result.as_of:%Y-%m-%d %H:%M UTC}\n")
    print(result.narrative)
    if result.snapshots:
        print("\n--- Snapshots ---")
        for s in result.snapshots:
            chg = f"{s.change_pct:+.2f}%" if s.change_pct is not None else "n/a"
            print(f"  {s.ticker}: ${s.price:,.2f} ({chg})")
    if result.trends:
        print("\n--- Trends ---")
        for t in result.trends:
            print(f"  {t.ticker} over {t.period.label}: {t.pct_change:+.1f}% [{t.trend_signal.value}]")
    if result.citations:
        print("\n--- Sources ---")
        for c in result.citations:
            print(f"  • {c.title} ({c.source})")
    print(f"\n{result.disclaimer}")


if __name__ == "__main__":
    main()
