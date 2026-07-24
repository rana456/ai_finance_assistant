"""Tests for the FastAPI backend, using a stub context (no real agents/network)."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage

from src.api.main import create_app
from src.web_app.thread_store import ThreadStore


class StubGraph:
    def __init__(self):
        self.store: dict[str, list] = {}

    def get_state(self, config):
        tid = config["configurable"]["thread_id"]
        return SimpleNamespace(values={"messages": self.store.get(tid, [])})


class StubAssistant:
    def __init__(self):
        self.graph = StubGraph()

    def chat(self, message, thread_id, knowledge_level="beginner"):
        self.graph.store.setdefault(thread_id, []).extend([
            HumanMessage(content=message),
            AIMessage(content=f"reply: {message}",
                      additional_kwargs={"route": "finance_qa", "result_data": {"x": 1}}),
        ])
        return SimpleNamespace(text=f"reply: {message}", route="finance_qa", data={"x": 1})


class StubAgent:
    def __init__(self, result):
        self.result = result
        self.last = None

    def run(self, inp):
        self.last = inp
        return self.result


@dataclass
class StubCtx:
    assistant: object
    agents: dict
    thread_store: object


@pytest.fixture
def client(tmp_path):
    ctx = StubCtx(
        assistant=StubAssistant(),
        agents={
            "portfolio_analysis": StubAgent({"narrative": "port", "metrics": {}}),
            "market_analysis": StubAgent({"narrative": "mkt", "snapshots": []}),
            "goal_planning": StubAgent({"narrative": "goal"}),
        },
        thread_store=ThreadStore(tmp_path / "t.db"),
    )
    return TestClient(create_app(ctx))


class TestHealthAndThreads:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_create_and_list_threads(self, client):
        tid = client.post("/threads").json()["thread_id"]
        ids = [t["id"] for t in client.get("/threads").json()]
        assert tid in ids

    def test_delete_thread(self, client):
        tid = client.post("/threads").json()["thread_id"]
        client.delete(f"/threads/{tid}")
        assert tid not in [t["id"] for t in client.get("/threads").json()]


class TestChat:
    def test_chat_creates_thread_and_replies(self, client):
        r = client.post("/chat", json={"message": "What is an ETF?"}).json()
        assert r["text"] == "reply: What is an ETF?"
        assert r["route"] == "finance_qa"
        assert r["thread_id"]

    def test_first_message_titles_thread(self, client):
        tid = client.post("/chat", json={"message": "How is Apple doing?"}).json()["thread_id"]
        titles = {t["id"]: t["title"] for t in client.get("/threads").json()}
        assert titles[tid].startswith("How is Apple")

    def test_empty_message_rejected(self, client):
        assert client.post("/chat", json={"message": ""}).status_code == 422

    def test_get_messages(self, client):
        tid = client.post("/chat", json={"message": "hello"}).json()["thread_id"]
        msgs = client.get(f"/threads/{tid}/messages").json()
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert msgs[1]["route"] == "finance_qa"
        assert msgs[1]["result_data"] == {"x": 1}


class TestStructured:
    def test_portfolio_analyze(self, client):
        r = client.post("/portfolio/analyze", json={
            "holdings": [{"ticker": "AAPL", "quantity": 10}], "source": "manual"})
        assert r.status_code == 200
        assert r.json()["narrative"] == "port"

    def test_portfolio_empty_holdings_rejected(self, client):
        r = client.post("/portfolio/analyze", json={"holdings": [], "source": "manual"})
        assert r.status_code == 422  # PortfolioInput requires >=1 holding

    def test_market_analyze(self, client):
        r = client.post("/market/analyze", json={
            "ticker": "AAPL", "analysis_type": "snapshot", "period": "1y"})
        assert r.status_code == 200
        assert r.json()["narrative"] == "mkt"

    def test_market_invalid_type_rejected(self, client):
        r = client.post("/market/analyze", json={
            "ticker": "AAPL", "analysis_type": "bogus", "period": "1y"})
        assert r.status_code == 422

    def test_goals_plan(self, client):
        r = client.post("/goals/plan", json={
            "time_horizon_years": 30, "current_savings": 1000,
            "monthly_contribution": 100, "risk_tolerance": "moderate"})
        assert r.status_code == 200
        assert r.json()["narrative"] == "goal"
