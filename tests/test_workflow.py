"""Tests for the LangGraph workflow: router, node routing, the portfolio
extract-confirm flow, multi-turn memory, and fallbacks. Offline throughout."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.workflow.assistant import FinanceAssistant
from src.workflow.nodes import WorkflowNodes
from src.workflow.router import Router


# --- stubs ---

class StubResult:
    def __init__(self, text, data=None):
        self.answer = text
        self.narrative = text
        self._data = data or {"text": text}

    def model_dump(self, mode=None):
        return self._data


class StubAgent:
    def __init__(self, text):
        self.result = StubResult(text)
        self.last_input = None

    def run(self, inp):
        self.last_input = inp
        return self.result


class RaisingAgent:
    def run(self, inp):
        raise RuntimeError("boom")


class StubRouter:
    def __init__(self, label):
        self.label = label

    def classify(self, query):
        return self.label


def default_agents():
    return {
        "finance_qa": StubAgent("qa reply"),
        "market_analysis": StubAgent("market reply"),
        "goal_planning": StubAgent("goal reply"),
        "tax_education": StubAgent("tax reply"),
        "portfolio_analysis": StubAgent("portfolio reply"),
    }


def build_assistant(route_label, agents=None, extractor_responses=None):
    agents = agents if agents is not None else default_agents()
    extractor = FakeListChatModel(responses=extractor_responses or ['{"holdings": []}'])
    nodes = WorkflowNodes(router=StubRouter(route_label), agents=agents, extractor_llm=extractor)
    return FinanceAssistant(nodes)


class ExplodingLLM(FakeListChatModel):
    def _generate(self, *a, **k):
        raise RuntimeError("router down")


# --- Router unit tests ---

class TestRouter:
    def test_llm_label_used(self):
        r = Router(FakeListChatModel(responses=["market_analysis"]))
        assert r.classify("anything") == "market_analysis"

    def test_unrecognized_llm_falls_back_to_heuristic(self):
        r = Router(FakeListChatModel(responses=["hmm not sure"]))
        assert r.classify("What is an ETF?") == "finance_qa"

    def test_llm_failure_falls_back_to_heuristic(self):
        r = Router(ExplodingLLM(responses=["x"]))
        assert r.classify("How much tax will I owe?") == "tax_education"

    @pytest.mark.parametrize("query,expected", [
        ("Analyze my portfolio please", "portfolio_analysis"),
        ("What's the P/E of AAPL?", "market_analysis"),
        ("How should I save for retirement?", "goal_planning"),
        ("How is a Roth IRA taxed?", "tax_education"),
        ("What is compound interest?", "finance_qa"),
        ("hello there", "clarify"),
    ])
    def test_heuristic_routing(self, query, expected):
        # Force the heuristic by making the LLM raise.
        r = Router(ExplodingLLM(responses=["x"]))
        assert r.classify(query) == expected


# --- Node routing ---

class TestRouting:
    @pytest.mark.parametrize("route,expected", [
        ("finance_qa", "qa reply"),
        ("market_analysis", "market reply"),
        ("goal_planning", "goal reply"),
        ("tax_education", "tax reply"),
    ])
    def test_routes_to_agent(self, route, expected):
        reply = build_assistant(route).chat("some question", thread_id=route)
        assert reply.route == route
        assert reply.text == expected

    def test_clarify(self):
        reply = build_assistant("clarify").chat("???", thread_id="c")
        assert "help with" in reply.text.lower()

    def test_agent_error_is_caught(self):
        agents = default_agents()
        agents["finance_qa"] = RaisingAgent()
        reply = build_assistant("finance_qa", agents=agents).chat("q", thread_id="e")
        assert "something went wrong" in reply.text.lower()


# --- Portfolio extract -> confirm flow ---

class TestPortfolioFlow:
    HOLDINGS_JSON = '{"holdings": [{"ticker": "AAPL", "quantity": 10, "cost_basis_per_share": 150}, {"ticker": "MSFT", "quantity": 5, "cost_basis_per_share": null}]}'

    def test_echo_back_then_confirm(self):
        agents = default_agents()
        assistant = build_assistant("portfolio_analysis", agents=agents,
                                    extractor_responses=[self.HOLDINGS_JSON])

        # Turn 1: state holdings -> agent echoes back, does NOT analyze yet.
        r1 = assistant.chat("I own 10 Apple and 5 Microsoft at 150", thread_id="p1")
        assert "AAPL" in r1.text and "MSFT" in r1.text
        assert "confirm" in r1.text.lower()
        assert agents["portfolio_analysis"].last_input is None  # not run yet

        # Turn 2: confirm -> now it analyzes.
        r2 = assistant.chat("yes", thread_id="p1")
        assert r2.text == "portfolio reply"
        pin = agents["portfolio_analysis"].last_input
        assert pin is not None
        assert pin.confirmed_by_user is True
        assert {h.ticker for h in pin.holdings} == {"AAPL", "MSFT"}

    def test_no_holdings_asks(self):
        assistant = build_assistant("portfolio_analysis",
                                    extractor_responses=['{"holdings": []}'])
        reply = assistant.chat("analyze my portfolio", thread_id="p2")
        assert "tell me what you hold" in reply.text.lower()

    def test_negative_after_echo_clears_pending(self):
        agents = default_agents()
        assistant = build_assistant("portfolio_analysis", agents=agents,
                                    extractor_responses=[self.HOLDINGS_JSON])
        assistant.chat("I own 10 AAPL", thread_id="p3")
        reply = assistant.chat("no", thread_id="p3")
        assert "again" in reply.text.lower()
        assert agents["portfolio_analysis"].last_input is None

    def test_threads_are_isolated(self):
        # A pending draft on one thread doesn't leak into another.
        agents = default_agents()
        assistant = build_assistant("portfolio_analysis", agents=agents,
                                    extractor_responses=[self.HOLDINGS_JSON])
        assistant.chat("I own 10 AAPL", thread_id="A")
        # New thread, 'yes' with no pending -> re-extraction path, not an analysis.
        reply = assistant.chat("yes", thread_id="B")
        assert agents["portfolio_analysis"].last_input is None
        assert reply.route == "portfolio_analysis"
