"""
AI Finance Assistant — Streamlit UI.

A thin client over the FastAPI backend (src/api). It imports no agent code — all
work happens over HTTP via ApiClient, so this can be deployed separately from the
agents. Set API_BASE_URL to point at the backend (default http://localhost:8000).

Run:  streamlit run src/web_app/app.py   (with the API server running)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root on path

import pandas as pd
import streamlit as st

from src.web_app.api_client import ApiClient
from src.web_app.render import render_badge, render_result

st.set_page_config(page_title="AI Finance Assistant", page_icon="📈", layout="wide")

DISCLAIMER = (
    "Educational information only, not financial or tax advice. "
    "Consult a licensed professional before making decisions."
)

GOAL_TYPES = ["retirement", "house", "education", "emergency_fund", "custom"]
DRAWDOWN = {"Sell to stop losses": "sell", "Hold and wait": "hold", "Buy more": "buy_more"}
FLEX = {"Fixed": "fixed", "Somewhat flexible": "somewhat_flexible", "Flexible": "flexible"}


@st.cache_resource
def api() -> ApiClient:
    return ApiClient()


client = api()

try:
    client.health()
except Exception:
    st.error(
        "Can't reach the backend API. Start it with "
        "`uvicorn src.api.main:app` and set API_BASE_URL if it isn't on "
        "http://localhost:8000."
    )
    st.stop()


def send(thread_id: str, text: str):
    with st.spinner("Thinking…"):
        client.chat(text, thread_id=thread_id)
    st.rerun()


# Ensure an active thread.
if "thread_id" not in st.session_state:
    threads = client.list_threads()
    st.session_state.thread_id = threads[0]["id"] if threads else client.create_thread()["thread_id"]


# --- sidebar: thread nav ----------------------------------------------------

with st.sidebar:
    st.markdown("### 📈 Finance assistant")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.thread_id = client.create_thread()["thread_id"]
        st.rerun()

    st.caption("Chats")
    for t in client.list_threads():
        active = t["id"] == st.session_state.thread_id
        if st.button(("• " if active else "") + t["title"], key=f"th_{t['id']}",
                     use_container_width=True):
            st.session_state.thread_id = t["id"]
            st.rerun()

    st.divider()
    st.caption(DISCLAIMER)


# --- main: tabs -------------------------------------------------------------

chat_tab, portfolio_tab, market_tab, goals_tab = st.tabs(
    ["💬 Chat", "📊 Portfolio", "📈 Market", "🎯 Goals"]
)


# ============================ CHAT ==========================================

with chat_tab:
    tid = st.session_state.thread_id

    for m in client.get_messages(tid):
        if m["role"] == "user":
            content = m["content"]
            if content.startswith("Please analyze the portfolio holdings"):
                content = "📎 " + (content.split("'")[1] if "'" in content else "Uploaded document")
            st.chat_message("user").write(content)
        else:
            with st.chat_message("assistant"):
                route = m.get("route")
                if route:
                    render_badge(route)
                st.write(m["content"])
                render_result(route, m.get("result_data"))

    with st.expander("📎 Upload a portfolio document (CSV, PDF, TXT)"):
        uploaded = st.file_uploader("Document", type=["csv", "pdf", "txt", "tsv"],
                                    label_visibility="collapsed", key=f"up_{tid}")
        if uploaded is not None and st.button("Analyze this document"):
            with st.spinner("Reading document…"):
                try:
                    client.upload(tid, uploaded.name, uploaded.getvalue())
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if prompt := st.chat_input("Ask about stocks, your portfolio, goals, or taxes…"):
        send(tid, prompt)


# ============================ PORTFOLIO =====================================

with portfolio_tab:
    st.subheader("Portfolio analysis")
    st.caption("Enter your holdings, then analyze. Cost basis is optional (enables gain/loss).")

    default = pd.DataFrame([
        {"ticker": "AAPL", "quantity": 10, "cost_basis_per_share": 150.0},
        {"ticker": "MSFT", "quantity": 5, "cost_basis_per_share": None},
    ])
    edited = st.data_editor(default, num_rows="dynamic", use_container_width=True,
                            key="port_editor")

    if st.button("Analyze portfolio", type="primary"):
        holdings = []
        for _, r in edited.iterrows():
            if not str(r.get("ticker") or "").strip():
                continue
            holdings.append({
                "ticker": str(r["ticker"]),
                "quantity": float(r["quantity"]),
                "cost_basis_per_share": (float(r["cost_basis_per_share"])
                                         if pd.notna(r.get("cost_basis_per_share")) else None),
            })
        if not holdings:
            st.warning("Add at least one holding.")
        else:
            try:
                with st.spinner("Analyzing…"):
                    result = client.analyze_portfolio(holdings)
                render_result("portfolio_analysis", result)
                st.write(result.get("narrative", ""))
            except Exception as e:
                st.error(f"Couldn't analyze: {e}")


# ============================ MARKET ========================================

with market_tab:
    st.subheader("Market analysis")
    col1, col2 = st.columns([2, 1])
    ticker = col1.text_input("Ticker symbol", value="AAPL").strip().upper()
    kind = col2.selectbox("Analysis", ["Snapshot", "Trend", "Metrics"])

    if st.button("Look up", type="primary") and ticker:
        atype = {"Snapshot": "snapshot", "Trend": "trend", "Metrics": "metric"}[kind]
        try:
            with st.spinner("Fetching market data…"):
                result = client.analyze_market(ticker, atype, "1y")
            render_result("market_analysis", result)
            st.write(result.get("narrative", ""))
        except Exception as e:
            st.error(f"Couldn't fetch: {e}")


# ============================ GOALS =========================================

with goals_tab:
    st.subheader("Goal planning")
    c1, c2, c3 = st.columns(3)
    goal = c1.selectbox("Goal", GOAL_TYPES)
    target = c2.number_input("Target amount ($)", min_value=0, value=1_000_000, step=10_000)
    horizon = c3.slider("Years to goal", 1, 50, 30)

    c4, c5 = st.columns(2)
    savings = c4.number_input("Current savings ($)", min_value=0, value=25_000, step=1_000)
    monthly = c5.number_input("Monthly contribution ($)", min_value=0, value=800, step=50)

    st.caption("Quick risk check")
    c6, c7 = st.columns(2)
    drawdown = c6.radio("If your portfolio dropped 20%, you'd…", list(DRAWDOWN), index=1)
    flex = c7.radio("This goal's target/date is…", list(FLEX), index=1)

    if st.button("Build plan", type="primary"):
        payload = {
            "goal_type": goal,
            "target_amount": target or None,
            "current_savings": savings,
            "monthly_contribution": monthly,
            "time_horizon_years": horizon,
            "risk_answers": {
                "drawdown_reaction": DRAWDOWN[drawdown],
                "goal_flexibility": FLEX[flex],
            },
        }
        try:
            with st.spinner("Projecting…"):
                result = client.plan_goal(payload)
            render_result("goal_planning", result)
            st.write(result.get("narrative", ""))
            if result.get("assumptions"):
                with st.expander("Assumptions"):
                    for a in result["assumptions"]:
                        st.markdown(f"- {a}")
        except Exception as e:
            st.error(f"Couldn't plan: {e}")
