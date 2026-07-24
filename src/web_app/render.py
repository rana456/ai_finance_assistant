"""
Rendering helpers: turn an agent's `result_data` into Streamlit visuals.

The chat and the structured tabs both call these, so a portfolio result looks
the same whether it came from a typed message or the Portfolio tab.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# route -> (label, background, text color). Self-contained pill colors.
AGENT_BADGES = {
    "finance_qa": ("Finance Q&A", "#E1F5EE", "#085041"),
    "portfolio_analysis": ("Portfolio analysis", "#EEEDFE", "#3C3489"),
    "market_analysis": ("Market analysis", "#E6F1FB", "#0C447C"),
    "goal_planning": ("Goal planning", "#FAEEDA", "#633806"),
    "tax_education": ("Tax education", "#FBEAF0", "#72243E"),
    "clarify": ("Assistant", "#F1EFE8", "#444441"),
}

_PIE_COLORS = ["#7F77DD", "#5DCAA5", "#378ADD", "#EF9F27", "#D4537E", "#97C459",
               "#D85A30", "#888780"]


def render_badge(route: str) -> None:
    label, bg, fg = AGENT_BADGES.get(route, AGENT_BADGES["clarify"])
    st.markdown(
        f"<span style='font-size:12px;font-weight:500;padding:2px 10px;"
        f"border-radius:20px;background:{bg};color:{fg};'>{label}</span>",
        unsafe_allow_html=True,
    )


def render_result(route: str, data: dict | None) -> None:
    """Dispatch a result payload to the right renderer."""
    if not data:
        return
    if route == "portfolio_analysis":
        _render_portfolio(data)
    elif route == "market_analysis":
        _render_market(data)
    elif route == "goal_planning":
        _render_goal(data)
    elif route in ("finance_qa", "tax_education"):
        _render_citations(data)


def _allocation_pie(slices: list[dict], title: str) -> None:
    if not slices:
        return
    fig = go.Figure(go.Pie(
        labels=[s["label"] for s in slices],
        values=[s["value"] for s in slices],
        hole=0.55, sort=False,
        marker=dict(colors=_PIE_COLORS[: len(slices)]),
        textinfo="label+percent",
    ))
    fig.update_layout(
        title=title, height=280, margin=dict(t=40, b=0, l=0, r=0),
        showlegend=False, paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_portfolio(data: dict) -> None:
    m = data.get("metrics", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total value", f"${m.get('total_value', 0):,.0f}")
    gl = m.get("total_gain_loss_percent")
    c2.metric("Gain / loss", f"{gl:+.1f}%" if gl is not None else "—")
    c3.metric("Risk level", str(m.get("risk_level", "—")).title())

    c4, c5 = st.columns([3, 2])
    with c4:
        _allocation_pie(m.get("allocation_by_ticker", []), "Allocation by holding")
    with c5:
        st.metric("Diversification", f"{m.get('diversification_score', 0):.0f}/100")
        for w in m.get("concentration_warnings", []):
            st.warning(w, icon="⚠️")
    if data.get("failed_tickers"):
        st.caption("Couldn't price: " + ", ".join(data["failed_tickers"]))


def _render_market(data: dict) -> None:
    snapshots = data.get("snapshots", [])
    if snapshots:
        cols = st.columns(min(len(snapshots), 3))
        for i, s in enumerate(snapshots):
            chg = s.get("change_pct")
            cols[i % len(cols)].metric(
                s.get("ticker", "—"),
                f"${s.get('price', 0):,.2f}",
                f"{chg:+.2f}%" if chg is not None else None,
            )
    for t in data.get("trends", []):
        st.caption(
            f"{t['ticker']} · {t['period']}: {t['pct_change']:+.1f}% "
            f"({t['trend_signal']})"
        )
    news = data.get("news")
    if news and news.get("articles"):
        with st.expander(f"News sentiment ({news.get('overall_label') or 'n/a'})"):
            for a in news["articles"]:
                st.markdown(f"**{a['title']}** — {a.get('source', '')}")


def _render_goal(data: dict) -> None:
    proj = data.get("projection")
    alloc = data.get("allocation")
    if proj:
        c1, c2, c3 = st.columns(3)
        c1.metric("Projected (base)", f"${proj['nominal_base']:,.0f}")
        c2.metric("Today's dollars", f"${proj['real_base']:,.0f}")
        c3.metric("Contributed", f"${proj['total_contributions']:,.0f}")
        series = proj.get("year_by_year", [])
        if series:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=[p["year"] for p in series], y=[p["balance"] for p in series],
                name="Projected value", line=dict(color="#7F77DD", width=2)))
            fig.add_trace(go.Scatter(
                x=[p["year"] for p in series], y=[p["contributed"] for p in series],
                name="Contributions", line=dict(color="#888780", width=1.5, dash="dot")))
            fig.update_layout(
                height=300, margin=dict(t=20, b=0, l=0, r=0),
                xaxis_title="Year", yaxis_title="Value ($)",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.25))
            st.plotly_chart(fig, use_container_width=True)
    if alloc:
        st.caption(
            f"{alloc['risk_tolerance'].title()} framework: "
            f"{alloc['stock_pct']:.0f}% stocks / {alloc['bond_pct']:.0f}% bonds / "
            f"{alloc['cash_pct']:.0f}% cash"
        )
    a = data.get("assessment")
    if a:
        if a["on_track"]:
            st.success(f"On track to reach ${a['target_amount']:,.0f}.")
        else:
            st.info(
                f"Short by ${a['gap_amount']:,.0f}. Reaching the target would take "
                f"about ${a['required_monthly_contribution']:,.0f}/month."
            )


def _render_citations(data: dict) -> None:
    cites = data.get("citations", [])
    if not cites:
        return
    with st.expander(f"Sources ({len(cites)})"):
        seen = set()
        for c in cites:
            key = (c["title"], c["source"])
            if key in seen:
                continue
            seen.add(key)
            line = f"**{c['title']}** — {c['source']}"
            if c.get("url"):
                line += f"  ·  [link]({c['url']})"
            st.markdown(line)
