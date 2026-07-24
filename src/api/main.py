"""
FastAPI backend for the AI Finance Assistant.

Hosts the agents, the LangGraph workflow, the RAG index, and the SQLite-backed
conversation store behind an HTTP/JSON API. The UI (or any client) talks to this
instead of importing the agent code — so keys, the index, and heavy deps live
only here.

Run:  uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from langchain_core.messages import AIMessage, HumanMessage

from src.agents.goal_planning.model import GoalPlanRequest
from src.agents.market_analysis.model import AnalysisType, MarketQuery, Period
from src.agents.portfolio_analysis.model import PortfolioInput
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    CreateThreadResponse,
    MarketRequest,
    MessageOut,
    ThreadInfo,
)
from src.web_app.doc_extract import build_upload_message, extract_text


def create_app(context=None) -> FastAPI:
    """Build the API. `context` can be injected (tests); otherwise it's built
    lazily on first use so importing this module stays cheap."""
    app = FastAPI(
        title="AI Finance Assistant API",
        description="Multi-agent financial education backend. Educational only — not advice.",
        version="1.0.0",
    )
    _state = {"ctx": context}

    def get_ctx():
        if _state["ctx"] is None:
            from src.web_app.context import build_context
            _state["ctx"] = build_context()
        return _state["ctx"]

    # --- health ---

    @app.get("/health")
    def health():
        return {"status": "ok"}

    # --- chat / workflow ---

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, ctx=Depends(get_ctx)):
        thread_id = req.thread_id or ctx.thread_store.create_thread()
        if not _thread_messages(ctx, thread_id):  # first message names the thread
            ctx.thread_store.set_title(
                thread_id, ctx.thread_store.title_from_message(req.message)
            )
        reply = ctx.assistant.chat(
            req.message, thread_id=thread_id, knowledge_level=req.knowledge_level
        )
        ctx.thread_store.touch(thread_id)
        return ChatResponse(thread_id=thread_id, text=reply.text,
                            route=reply.route, data=reply.data)

    @app.post("/chat/upload", response_model=ChatResponse)
    async def chat_upload(thread_id: str = Form(...), file: UploadFile = File(...),
                          ctx=Depends(get_ctx)):
        text = extract_text(file.filename, await file.read())
        if not text:
            raise HTTPException(400, "Couldn't read any text from that file.")
        message = build_upload_message(file.filename, text)
        reply = ctx.assistant.chat(message, thread_id=thread_id)
        ctx.thread_store.touch(thread_id)
        return ChatResponse(thread_id=thread_id, text=reply.text,
                            route=reply.route, data=reply.data)

    # --- threads ---

    @app.post("/threads", response_model=CreateThreadResponse)
    def create_thread(ctx=Depends(get_ctx)):
        tid = ctx.thread_store.create_thread()
        return CreateThreadResponse(thread_id=tid, title="New chat")

    @app.get("/threads", response_model=list[ThreadInfo])
    def list_threads(ctx=Depends(get_ctx)):
        return [ThreadInfo(**t) for t in ctx.thread_store.list_threads()]

    @app.get("/threads/{thread_id}/messages", response_model=list[MessageOut])
    def get_messages(thread_id: str, ctx=Depends(get_ctx)):
        out: list[MessageOut] = []
        for m in _thread_messages(ctx, thread_id):
            if isinstance(m, HumanMessage):
                out.append(MessageOut(role="user", content=m.content))
            elif isinstance(m, AIMessage):
                out.append(MessageOut(
                    role="assistant", content=m.content,
                    route=m.additional_kwargs.get("route"),
                    result_data=m.additional_kwargs.get("result_data"),
                ))
        return out

    @app.delete("/threads/{thread_id}")
    def delete_thread(thread_id: str, ctx=Depends(get_ctx)):
        ctx.thread_store.delete(thread_id)
        return {"ok": True}

    # --- structured agent endpoints ---

    @app.post("/portfolio/analyze")
    def portfolio_analyze(req: PortfolioInput, ctx=Depends(get_ctx)):
        try:
            return ctx.agents["portfolio_analysis"].run(req)
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/market/analyze")
    def market_analyze(req: MarketRequest, ctx=Depends(get_ctx)):
        try:
            query = MarketQuery(
                question=f"{req.analysis_type} for {req.ticker}",
                tickers=[req.ticker.upper()],
                analysis_type=AnalysisType(req.analysis_type),
                period=Period(req.period),
            )
        except ValueError as e:
            raise HTTPException(422, f"Invalid analysis_type or period: {e}")
        return ctx.agents["market_analysis"].run(query)

    @app.post("/goals/plan")
    def goals_plan(req: GoalPlanRequest, ctx=Depends(get_ctx)):
        return ctx.agents["goal_planning"].run(req)

    return app


def _thread_messages(ctx, thread_id: str) -> list:
    state = ctx.assistant.graph.get_state({"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", []) if state and state.values else []


app = create_app()
