"""
HTTP client the Streamlit UI uses to talk to the FastAPI backend.

This is the whole coupling between UI and agents now — the UI imports this, not
the agent code. The backend URL comes from API_BASE_URL (default localhost:8000).
"""

from __future__ import annotations

import os

import httpx


class ApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def _json(self, resp: httpx.Response):
        if resp.status_code >= 400:
            detail = resp.json().get("detail", resp.text) if resp.content else resp.reason_phrase
            raise RuntimeError(detail)
        return resp.json()

    # --- health ---

    def health(self) -> dict:
        return self._json(self._client.get("/health"))

    # --- chat / threads ---

    def chat(self, message: str, thread_id: str | None = None,
             knowledge_level: str = "beginner") -> dict:
        return self._json(self._client.post("/chat", json={
            "message": message, "thread_id": thread_id,
            "knowledge_level": knowledge_level,
        }))

    def upload(self, thread_id: str, filename: str, data: bytes) -> dict:
        return self._json(self._client.post(
            "/chat/upload",
            data={"thread_id": thread_id},
            files={"file": (filename, data)},
        ))

    def create_thread(self) -> dict:
        return self._json(self._client.post("/threads"))

    def list_threads(self) -> list[dict]:
        return self._json(self._client.get("/threads"))

    def get_messages(self, thread_id: str) -> list[dict]:
        return self._json(self._client.get(f"/threads/{thread_id}/messages"))

    def delete_thread(self, thread_id: str) -> dict:
        return self._json(self._client.delete(f"/threads/{thread_id}"))

    # --- structured agents ---

    def analyze_portfolio(self, holdings: list[dict]) -> dict:
        return self._json(self._client.post("/portfolio/analyze", json={
            "holdings": holdings, "source": "manual",
        }))

    def analyze_market(self, ticker: str, analysis_type: str, period: str = "1y") -> dict:
        return self._json(self._client.post("/market/analyze", json={
            "ticker": ticker, "analysis_type": analysis_type, "period": period,
        }))

    def plan_goal(self, payload: dict) -> dict:
        return self._json(self._client.post("/goals/plan", json=payload))
