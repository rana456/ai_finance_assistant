"""
Persistence for chat threads.

Two things persist, both in one SQLite file:
- The LangGraph conversation checkpoints (via SqliteSaver) — the actual state
  of each thread, keyed by thread_id.
- A small `threads` metadata table (id, title, timestamps) — so the sidebar can
  list past chats with titles, which the checkpointer alone doesn't provide.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

DEFAULT_DB = Path(__file__).resolve().parents[1] / "data" / "assistant.db"


def get_checkpointer(db_path: Path = DEFAULT_DB) -> SqliteSaver:
    """A SqliteSaver on the shared DB (its own connection; thread-safe for
    Streamlit's re-runs)."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    return SqliteSaver(conn)


class ThreadStore:
    """CRUD for chat-thread metadata."""

    def __init__(self, db_path: Path = DEFAULT_DB):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS threads (
                   id TEXT PRIMARY KEY,
                   title TEXT NOT NULL,
                   created_at TEXT NOT NULL,
                   updated_at TEXT NOT NULL
               )"""
        )
        self._conn.commit()

    def create_thread(self, title: str = "New chat") -> str:
        thread_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO threads (id, title, created_at, updated_at) VALUES (?,?,?,?)",
            (thread_id, title[:80], now, now),
        )
        self._conn.commit()
        return thread_id

    def list_threads(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, title, updated_at FROM threads ORDER BY updated_at DESC"
        ).fetchall()
        return [{"id": r[0], "title": r[1], "updated_at": r[2]} for r in rows]

    def set_title(self, thread_id: str, title: str) -> None:
        self._conn.execute(
            "UPDATE threads SET title = ? WHERE id = ?", (title[:80], thread_id)
        )
        self._conn.commit()

    def touch(self, thread_id: str) -> None:
        self._conn.execute(
            "UPDATE threads SET updated_at = ? WHERE id = ?",
            (datetime.now(timezone.utc).isoformat(), thread_id),
        )
        self._conn.commit()

    def delete(self, thread_id: str) -> None:
        self._conn.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
        self._conn.commit()

    def title_from_message(self, message: str) -> str:
        """A concise thread title derived from the first user message."""
        title = " ".join(message.strip().split())
        return (title[:47] + "...") if len(title) > 50 else title or "New chat"
