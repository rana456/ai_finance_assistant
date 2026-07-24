"""End-to-end demo of the workflow: routes several queries through the full
graph with real agents, then runs the multi-turn portfolio confirm flow.

    .venv/bin/python scripts/demo_workflow.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.workflow.assistant import build_default_assistant


def turn(assistant, msg, thread="demo"):
    reply = assistant.chat(msg, thread_id=thread)
    print(f"\n[you]   {msg}")
    print(f"[route] {reply.route}")
    print(f"[bot]   {reply.text}")


def main() -> None:
    a = build_default_assistant()

    # Single-turn routing across agents.
    turn(a, "What is dollar-cost averaging?", "t1")
    turn(a, "How is Apple stock doing today?", "t1")
    turn(a, "How is a Roth IRA taxed?", "t1")
    turn(a, "How much should I save monthly to retire with a million dollars?", "t1")

    # Multi-turn portfolio flow (extract -> confirm -> analyze) on its own thread.
    turn(a, "I own 10 shares of Apple at $150 and 5 of Microsoft", "port")
    turn(a, "yes", "port")


if __name__ == "__main__":
    main()
