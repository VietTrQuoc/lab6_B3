"""
Compatibility wrapper for the active LangGraph agent implementation.

The application and tests use `app.agent_langgraph` directly. This module is
kept only so older imports of `app.agent` continue to work without carrying a
second, stale agent implementation.
"""

from app.agent_langgraph import *  # noqa: F401,F403
