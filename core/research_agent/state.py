"""
State type definition for the Research Agent LangGraph workflow.
"""
import operator
from typing import Annotated, List, TypedDict


class AgentState(TypedDict):
    user_query: str
    session_id: str
    is_safe: bool
    safety_reason: str
    is_cache_hit: bool
    search_query: str
    raw_sources: str
    summary: str
    draft_report: str
    critique: str
    accuracy_score: int
    loop_count: int
    is_finalized: bool
    agent_logs: Annotated[List[str], operator.add]
