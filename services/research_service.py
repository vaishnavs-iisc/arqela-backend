"""
Research service — runs the legacy research agent graph.
"""
import logging

from core.research_agent.graph import research_graph

logger = logging.getLogger("ResearchService")


def run_research(query: str, session_id: str) -> dict:
    """
    Invoke the research graph synchronously and return the final state dict.
    Raises on unrecoverable graph errors.
    """
    initial_state = {
        "user_query": query,
        "session_id": session_id,
        "is_safe": False,
        "safety_reason": "",
        "is_cache_hit": False,
        "search_query": "",
        "raw_sources": "",
        "summary": "",
        "draft_report": "",
        "critique": "",
        "accuracy_score": 0,
        "loop_count": 0,
        "is_finalized": False,
        "agent_logs": [],
    }
    final_state = research_graph.invoke(initial_state)
    return {
        "report": final_state["draft_report"],
        "logs": final_state["agent_logs"],
        "accuracy_score": final_state.get("accuracy_score", 5),
        "is_cache_hit": final_state.get("is_cache_hit", False),
    }
