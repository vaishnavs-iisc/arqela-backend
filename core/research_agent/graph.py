"""
Builds and compiles the Research Agent LangGraph StateGraph.

The graph wires together guardrail → cache check → search → summarize →
writer ↔ verify (loop up to 3 times) → save_memories.
"""
import logging
from functools import partial

import redis as redis_lib
from langgraph.graph import StateGraph, START, END

from config import config
from database import db_manager
from core.research_agent.state import AgentState
from core.research_agent.nodes import (
    guardrail_node,
    cache_check_node,
    search_agent_node,
    summarize_agent_node,
    writer_agent_node,
    verify_agent_node,
    save_memories_node,
    rejection_node,
)

logger = logging.getLogger("ResearchAgentGraph")

# ---------------------------------------------------------------------------
# Redis client (shared across nodes that need it)
# ---------------------------------------------------------------------------
redis_client = redis_lib.Redis(
    host=config.REDIS_HOST,
    port=config.REDIS_PORT,
    db=0,
    decode_responses=True,
)

# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def route_guardrail(state: AgentState) -> str:
    return "check_cache" if state["is_safe"] else "reject"


def route_cache(state: AgentState) -> str:
    return "end" if state["is_cache_hit"] else "run_search"


def route_evaluator(state: AgentState) -> str:
    if state["is_finalized"] or state["loop_count"] >= 3:
        return "save_memories"
    return "re-write"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def _build_graph() -> StateGraph:
    """Construct and compile the research agent graph."""
    builder = StateGraph(AgentState)

    # Bind infrastructure dependencies into nodes that need them
    _cache_check = partial(cache_check_node, redis_client=redis_client)
    _writer = partial(writer_agent_node, redis_client=redis_client)
    _save = partial(save_memories_node, redis_client=redis_client, db_manager=db_manager)

    builder.add_node("guardrail", guardrail_node)
    builder.add_node("cache_check", _cache_check)
    builder.add_node("search", search_agent_node)
    builder.add_node("summarize", summarize_agent_node)
    builder.add_node("writer", _writer)
    builder.add_node("verify", verify_agent_node)
    builder.add_node("save_memories", _save)
    builder.add_node("rejection", rejection_node)

    builder.add_edge(START, "guardrail")
    builder.add_conditional_edges("guardrail", route_guardrail, {"check_cache": "cache_check", "reject": "rejection"})
    builder.add_conditional_edges("cache_check", route_cache, {"end": END, "run_search": "search"})
    builder.add_edge("search", "summarize")
    builder.add_edge("summarize", "writer")
    builder.add_edge("writer", "verify")
    builder.add_conditional_edges("verify", route_evaluator, {"save_memories": "save_memories", "re-write": "writer"})
    builder.add_edge("save_memories", END)
    builder.add_edge("rejection", END)

    return builder.compile()


# Singleton compiled graph — imported by the service layer
research_graph = _build_graph()
