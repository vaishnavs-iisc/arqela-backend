"""
Builds and compiles the Hypothesis Tester LangGraph StateGraph.

Topology:
  START → analyze_hypothesis → [advocate, adversary] → arbiter → END
  (advocate and adversary run concurrently via LangGraph fan-out)
"""
import logging

from langgraph.graph import StateGraph, START, END

from core.hypothesis.state import HypothesisState
from core.hypothesis.nodes import (
    analyze_hypothesis_node,
    advocate_node,
    adversary_node,
    arbiter_node,
)

logger = logging.getLogger("HypothesisGraph")


def _build_graph() -> StateGraph:
    builder = StateGraph(HypothesisState)

    builder.add_node("analyze_hypothesis", analyze_hypothesis_node)
    builder.add_node("advocate", advocate_node)
    builder.add_node("adversary", adversary_node)
    builder.add_node("arbiter", arbiter_node)

    builder.add_edge(START, "analyze_hypothesis")
    # Fan-out: both advocate and adversary start after analysis completes
    builder.add_edge("analyze_hypothesis", "advocate")
    builder.add_edge("analyze_hypothesis", "adversary")
    # Fan-in: arbiter waits for both to complete
    builder.add_edge("advocate", "arbiter")
    builder.add_edge("adversary", "arbiter")
    builder.add_edge("arbiter", END)

    return builder.compile()


# Singleton compiled graph — imported by the service layer
hypothesis_graph = _build_graph()


def build_initial_state(hypothesis: str, domain: str) -> dict:
    """Return a zeroed initial state dict for graph invocation."""
    return {
        "raw_hypothesis": hypothesis,
        "academic_domain": domain,
        "core_claim": "",
        "underlying_assumptions": [],
        "causal_chain": [],
        "advocate_keywords": [],
        "adversary_keywords": [],
        "advocate_sources": "",
        "adversary_sources": "",
        "supporting_evidence": "",
        "counter_evidence": "",
        "vulnerability_score": 0,
        "empirical_evidence_score": 0,
        "logical_consistency_score": 0,
        "confounder_vulnerability_score": 0,
        "methodological_feasibility_score": 0,
        "expected_effect_size": "",
        "statistical_power_estimation": "",
        "scientific_consensus_index": 0.0,
        "bias_vulnerability_score": 0,
        "evaluation_summary": "",
        "critical_weaknesses": [],
        "proposed_validation_protocol": "",
        "agent_logs": [],
    }
