"""
State type definition for the Hypothesis Tester LangGraph workflow.
"""
import operator
from typing import Annotated, List, TypedDict


class HypothesisState(TypedDict):
    raw_hypothesis: str
    academic_domain: str

    core_claim: str
    underlying_assumptions: List[str]
    causal_chain: List[str]
    advocate_keywords: List[str]
    adversary_keywords: List[str]

    advocate_sources: str
    adversary_sources: str

    supporting_evidence: str
    counter_evidence: str
    companies_and_labs: str

    vulnerability_score: int
    empirical_evidence_score: int
    logical_consistency_score: int
    confounder_vulnerability_score: int
    methodological_feasibility_score: int
    expected_effect_size: str
    statistical_power_estimation: str
    scientific_consensus_index: float
    bias_vulnerability_score: int
    evaluation_summary: str
    critical_weaknesses: List[str]
    proposed_validation_protocol: str

    agent_logs: Annotated[List[str], operator.add]
