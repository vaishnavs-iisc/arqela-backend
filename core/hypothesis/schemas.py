"""
Pydantic output schemas for the Hypothesis Tester LLM calls.
Used with litellm structured output.
"""
from typing import List
from pydantic import BaseModel, Field


class HypothesisBreakdown(BaseModel):
    core_claim: str = Field(
        description="The primary relationship or assertion being made."
    )
    underlying_assumptions: List[str] = Field(
        description="Implicit assumptions that must hold true for the hypothesis to be valid."
    )
    causal_chain: List[str] = Field(
        description="Step-by-step sequence of events detailing how the cause leads to the effect."
    )
    advocate_keywords: List[str] = Field(
        description=(
            "2-3 highly specific academic search queries optimised for scientific literature, "
            "e.g. 'NMN supplementation autophagy mammalian lifespan clinical study'. "
            "Avoid single-word/generic queries."
        )
    )
    adversary_keywords: List[str] = Field(
        description=(
            "2-3 highly specific academic search queries to find counter-arguments, "
            "confounding variables, or contradictory studies, e.g. "
            "'NMN lifespan study failure replication cohort'. Avoid single-word/generic queries."
        )
    )


class FinalEvaluation(BaseModel):
    vulnerability_score: int = Field(
        description="Overall vulnerability score from 1 (robust) to 5 (highly vulnerable)."
    )
    empirical_evidence_score: int = Field(
        description="Strength of supporting literature from 1 (no support/contradicted) to 5 (highly validated)."
    )
    logical_consistency_score: int = Field(
        description="Causal coherence and logic from 1 (leaps/contradictions) to 5 (fully coherent)."
    )
    confounder_vulnerability_score: int = Field(
        description="Susceptibility to alternate explanations from 1 (highly vulnerable) to 5 (resilient)."
    )
    methodological_feasibility_score: int = Field(
        description="Feasibility of designing an empirical test from 1 (impossible) to 5 (easy/standard)."
    )
    expected_effect_size: str = Field(
        description="Expected effect size (e.g. Cohen's d = 0.45, Hazard Ratio = 0.72) from current literature."
    )
    statistical_power_estimation: str = Field(
        description="Estimated sample size and power (e.g. N=120, 1-beta=0.80, alpha=0.05)."
    )
    scientific_consensus_index: float = Field(
        description="Consensus score from 0.0 (no consensus/contradicted) to 1.0 (unanimous agreement)."
    )
    bias_vulnerability_score: int = Field(
        description="Vulnerability to common scientific biases from 1 (low risk) to 5 (high risk)."
    )
    evaluation_summary: str = Field(
        description="A 1-2 paragraph professional peer-review style evaluation of the hypothesis."
    )
    critical_weaknesses: List[str] = Field(
        description="The top 3 critical weaknesses, gaps, or confounding factors identified."
    )
    proposed_validation_protocol: str = Field(
        description=(
            "A step-by-step empirical experiment design. Include independent/dependent variables "
            "and control conditions in Markdown format."
        )
    )
