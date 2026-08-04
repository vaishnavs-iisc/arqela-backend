"""
LangGraph node functions for the Hypothesis Tester workflow.
Pure functions — no HTTP, no DB imports.
Primary model: Cohere Command-R Plus (for scientific citations & research).
Secondary model: Cohere Command-R / Groq Llama-3.3 70B (for chat & tasks).
Gemini is COMPLETELY EXCLUDED from responses.
"""
import json
import logging
import litellm
from concurrent.futures import ThreadPoolExecutor

from config import config
from core.hypothesis.state import HypothesisState
from core.hypothesis.search import academic_search, retry_on_exception

litellm.drop_params = True

logger = logging.getLogger("HypothesisNodes")


def safe_llm_completion(model: str, messages: list, fallback_models: list = None, **kwargs):
    """Call Cohere/Groq with generous 25s timeout and max_tokens optimization."""
    if fallback_models is None:
        fallback_models = ["cohere/command-r-08-2024", "groq/llama-3.3-70b-versatile"]

    kwargs.setdefault("max_tokens", 350)

    try:
        return litellm.completion(model=model, messages=messages, timeout=25, **kwargs)
    except Exception as e:
        logger.warning(f"Primary model '{model}' failed: {e}. Trying fallbacks {fallback_models}.")

    for fallback in fallback_models:
        if fallback == model:
            continue
        try:
            return litellm.completion(model=fallback, messages=messages, timeout=25, **kwargs)
        except Exception as fb_err:
            logger.warning(f"Fallback model '{fallback}' failed: {fb_err}.")

    # Absolute fallback is Cohere Command-R
    return litellm.completion(model="cohere/command-r-08-2024", messages=messages, timeout=25, **kwargs)


# ---------------------------------------------------------------------------
# Domain instructions helper
# ---------------------------------------------------------------------------

def get_domain_specific_instructions(domain: str) -> str:
    """Return domain-tailored evaluation guidelines for the LLM prompt."""
    domain_lower = domain.lower()

    pure_sciences = {"physics", "astrophysics", "chemistry", "biology", "neuroscience",
                     "materials science", "environmental science"}
    social_sciences = {"economics", "sociology", "management",
                       "organizational behavior", "strategy & innovation", "psychology"}

    if domain_lower in pure_sciences:
        return (
            f"As this is a pure science hypothesis in the field of {domain}, prioritise physical/biological "
            "mechanisms, chemical pathways, fundamental mathematical equations, control variables, and "
            "experimental reproducibility. Detail how micro-level interactions scale to macro-level phenomena."
        )
    if domain_lower == "medicine":
        return (
            "As this is a clinical medicine hypothesis, prioritise RCTs, double-blind clinical endpoints, "
            "patient cohort sizes, confounding variables, pharmacokinetic/pharmacodynamic pathways, "
            "hazard/risk ratios, and physiological safety markers."
        )
    if domain_lower in social_sciences:
        return (
            f"As this is a social, behavioural, or management hypothesis in the field of {domain}, focus on "
            "econometric modelling, causal inference, endogeneity issues, selection/survivorship bias, human "
            "incentives, organisational performance indicators, and structural observational boundaries."
        )
    if domain_lower == "software engineering":
        return (
            "As this is a software engineering hypothesis, focus on empirical systems metrics (latency, "
            "throughput, memory allocation, complexity bounds), repository analyses, quantitative developer "
            "performance surveys, and automated test suite coverage."
        )
    return ""


# ---------------------------------------------------------------------------
# Node definitions
# ---------------------------------------------------------------------------

def analyze_hypothesis_node(state: HypothesisState) -> dict:
    """Deconstruct the hypothesis using Cohere Command-R Plus."""
    logger.info("Entering analyze_hypothesis_node (Cohere Command-R Plus)")
    hypothesis = state["raw_hypothesis"]
    domain = state["academic_domain"]
    logs = ["[Node: Theory Breakdown] Deconstructing core scientific claims via Cohere."]

    custom_inst = get_domain_specific_instructions(domain)
    prompt = (
        f"You are a Senior Research Analyst. Deconstruct the following hypothesis within the domain of '{domain}'.\n"
        f"Domain-Specific Guidelines: {custom_inst}\n\n"
        f"Hypothesis: '{hypothesis}'\n\n"
        "You MUST return a JSON object with the following keys:\n"
        "1. 'core_claim' (string): The primary relationship or assertion.\n"
        "2. 'underlying_assumptions' (list of strings): Implicit assumptions that must hold true. Be highly specific and granular, avoiding generic statements.\n"
        "3. 'causal_chain' (list of strings): Detailed step-by-step sequence of how cause leads to effect. Identify exact biological, physical, or logical intermediate variables.\n"
        "4. 'advocate_keywords' (list of strings): 2-3 highly specific academic search queries for supporting literature. Formulate these as multi-word queries featuring exact biochemical, physical, or technical terms to pull high-quality literature. Avoid single-word/generic queries.\n"
        "5. 'adversary_keywords' (list of strings): 2-3 highly specific academic search queries for counter-arguments, alternate pathways, or confounding variables. Avoid generic terms and focus on technical critiques.\n\n"
        "Return ONLY a valid JSON object with no explanation before or after."
    )

    try:
        response = safe_llm_completion(
            model=config.THEORY_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        logs.append("[Theory Breakdown] Parsed core claims, assumptions, and causal links.")
        return {
            "core_claim": result.get("core_claim", hypothesis),
            "underlying_assumptions": result.get("underlying_assumptions", ["Implicit correlation"]),
            "causal_chain": result.get("causal_chain", ["Hypothesis evaluated as a single unit"]),
            "advocate_keywords": result.get("advocate_keywords", [hypothesis]),
            "adversary_keywords": result.get("adversary_keywords", [f"critique of {hypothesis}"]),
            "agent_logs": logs,
        }
    except Exception as e:
        logger.error(f"Hypothesis breakdown failed: {e}")
        return {
            "core_claim": hypothesis,
            "underlying_assumptions": ["Implicit correlation"],
            "causal_chain": ["Hypothesis evaluated as a single unit"],
            "advocate_keywords": [hypothesis],
            "adversary_keywords": [f"critique of {hypothesis}"],
            "agent_logs": logs + [f"[Theory Breakdown] Error: {e}. Reverting to defaults."],
        }


def advocate_node(state: HypothesisState) -> dict:
    """Gather supporting literature concurrently and synthesise with Cohere Command-R Plus."""
    logger.info("Entering advocate_node (Cohere Command-R Plus)")
    core_claim = state["core_claim"]
    keywords = state["advocate_keywords"]
    domain = state["academic_domain"]
    logs = ["[Node: Supporting Evidence Gatherer] Searching literature & compiling citation brief via Cohere."]

    search_payloads = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(academic_search, kw, domain, 3): kw for kw in keywords[:2]}
        for future in futures:
            kw = futures[future]
            try:
                res = future.result(timeout=3)
                search_payloads.append(f"--- Search Query: '{kw}' ---\n{res}")
            except Exception as e:
                logger.warning(f"Search failed for keyword '{kw}': {e}")
                search_payloads.append(f"--- Search Query: '{kw}' ---\nSearch failed: {e}")

    all_search_data = "\n\n".join(search_payloads)
    custom_inst = get_domain_specific_instructions(domain)
    prompt = (
        f"You are a Research Proponent advocating for the following core claim:\n\n"
        f"Claim: '{core_claim}'\n"
        f"Domain-Specific Guidelines: {custom_inst}\n\n"
        f"Search data from academic/literature searches:\n\n{all_search_data}\n\n"
        "Synthesise this into a highly supportive scientific brief. Your goal is to deliver a response of exceptional depth, far exceeding standard conversational chatbots like ChatGPT or Gemini. "
        "Rules:\n"
        "1. Avoid vague generalities or generic scientific filler. Dig deep into specific biochemical, physical, or systemic mechanisms.\n"
        "2. Provide quantitative and empirical precision: quote specific statistical parameters (p-values, hazard ratios, sample sizes (N), Cohen's d, confidence intervals) and experimental methodology (RCTs, cohort replication) from the search data.\n"
        "3. Cite sources using [Title](Link) markdown links. Cite as many distinct papers/links as possible from the provided search data (at least 3-4 distinct citations if available).\n"
        "4. Write exactly 4 bullet points, with each bullet point under 250 characters.\n"
        "5. Use Markdown list format (- bullet). Never fabricate links or make up papers."
    )

    try:
        response = safe_llm_completion(
            model=config.ADVOCATE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        synthesis = response.choices[0].message.content.strip()
        logs.append("[Supporting Evidence Gatherer] Compiled supporting scientific brief via Cohere.")
        return {"advocate_sources": all_search_data, "supporting_evidence": synthesis, "agent_logs": logs}
    except Exception as e:
        logger.error(f"Advocate node failed: {e}")
        return {
            "advocate_sources": all_search_data,
            "supporting_evidence": "Failed to compile supporting evidence.",
            "agent_logs": logs + [f"[Supporting Evidence Gatherer] Error: {e}"],
        }


def adversary_node(state: HypothesisState) -> dict:
    """Gather counter-arguments concurrently and synthesise with Cohere Command-R Plus."""
    logger.info("Entering adversary_node (Cohere Command-R Plus)")
    core_claim = state["core_claim"]
    keywords = state["adversary_keywords"]
    domain = state["academic_domain"]
    logs = ["[Node: Skeptic Auditor] Auditing counter-arguments via Cohere Command-R Plus."]

    search_payloads = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(academic_search, kw, domain, 3): kw for kw in keywords[:2]}
        for future in futures:
            kw = futures[future]
            try:
                res = future.result(timeout=3)
                search_payloads.append(f"--- Search Query: '{kw}' ---\n{res}")
            except Exception as e:
                logger.warning(f"Search failed for keyword '{kw}': {e}")
                search_payloads.append(f"--- Search Query: '{kw}' ---\nSearch failed: {e}")

    all_search_data = "\n\n".join(search_payloads)
    custom_inst = get_domain_specific_instructions(domain)
    prompt = (
        f"You are a Skeptical Scientific Auditor reviewing:\n\n"
        f"Claim: '{core_claim}'\n"
        f"Domain-Specific Guidelines: {custom_inst}\n\n"
        f"Search data regarding critiques, alternatives, and limitations:\n\n{all_search_data}\n\n"
        "Synthesise this into a counterargument brief of exceptional depth, far exceeding standard conversational chatbots like ChatGPT or Gemini. "
        "Rules:\n"
        "1. Identify specific confounding variables, alternative explanations, and structural limits to generalisability.\n"
        "2. Critique the methodological constraints, potential biases (selection, attrition, funding, reporting), and statistical power limits of the positive claims.\n"
        "3. Quote specific contradictory data, statistical discrepancies, or replication failures where available.\n"
        "4. Cite sources using [Title](Link) markdown links. Cite as many distinct papers/links as possible from the provided search data (at least 3-4 distinct citations if available).\n"
        "5. Write exactly 4 bullet points, with each bullet point under 250 characters.\n"
        "6. Use Markdown list format (- bullet). Never fabricate links or make up papers."
    )

    try:
        response = safe_llm_completion(
            model=config.ADVERSARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        synthesis = response.choices[0].message.content.strip()
        logs.append("[Skeptic Auditor] Compiled counter-arguments via Cohere.")
        return {"adversary_sources": all_search_data, "counter_evidence": synthesis, "agent_logs": logs}
    except Exception as e:
        logger.error(f"Adversary node failed: {e}")
        return {
            "adversary_sources": all_search_data,
            "counter_evidence": "Failed to compile counter-arguments.",
            "agent_logs": logs + [f"[Skeptic Auditor] Error: {e}"],
        }


def arbiter_node(state: HypothesisState) -> dict:
    """Synthesise both sides and produce quantified evaluation via Cohere Command-R Plus."""
    logger.info("Entering arbiter_node (Cohere Command-R Plus)")
    core_claim = state["core_claim"]
    adv_brief = state["supporting_evidence"]
    opp_brief = state["counter_evidence"]
    domain = state.get("academic_domain", "Biology")
    logs = ["[Node: Scientific Arbiter] Synthesising evidence and drafting validation protocol via Cohere."]

    custom_inst = get_domain_specific_instructions(domain)
    prompt = (
        f"You are the Scientific Arbiter & Peer Review Chair.\n"
        f"Domain-Specific Guidelines: {custom_inst}\n\n"
        f"Core Claim: '{core_claim}'\n\n"
        f"--- Supporting Evidence ---\n{adv_brief}\n\n"
        f"--- Counterarguments & Gaps ---\n{opp_brief}\n\n"
        "Synthesise both sides and produce a quantified, highly rigorous evaluation that surpasses standard general-purpose models like ChatGPT or Gemini. "
        "Return ONLY a valid JSON object with these exact keys:\n"
        "vulnerability_score (int 1-5), empirical_evidence_score (int 1-5), "
        "logical_consistency_score (int 1-5), confounder_vulnerability_score (int 1-5), "
        "methodological_feasibility_score (int 1-5), expected_effect_size (str), "
        "statistical_power_estimation (str), scientific_consensus_index (float 0.0-1.0), "
        "bias_vulnerability_score (int 1-5), evaluation_summary (str, max 250 chars), "
        "critical_weaknesses (list of 3 strings each under 100 chars), "
        "proposed_validation_protocol (str, max 350 chars, Phase 1/2/3 layout)."
    )

    defaults = {
        "vulnerability_score": 3,
        "empirical_evidence_score": 3,
        "logical_consistency_score": 3,
        "confounder_vulnerability_score": 3,
        "methodological_feasibility_score": 3,
        "expected_effect_size": "Cohen's d = 0.50 (estimated)",
        "statistical_power_estimation": "N=100, Power=0.80, alpha=0.05 (estimated)",
        "scientific_consensus_index": 0.5,
        "bias_vulnerability_score": 3,
        "evaluation_summary": "Failed to synthesise final evaluation due to parser error.",
        "critical_weaknesses": ["Undetermined weaknesses due to process failure"],
        "proposed_validation_protocol": "Phase 1: Design empirical test with control groups.",
    }

    try:
        response = safe_llm_completion(
            model=config.ARBITER_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
        result = json.loads(content)
        logs.append("[Scientific Arbiter] Finalised peer evaluation via Cohere Command-R Plus.")
        return {
            "vulnerability_score": result.get("vulnerability_score", defaults["vulnerability_score"]),
            "empirical_evidence_score": result.get("empirical_evidence_score", defaults["empirical_evidence_score"]),
            "logical_consistency_score": result.get("logical_consistency_score", defaults["logical_consistency_score"]),
            "confounder_vulnerability_score": result.get("confounder_vulnerability_score", defaults["confounder_vulnerability_score"]),
            "methodological_feasibility_score": result.get("methodological_feasibility_score", defaults["methodological_feasibility_score"]),
            "expected_effect_size": result.get("expected_effect_size", defaults["expected_effect_size"]),
            "statistical_power_estimation": result.get("statistical_power_estimation", defaults["statistical_power_estimation"]),
            "scientific_consensus_index": result.get("scientific_consensus_index", defaults["scientific_consensus_index"]),
            "bias_vulnerability_score": result.get("bias_vulnerability_score", defaults["bias_vulnerability_score"]),
            "evaluation_summary": result.get("evaluation_summary", defaults["evaluation_summary"]),
            "critical_weaknesses": result.get("critical_weaknesses", defaults["critical_weaknesses"]),
            "proposed_validation_protocol": result.get("proposed_validation_protocol", defaults["proposed_validation_protocol"]),
            "agent_logs": logs,
        }
    except Exception as e:
        logger.error(f"Arbiter node failed: {e}")
        return {**defaults, "agent_logs": logs + [f"[Scientific Arbiter] Error: {e}"]}
