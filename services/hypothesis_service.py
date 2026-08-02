"""
Hypothesis service — coordinates graph execution, DB persistence, and caching.
Routes call this service; the service never deals with HTTP concerns.
"""
import uuid
import logging
from typing import Iterator, Optional

from database import db_manager
from utils import get_embedding
from core.hypothesis.graph import hypothesis_graph, build_initial_state
from core.hypothesis.nodes import get_domain_specific_instructions
from core.hypothesis.search import academic_search
from repositories import hypothesis_repo
from services import cache_service
import litellm
from config import config

logger = logging.getLogger("HypothesisService")


# ---------------------------------------------------------------------------
# Evaluation (streaming)
# ---------------------------------------------------------------------------

def stream_evaluation(hypothesis: str, domain: str, user_id: str) -> Iterator[dict]:
    """
    Run the hypothesis evaluation pipeline and yield progress / result dicts.
    Each yielded dict has a 'type' key: 'progress', 'result', or 'error'.
    """
    active_conversation_id = str(uuid.uuid4())

    # 1. Embedding + vector cache check
    try:
        query_vector = get_embedding(hypothesis)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        yield {"type": "error", "message": "Unable to initialise analysis. Check your network and try again."}
        return

    try:
        with db_manager.get_connection() as conn:
            cached = hypothesis_repo.find_similar_hypothesis(conn, user_id, query_vector)
            if cached:
                yield {"type": "progress", "node": "cache", "percentage": 100, "message": "Retrieving cached evaluation..."}
                yield {"type": "result", "data": cached}
                return
    except Exception as e:
        logger.error(f"Cache lookup failed: {e}")

    # 2. Cache miss → run multi-agent graph with progress streaming
    logger.info("Cache MISS — triggering multi-agent graph.")
    yield {"type": "progress", "node": "start", "percentage": 5, "message": "Initiating multi-agent scientific evaluation..."}

    initial_state = build_initial_state(hypothesis, domain)
    final_state = dict(initial_state)

    try:
        for event in hypothesis_graph.stream(initial_state):
            for node_name, state_update in event.items():
                final_state.update(state_update)
                progress_map = {
                    "analyze_hypothesis": (25, "Deconstructing core claims and causal assumptions..."),
                    "advocate":           (50, "Gathering supporting evidence from academic literature..."),
                    "adversary":          (75, "Auditing counter-arguments, biases, and confounders..."),
                    "arbiter":            (95, "Synthesising consensus and compiling validation protocol..."),
                }
                if node_name in progress_map:
                    pct, msg = progress_map[node_name]
                    yield {"type": "progress", "node": node_name, "percentage": pct, "message": msg}
    except Exception as e:
        logger.error(f"Graph stream failed: {e}")
        yield {"type": "error", "message": "An unexpected issue occurred during evaluation. Please try again."}
        return

    # 3. Persist to DB
    result = {
        "conversation_id":              active_conversation_id,
        "raw_hypothesis":               final_state["raw_hypothesis"],
        "academic_domain":              final_state["academic_domain"],
        "core_claim":                   final_state["core_claim"],
        "underlying_assumptions":       final_state["underlying_assumptions"],
        "causal_chain":                 final_state["causal_chain"],
        "supporting_evidence":          final_state["supporting_evidence"],
        "counter_evidence":             final_state["counter_evidence"],
        "vulnerability_score":          final_state["vulnerability_score"],
        "empirical_evidence_score":     final_state["empirical_evidence_score"],
        "logical_consistency_score":    final_state["logical_consistency_score"],
        "confounder_vulnerability_score": final_state["confounder_vulnerability_score"],
        "methodological_feasibility_score": final_state["methodological_feasibility_score"],
        "expected_effect_size":         final_state["expected_effect_size"],
        "statistical_power_estimation": final_state["statistical_power_estimation"],
        "scientific_consensus_index":   final_state["scientific_consensus_index"],
        "bias_vulnerability_score":     final_state["bias_vulnerability_score"],
        "evaluation_summary":           final_state["evaluation_summary"],
        "critical_weaknesses":          final_state["critical_weaknesses"],
        "proposed_validation_protocol": final_state["proposed_validation_protocol"],
        "agent_logs":                   final_state["agent_logs"],
        "is_cache_hit":                 False,
        "conversation_history":         [],
    }

    try:
        with db_manager.get_connection() as conn:
            _, db_conv_id = hypothesis_repo.upsert_evaluation(conn, user_id, result, query_vector)
            if db_conv_id:
                result["conversation_id"] = db_conv_id
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to store evaluation in PostgreSQL: {e}")

    yield {"type": "result", "data": result}


# ---------------------------------------------------------------------------
# History & detail
# ---------------------------------------------------------------------------

def get_history(user_id: str) -> list[dict]:
    with db_manager.get_connection() as conn:
        return hypothesis_repo.get_history(conn, user_id)

def get_detail(conversation_id: str, user_id: str) -> Optional[dict]:
    with db_manager.get_connection() as conn:
        return hypothesis_repo.get_detail(conn, conversation_id, user_id)


def clear_history(user_id: str) -> None:
    with db_manager.get_connection() as conn:
        hypothesis_repo.clear_all(conn, user_id)
        conn.commit()


# ---------------------------------------------------------------------------
# Conversational Q&A streaming
# ---------------------------------------------------------------------------

def stream_conversation(conversation_id: str, new_message: str, user_id: str) -> Iterator[str]:
    """
    Stream a token-by-token chat response, enriched with:
      - The active hypothesis context from DB
      - Semantically related historical evaluations
      - Optional fresh literature search
    Yields raw text chunks (not SSE-wrapped).
    """
    active_context = ""
    academic_domain = "Biology"
    row = None

    try:
        with db_manager.get_connection() as conn:
            row = hypothesis_repo.get_by_conversation_id(conn, conversation_id, user_id)
            if row:
                academic_domain = row["academic_domain"]
                active_context = (
                    f"Active Hypothesis: {row['raw_hypothesis']}\n"
                    f"Domain: {row['academic_domain']}\n"
                    f"Core Claim: {row['core_claim']}\n"
                    f"Underlying Assumptions: {row['underlying_assumptions']}\n"
                    f"Causal Chain: {row['causal_chain']}\n"
                    f"Overall Vulnerability Score: {row['vulnerability_score']}/5\n"
                    f"Empirical Evidence Score: {row['empirical_evidence_score']}/5\n"
                    f"Logical Consistency Score: {row['logical_consistency_score']}/5\n"
                    f"Confounder Resiliency Score: {row['confounder_vulnerability_score']}/5\n"
                    f"Methodological Feasibility Score: {row['methodological_feasibility_score']}/5\n"
                    f"Evaluation Summary: {row['evaluation_summary']}\n"
                    f"Critical Weaknesses: {row['critical_weaknesses']}\n"
                    f"Proposed Protocol: {row['proposed_validation_protocol']}\n"
                )
    except Exception as e:
        logger.error(f"Failed to fetch active hypothesis context: {e}")

    # Semantic precedent lookup
    semantic_context = ""
    try:
        query_vector = get_embedding(new_message)
        with db_manager.get_connection() as conn:
            precedents = hypothesis_repo.find_related_precedents(conn, user_id, query_vector, conversation_id)
            refs = [
                f"- Related Precedent: {p['hypothesis']}\n"
                f"  Core Claim: {p['core_claim']}\n"
                f"  Summary: {p['evaluation_summary']}\n"
                f"  Critical Weaknesses: {p['critical_weaknesses']}\n"
                for p in precedents if p["distance"] < 0.3
            ]
            if refs:
                semantic_context = "\n### Related Historical Precedents:\n" + "\n".join(refs)
    except Exception as e:
        logger.error(f"Semantic precedent lookup failed: {e}")

    # Optional fresh literature search
    fresh_literature = ""
    search_triggers = {"search", "look up", "paper", "study", "reference", "citation",
                       "arxiv", "pubmed", "evidence", "proof", "find", "latest", "recent", "journal"}
    if any(t in new_message.lower() for t in search_triggers):
        try:
            classify_prompt = (
                "Given the user's message about a scientific hypothesis, determine if answering requires "
                "looking up new academic literature.\nIf yes, return a single optimised search query "
                "(e.g. 'NAD+ cellular autophagy mammalian lifespan').\nIf no, return 'NO'.\n\n"
                f"User message: '{new_message}'\nResponse:"
            )
            resp = litellm.completion(
                model=config.CHAT_MODEL,
                messages=[{"role": "user", "content": classify_prompt}],
                max_tokens=30,
            )
            search_query = resp.choices[0].message.content.strip()
            if search_query != "NO" and len(search_query) > 2:
                logger.info(f"Chat literature search triggered: '{search_query}'")
                results = academic_search(search_query, domain=academic_domain, max_results=3)
                fresh_literature = f"\n### Fresh Academic Search Results for '{search_query}':\n{results}\n"
        except Exception as e:
            logger.error(f"Chat literature search failed: {e}")

    # Primary model: Groq Llama 3.3 70B (with fallback to Cohere & Gemini)
    model_to_use = config.CHAT_MODEL
    logger.info(f"Chat routed to model: {model_to_use}")

    custom_inst = get_domain_specific_instructions(academic_domain)
    system_prompt = (
        "You are an expert scientific research assistant answering questions about a hypothesis evaluation.\n"
        f"Domain-Specific Guidelines: {custom_inst}\n"
        "Rules:\n"
        "1. Provide a detailed scientific answer (max 3 paragraphs or 4-5 bullet points).\n"
        "2. Do not reference database internals.\n"
        "3. Cite source papers where possible using [Title](URL) markdown links.\n"
        "4. Never fabricate links.\n"
        "5. Include statistical parameters (p-values, CI, effect sizes) where appropriate.\n\n"
        f"### Active Hypothesis Context:\n{active_context}\n"
        f"{semantic_context}"
        f"{fresh_literature}"
    )

    messages = [{"role": "system", "content": system_prompt}]
    history = row["conversation_history"] if row else []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": new_message})

    assistant_reply = ""
    try:
        response = litellm.completion(model=model_to_use, messages=messages, stream=True)
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                assistant_reply += content
                yield content
    except Exception as e:
        logger.error(f"Chat completion with '{model_to_use}' failed: {e}. Falling back to Cohere/Gemini.")
        try:
            fallback_response = litellm.completion(model="cohere/command-r-plus-08-2024", messages=messages, stream=True)
            for chunk in fallback_response:
                content = chunk.choices[0].delta.content
                if content:
                    assistant_reply += content
                    yield content
        except Exception as fb_err:
            logger.error(f"Fallback completion failed: {fb_err}")
            yield f"[Error: {fb_err}]"
            return

    # Persist updated chat history if DB is available
    if assistant_reply:
        updated_history = list(history) + [
            {"role": "user", "content": new_message},
            {"role": "assistant", "content": assistant_reply},
        ]
        try:
            with db_manager.get_connection() as conn:
                hypothesis_repo.update_conversation_history(conn, conversation_id, user_id, updated_history)
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to update chat history: {e}")
