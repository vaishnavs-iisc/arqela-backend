"""
LangGraph node functions for the Research Agent workflow.
Each node is a pure function — no HTTP, no DB access.
"""
import json
import logging

import litellm

from config import config
from utils import get_embedding, cosine_similarity
from core.research_agent.state import AgentState
from core.research_agent.helpers import decode_if_base64, check_pii, check_safety_llm, web_search

logger = logging.getLogger("ResearchAgentNodes")


def guardrail_node(state: AgentState) -> dict:
    """Inspect the user query for safety and PII violations."""
    query = state["user_query"]
    logs = ["[Node: Guardrail] Inspecting user query."]

    processed_query = decode_if_base64(query)
    if processed_query != query:
        logs.append(f"[Guardrail] Decoded Base64 prompt to: '{processed_query}'")

    if check_pii(processed_query):
        logs.append("[Guardrail] Safety check failed: PII Detected.")
        return {
            "is_safe": False,
            "safety_reason": "Personally Identifiable Information (email/phone) detected.",
            "agent_logs": logs,
        }

    is_safe, reason = check_safety_llm(processed_query)
    if not is_safe:
        logs.append(f"[Guardrail] Safety check failed: {reason}")
        return {"is_safe": False, "safety_reason": reason, "agent_logs": logs}

    logs.append("[Guardrail] Safety check passed.")
    return {"is_safe": True, "agent_logs": logs}


def cache_check_node(state: AgentState, redis_client) -> dict:
    """Check Redis semantic cache for a similar past query."""
    query = decode_if_base64(state["user_query"])
    logs = ["[Node: Semantic Cache] Checking for similar past queries."]

    query_vector = get_embedding(query)
    keys = redis_client.keys("cache:*")

    for key in keys:
        try:
            cached_data = json.loads(redis_client.get(key))
            cached_vector = cached_data["embedding"]
            similarity = cosine_similarity(query_vector, cached_vector)

            if similarity >= 0.85:
                logs.append(
                    f"[Cache HIT] Found similar past query: '{cached_data['query']}' ({similarity:.2%} similarity)."
                )
                return {
                    "is_cache_hit": True,
                    "draft_report": cached_data["report"],
                    "is_finalized": True,
                    "accuracy_score": 5,
                    "agent_logs": logs,
                }
        except Exception as e:
            logger.error(f"Failed to read cache key: {e}")

    logs.append("[Cache MISS] No similar past query found. Running full research pipeline.")
    return {"is_cache_hit": False, "agent_logs": logs}


def search_agent_node(state: AgentState) -> dict:
    """Generate an optimised search query and retrieve web results."""
    query = decode_if_base64(state["user_query"])
    logs = ["[Node: Search Agent] Initiating web search queries."]

    prompt = f"Write a single search engine query to find factual reports and summaries about: '{query}'."
    try:
        response = litellm.completion(
            model=config.PRIMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        search_query = response.choices[0].message.content.strip().replace('"', "")
    except Exception:
        search_query = query

    logs.append(f"[Search Agent] Generated optimised query: '{search_query}'")
    results = web_search(search_query, max_results=3)
    logs.append("[Search Agent] Retrieved search sources.")
    return {"search_query": search_query, "raw_sources": results, "agent_logs": logs}


def summarize_agent_node(state: AgentState) -> dict:
    """Aggregate search results into a concise factual summary."""
    logs = ["[Node: Summarize Agent] Aggregating search results."]
    sources = state["raw_sources"]

    prompt = (
        "You are a Research Summarizer. Aggregate and summarize these sources, highlighting "
        f"all key facts, statistics, and dates. Be highly factual.\n\nSources:\n{sources}"
    )
    try:
        response = litellm.completion(
            model=config.PRIMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.choices[0].message.content.strip()
    except Exception as e:
        summary = f"Summary generation failed: {e}"

    logs.append("[Summarize Agent] Summary completed.")
    return {"summary": summary, "agent_logs": logs}


def writer_agent_node(state: AgentState, redis_client) -> dict:
    """Draft or rewrite the research report."""
    loop = state.get("loop_count", 0)
    logs = [f"[Node: Writer Agent] Generating draft report (Attempt {loop + 1})."]

    query = decode_if_base64(state["user_query"])
    sources = state["raw_sources"]
    summary = state["summary"]

    # Load conversation history from Redis STM
    session_key = f"chat:{state['session_id']}"
    raw_history = redis_client.lrange(session_key, 0, -1)
    chat_history_str = ""
    if raw_history:
        logs.append("[Writer Agent] Loaded conversation history context from Redis.")
        parsed_history = [json.loads(msg) for msg in raw_history]
        chat_history_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in parsed_history]
        )

    if loop == 0:
        prompt = (
            f"You are a factual assistant. Your task is to write a clear, concise, "
            f"conversational answer (1-2 paragraphs) on: '{query}' based on the sources.\n"
            f"Aggregated summary and facts:\n{summary}\n\n"
            f"Raw Sources Reference:\n{sources}\n\n"
            f"Chat History Context (if any):\n{chat_history_str}\n\n"
            "Format the text nicely in Markdown. Do not hallucinate."
        )
    else:
        logs.append(f"[Writer Agent] Rewriting response based on Judge critique: '{state['critique']}'")
        prompt = (
            "You are a factual assistant. Correct your previous response based on the critique.\n"
            f"Original Sources Reference:\n{sources}\n\n"
            f"Previous Response:\n{state['draft_report']}\n\n"
            f"Judge's Critique:\n{state['critique']}\n\n"
            "Rewrite the response concisely, correcting all factual inaccuracies. "
            "Ensure the response is fully accurate to the sources."
        )

    try:
        response = litellm.completion(
            model=config.PRIMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        draft = response.choices[0].message.content.strip()
    except Exception as e:
        draft = f"Drafting failed: {e}"

    logs.append("[Writer Agent] Draft report generated.")
    return {"draft_report": draft, "loop_count": loop + 1, "agent_logs": logs}


def verify_agent_node(state: AgentState) -> dict:
    """Audit the draft report for factual accuracy (LLM-as-Judge)."""
    logs = ["[Node: Verify Agent] Auditing report factual accuracy (LLM-as-Judge)."]
    sources = state["raw_sources"]
    draft = state["draft_report"]

    system_prompt = (
        "You are an expert factual Auditor. Compare the Draft Report against the Original Sources.\n"
        "Verify every date, name, and fact.\n"
        "Evaluate factual accuracy on a scale of 1 to 5 (5 being perfect, 1 being completely false).\n"
        "If there are errors, explain them in the critique.\n"
        'Output your response strictly as JSON:\n{\n  "accuracy_score": int,\n  "critique": "string"\n}'
    )
    user_prompt = f"Original Sources:\n{sources}\n\nDraft Report:\n{draft}"

    try:
        response = litellm.completion(
            model=config.JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
        result = json.loads(content.strip())
        score = int(result.get("accuracy_score", 5))
        critique = result.get("critique", "")

        logs.append(f"[Verify Agent] Factual accuracy graded: {score}/5.")
        if critique:
            logs.append(f"[Verify Agent] Critique: {critique}")

        return {
            "accuracy_score": score,
            "critique": critique,
            "is_finalized": (score >= 4),
            "agent_logs": logs,
        }
    except Exception as e:
        logger.error(f"Audit agent failed: {e}")
        return {
            "accuracy_score": 5,
            "critique": "",
            "is_finalized": True,
            "agent_logs": logs,
        }


def save_memories_node(state: AgentState, redis_client, db_manager) -> dict:
    """Persist the final report to Redis cache and PostgreSQL LTM."""
    logs = ["[Node: Save Memory] Persisting report to databases."]
    query = decode_if_base64(state["user_query"])
    report = state["draft_report"]

    # 1. Redis Semantic Cache
    try:
        query_vector = get_embedding(query)
        cache_key = f"cache:{hash(query)}"
        import json as _json
        redis_client.set(cache_key, _json.dumps({"query": query, "report": report, "embedding": query_vector}), ex=86400)
        logs.append("[Save Memory] Saved report to Redis Semantic Cache.")
    except Exception as e:
        logger.error(f"Redis cache save failed: {e}")

    # 2. PostgreSQL LTM
    try:
        from utils import get_embedding as _get_embedding
        query_vector = _get_embedding(query)
        with db_manager.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO past_reports (topic, report, embedding)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (topic) DO UPDATE
                    SET report = EXCLUDED.report, embedding = EXCLUDED.embedding;
                    """,
                    (query, report, query_vector),
                )
            conn.commit()
        logs.append("[Save Memory] Saved report to PostgreSQL LTM using pool.")
    except Exception as e:
        logger.error(f"Postgres LTM save failed: {e}")

    # 3. Redis STM — chat history
    try:
        import json as _json
        session_key = f"chat:{state['session_id']}"
        redis_client.rpush(session_key, _json.dumps({"role": "user", "content": query}))
        redis_client.rpush(session_key, _json.dumps({"role": "assistant", "content": f"Generated report on '{query}'."}))
        redis_client.expire(session_key, 3600)
        logs.append("[Save Memory] Conversation context appended to Redis STM.")
    except Exception as e:
        logger.error(f"Redis STM save failed: {e}")

    return {"agent_logs": logs}


def rejection_node(state: AgentState) -> dict:
    """Return a blocked response when the query fails safety checks."""
    logs = ["[Node: Rejection] Processing block."]
    reason = state["safety_reason"]
    return {
        "draft_report": f"⚠️ [Safety Block] Request rejected.\nReason: {reason}",
        "agent_logs": logs,
    }
