"""Database access for hypothesis evaluations, always scoped to one user."""
import json
from typing import Optional


DETAIL_COLUMNS = """conversation_id, hypothesis, domain, core_claim, underlying_assumptions,
    causal_chain, supporting_evidence, counter_evidence, vulnerability_score,
    empirical_evidence_score, logical_consistency_score, confounder_vulnerability_score,
    methodological_feasibility_score, evaluation_summary, critical_weaknesses,
    proposed_validation_protocol, conversation_history, expected_effect_size,
    statistical_power_estimation, scientific_consensus_index, bias_vulnerability_score"""


def _detail(row: tuple, cache_hit: bool = False) -> dict:
    return {
        "conversation_id": str(row[0]), "raw_hypothesis": row[1], "academic_domain": row[2],
        "core_claim": row[3], "underlying_assumptions": row[4], "causal_chain": row[5],
        "supporting_evidence": row[6], "counter_evidence": row[7], "vulnerability_score": row[8],
        "empirical_evidence_score": row[9], "logical_consistency_score": row[10],
        "confounder_vulnerability_score": row[11], "methodological_feasibility_score": row[12],
        "evaluation_summary": row[13], "critical_weaknesses": row[14],
        "proposed_validation_protocol": row[15], "conversation_history": row[16] or [],
        "expected_effect_size": row[17], "statistical_power_estimation": row[18],
        "scientific_consensus_index": row[19], "bias_vulnerability_score": row[20],
        "is_cache_hit": cache_hit,
        "agent_logs": ["[Cache HIT] Retrieved a prior evaluation from your private history."] if cache_hit else ["[Loaded] Retrieved evaluation report."],
    }


def find_similar_hypothesis(conn, user_id: str, query_vector: list, distance_threshold: float = 0.1) -> Optional[dict]:
    with conn.cursor() as cur:
        cur.execute(f"""SELECT {DETAIL_COLUMNS}, embedding <=> %s::vector AS distance
            FROM hypothesis_evaluations WHERE user_id = %s
            ORDER BY distance ASC LIMIT 1""", (query_vector, user_id))
        row = cur.fetchone()
        return _detail(row, True) if row and row[21] < distance_threshold else None


def get_history(conn, user_id: str, limit: int = 20) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""SELECT conversation_id, hypothesis, domain, vulnerability_score, evaluation_summary
            FROM hypothesis_evaluations WHERE user_id = %s ORDER BY created_at DESC LIMIT %s""", (user_id, limit))
        return [{"conversation_id": str(r[0]), "hypothesis": r[1], "domain": r[2], "vulnerability_score": r[3], "evaluation_summary": r[4]} for r in cur.fetchall()]


def get_detail(conn, conversation_id: str, user_id: str) -> Optional[dict]:
    with conn.cursor() as cur:
        cur.execute(f"SELECT {DETAIL_COLUMNS} FROM hypothesis_evaluations WHERE conversation_id = %s AND user_id = %s", (conversation_id, user_id))
        row = cur.fetchone()
        return _detail(row) if row else None


def get_by_conversation_id(conn, conversation_id: str, user_id: str) -> Optional[dict]:
    return get_detail(conn, conversation_id, user_id)


def find_related_precedents(conn, user_id: str, query_vector: list, exclude_conversation_id: str, limit: int = 2) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute("""SELECT hypothesis, core_claim, evaluation_summary, critical_weaknesses,
            embedding <=> %s::vector AS distance FROM hypothesis_evaluations
            WHERE user_id = %s AND conversation_id != %s ORDER BY distance ASC LIMIT %s""", (query_vector, user_id, exclude_conversation_id, limit))
        return [{"hypothesis": r[0], "core_claim": r[1], "evaluation_summary": r[2], "critical_weaknesses": r[3], "distance": r[4]} for r in cur.fetchall()]


def upsert_evaluation(conn, user_id: str, data: dict, embedding: list) -> tuple[int, str]:
    conv_id = data.get("conversation_id")
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO hypothesis_evaluations (user_id, conversation_id, hypothesis, domain, core_claim, underlying_assumptions, causal_chain,
            supporting_evidence, counter_evidence, vulnerability_score, empirical_evidence_score, logical_consistency_score,
            confounder_vulnerability_score, methodological_feasibility_score, evaluation_summary, critical_weaknesses,
            proposed_validation_protocol, expected_effect_size, statistical_power_estimation, scientific_consensus_index,
            bias_vulnerability_score, conversation_history, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id, hypothesis) DO UPDATE SET domain = EXCLUDED.domain, core_claim = EXCLUDED.core_claim,
            underlying_assumptions = EXCLUDED.underlying_assumptions, causal_chain = EXCLUDED.causal_chain,
            supporting_evidence = EXCLUDED.supporting_evidence, counter_evidence = EXCLUDED.counter_evidence,
            vulnerability_score = EXCLUDED.vulnerability_score, empirical_evidence_score = EXCLUDED.empirical_evidence_score,
            logical_consistency_score = EXCLUDED.logical_consistency_score, confounder_vulnerability_score = EXCLUDED.confounder_vulnerability_score,
            methodological_feasibility_score = EXCLUDED.methodological_feasibility_score, evaluation_summary = EXCLUDED.evaluation_summary,
            critical_weaknesses = EXCLUDED.critical_weaknesses, proposed_validation_protocol = EXCLUDED.proposed_validation_protocol,
            expected_effect_size = EXCLUDED.expected_effect_size, statistical_power_estimation = EXCLUDED.statistical_power_estimation,
            scientific_consensus_index = EXCLUDED.scientific_consensus_index, bias_vulnerability_score = EXCLUDED.bias_vulnerability_score,
            embedding = EXCLUDED.embedding, created_at = NOW()
            RETURNING id, conversation_id""", (user_id, conv_id, data["raw_hypothesis"], data["academic_domain"], data["core_claim"], data["underlying_assumptions"], data["causal_chain"], data["supporting_evidence"], data["counter_evidence"], data["vulnerability_score"], data["empirical_evidence_score"], data["logical_consistency_score"], data["confounder_vulnerability_score"], data["methodological_feasibility_score"], data["evaluation_summary"], data["critical_weaknesses"], data["proposed_validation_protocol"], data["expected_effect_size"], data["statistical_power_estimation"], data["scientific_consensus_index"], data["bias_vulnerability_score"], json.dumps([]), embedding))
        row = cur.fetchone()
        return row[0], str(row[1])


def update_conversation_history(conn, conversation_id: str, user_id: str, history: list) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE hypothesis_evaluations SET conversation_history = %s::jsonb WHERE conversation_id = %s AND user_id = %s", (json.dumps(history), conversation_id, user_id))


def clear_all(conn, user_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("DELETE FROM hypothesis_evaluations WHERE user_id = %s", (user_id,))
