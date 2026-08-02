"""
Reports repository — SQL for past_reports (LTM for the research agent).
"""
import logging

logger = logging.getLogger("ReportsRepo")


def upsert_report(conn, topic: str, report: str, embedding: list) -> None:
    """Insert or update a research report in the past_reports LTM table."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO past_reports (topic, report, embedding)
            VALUES (%s, %s, %s)
            ON CONFLICT (topic) DO UPDATE
            SET report = EXCLUDED.report, embedding = EXCLUDED.embedding;
            """,
            (topic, report, embedding),
        )


def get_all_reports(conn) -> list[dict]:
    """Return a summary list of all stored reports (topic + character count)."""
    with conn.cursor() as cur:
        cur.execute("SELECT topic, LENGTH(report) FROM past_reports ORDER BY id DESC;")
        return [{"topic": row[0], "length": row[1]} for row in cur.fetchall()]
