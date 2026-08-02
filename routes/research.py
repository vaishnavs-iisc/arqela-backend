"""
Research routes — chat, history, and LTM endpoints.
Extracted from app.py; all logic delegated to ResearchService and CacheService.
"""
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from services import research_service, cache_service
from database import db_manager
from repositories import reports_repo

logger = logging.getLogger("ResearchRouter")

router = APIRouter(tags=["Research Agent"])


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/chat")
def run_chat(req: ChatRequest):
    """Run the legacy research agent graph and return the final report."""
    try:
        return research_service.run_research(req.query, req.session_id)
    except Exception as e:
        logger.error(f"Research agent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{session_id}")
def get_session_history(session_id: str):
    """Return the Redis STM conversation history for a session."""
    return cache_service.stm_get(session_id)


@router.post("/clear/{session_id}")
def clear_session_history(session_id: str):
    """Delete the Redis STM conversation history for a session."""
    cache_service.stm_delete(session_id)
    return {"status": "cleared"}


@router.get("/ltm")
def get_ltm_reports():
    """Return a summary of all stored long-term memory research reports."""
    try:
        with db_manager.get_connection() as conn:
            return reports_repo.get_all_reports(conn)
    except Exception as e:
        logger.error(f"LTM fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
