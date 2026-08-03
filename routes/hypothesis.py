"""
Hypothesis routes — thin HTTP handlers only.
All logic is delegated to HypothesisService.
"""
import json
import logging
from typing import List
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from services import hypothesis_service
from auth import get_current_user_id

logger = logging.getLogger("HypothesisRouter")

router = APIRouter(prefix="/api/hypothesis", tags=["Hypothesis Testing"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

from typing import List, Optional

class HypothesisRequest(BaseModel):
    hypothesis: str
    domain: str
    conversation_id: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ConverseRequest(BaseModel):
    conversation_id: UUID
    new_message: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/evaluate")
def evaluate_hypothesis(req: HypothesisRequest, user_id: str = Depends(get_current_user_id)):
    """
    Evaluate a scientific hypothesis via a multi-agent stress-test.
    Streams Server-Sent Events (SSE) with progress updates and a final JSON result.
    """
    def sse_generator():
        for event in hypothesis_service.stream_evaluation(
            req.hypothesis, req.domain, user_id, req.conversation_id
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")


@router.get("/history")
def get_hypothesis_history(user_id: str = Depends(get_current_user_id)):
    """Return the latest 20 hypothesis evaluations (summary only)."""
    try:
        return hypothesis_service.get_history(user_id)
    except Exception as e:
        logger.error(f"History fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detail/{eval_id}")
def get_hypothesis_detail(eval_id: UUID, user_id: str = Depends(get_current_user_id)):
    """Return full evaluation details for a specific hypothesis ID."""
    try:
        detail = hypothesis_service.get_detail(str(eval_id), user_id)
        if detail is None:
            raise HTTPException(status_code=404, detail="Evaluation report not found.")
        return detail
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detail fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear")
def clear_hypothesis_history(user_id: str = Depends(get_current_user_id)):
    """Truncate all hypothesis evaluation records."""
    try:
        hypothesis_service.clear_history(user_id)
        return {"status": "cleared"}
    except Exception as e:
        logger.error(f"Clear failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/converse")
def converse_about_hypothesis(req: ConverseRequest, user_id: str = Depends(get_current_user_id)):
    """
    Stream a token-by-token conversational Q&A response about the active hypothesis.
    Uses Server-Sent Events.
    """
    def sse_generator():
        for token in hypothesis_service.stream_conversation(str(req.conversation_id), req.new_message, user_id):
            yield f"data: {json.dumps({'text': token})}\n\n"

    return StreamingResponse(sse_generator(), media_type="text/event-stream")
