"""Task 38: the explicit redaction boundary between a QueryTrace/feedback
submission and an EvaluationEvent. Nothing downstream of this module should
ever see a raw QueryTrace or a raw session_id.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from evaluation.schemas import EvaluationEvent

_DEV_DEFAULT_SECRET = "dev-only-insecure-default-do-not-use-in-production"


def session_hash(session_id: Optional[str]) -> str:
    """HMAC-SHA256(server_secret, session_id), truncated for readability.
    Never store or log the raw session_id past this point. The secret must
    come from AUM_EVAL_HMAC_SECRET in any real deployment -- the fallback
    below exists only so local dev/tests don't require extra setup, and is
    deliberately named to make that obvious if it ever leaks into a log."""
    if not session_id:
        return ""
    secret = os.environ.get("AUM_EVAL_HMAC_SECRET", _DEV_DEFAULT_SECRET)
    return hmac.new(secret.encode("utf-8"), session_id.encode("utf-8"), hashlib.sha256).hexdigest()[:24]


def trace_to_event(
    trace,
    *,
    activity: str = "ask",
    session_id: Optional[str] = None,
    feedback: Optional[str] = None,
    feedback_comment: Optional[str] = None,
    include_text: bool = False,
) -> EvaluationEvent:
    """Build a sanitized EvaluationEvent from a QueryTrace.

    include_text=True retains query_text/answer_text -- use this ONLY for
    an explicit, consented feedback submission (the existing /api/feedback
    consent flow, Task 21/23). Never set it for passive/automatic telemetry.
    Even then, this never retains full_prompt, raw_llm_output, or uploaded
    document chunk text -- those stay out of every EvaluationEvent.
    """
    candidates = getattr(trace, "candidates", None) or []
    top_score = 0.0
    for c in candidates:
        score = max(
            getattr(c, "rerank_score", 0.0) or 0.0,
            getattr(c, "faiss_score", 0.0) or 0.0,
        )
        top_score = max(top_score, score)

    event = EvaluationEvent(
        event_id=str(uuid.uuid4()),
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        query_id=getattr(trace, "query_id", ""),
        session_hash=session_hash(session_id),
        activity=activity,
        source_mode=getattr(trace, "intent_type", ""),
        route_label=getattr(trace, "intent_type", ""),
        route_method=getattr(trace, "routing_path", ""),
        retrieval_top_score=top_score,
        retrieval_candidate_count=len(candidates),
        evidence_passed=getattr(trace, "threshold_passed", None),
        response_state=getattr(trace, "response_state", ""),
        latency_classify_ms=getattr(trace, "t_classify", 0.0),
        latency_retrieve_ms=getattr(trace, "t_faiss", 0.0),
        latency_rerank_ms=getattr(trace, "t_rerank", 0.0),
        latency_generate_ms=getattr(trace, "t_generate", 0.0),
        latency_total_ms=getattr(trace, "t_total", 0.0),
        feedback=feedback,
        feedback_comment=feedback_comment,
    )
    if include_text:
        event.query_text = getattr(trace, "query", None)
        event.answer_text = getattr(trace, "final_answer", None)
    return event
