"""Task 38: the narrow, explicit evaluation-event shape.

EvaluationEvent is NOT a serialization of QueryTrace. It is a deliberate
allowlist -- see workspace.md Entry 055/Task 38. No full prompt, no raw
model output, no uploaded-document text, no raw session identifier by
default. query_text/answer_text exist only for the explicit, consented
/api/feedback flow (Task 21/23's existing UI) -- never for passive
telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvaluationEvent:
    event_id: str
    timestamp_utc: str
    query_id: str
    session_hash: str

    activity: str              # "ask" | "quiz"
    source_mode: str           # cos / housing / document / general / general_aum / out_of_scope / open_ended_disabled / pretrained
    route_label: str = ""      # the resolved topic/intent label
    route_method: str = ""     # routing_path: person-override / Mistral-router label / fallback / explicit

    retrieval_top_score: float = 0.0
    retrieval_candidate_count: int = 0
    evidence_passed: Optional[bool] = None
    response_state: str = ""   # "paragraph" | "list" | "not_found" | "non_retrieval" | ...

    latency_classify_ms: float = 0.0
    latency_retrieve_ms: float = 0.0
    latency_rerank_ms: float = 0.0
    latency_generate_ms: float = 0.0
    latency_total_ms: float = 0.0

    feedback: Optional[str] = None            # "up" | "down" | None
    feedback_comment: Optional[str] = None
    error_code: Optional[str] = None

    # Only populated when the user explicitly submitted feedback for THIS
    # query (the consented flow) -- never for passive/automatic telemetry.
    query_text: Optional[str] = None
    answer_text: Optional[str] = None
