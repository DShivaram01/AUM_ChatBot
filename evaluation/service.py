"""Task 38: the boundary application code emits through. Nothing outside
this module decides which sink(s) an event reaches -- adding a remote sink
(Task 39) means changing how EvaluationService is constructed, not touching
any call site that emits an event.
"""

from __future__ import annotations

from typing import Optional

from evaluation.sanitizer import trace_to_event
from evaluation.sinks.base import EvaluationSink
from pipeline.memory import logger


class EvaluationService:
    def __init__(self, sinks: list[EvaluationSink]):
        self.sinks = sinks

    def record_trace(
        self,
        trace,
        *,
        activity: str = "ask",
        session_id: Optional[str] = None,
        feedback: Optional[str] = None,
        feedback_comment: Optional[str] = None,
        include_text: bool = False,
    ) -> None:
        if trace is None:
            return
        event = trace_to_event(
            trace, activity=activity, session_id=session_id,
            feedback=feedback, feedback_comment=feedback_comment,
            include_text=include_text,
        )
        for sink in self.sinks:
            try:
                sink.record(event)
            except Exception:
                # A sink failure must never break the request that triggered
                # it (e.g. a feedback submission the user is waiting on).
                logger.exception(f"[EvaluationService] sink {sink!r} failed to record event")
