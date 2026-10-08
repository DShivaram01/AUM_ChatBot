"""Task 38: the boundary application code emits through. Nothing outside
this module decides which sink(s) an event reaches -- adding a remote sink
(Task 39) means changing how EvaluationService is constructed, not touching
any call site that emits an event.
"""

from __future__ import annotations

from typing import Optional

from evaluation.sanitizer import trace_to_event
from evaluation.schemas import EvaluationEvent
from evaluation.sinks.base import EvaluationSink
from pipeline.memory import logger


class EvaluationService:
    def __init__(self, sinks: list[EvaluationSink]):
        self.sinks = sinks

    def record_event(self, event: EvaluationEvent) -> None:
        """Fan an already-built event out to every sink, isolating each
        sink's own failures so one broken sink can never turn the request
        that triggered this call (e.g. a feedback submission the user is
        waiting on) into an unhandled exception. record_trace() below is
        the normal entry point (trace -> sanitize -> this); this one also
        exists directly for the rare case where there is no QueryTrace to
        sanitize at all (Task 45, external review 2026-10-08: the
        missing-trace feedback fallback in server/api_server.py used to
        call sink.record() in its own unprotected loop, bypassing this
        exact isolation)."""
        for sink in self.sinks:
            try:
                sink.record(event)
            except Exception:
                logger.exception(f"[EvaluationService] sink {sink!r} failed to record event")

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
        self.record_event(event)
