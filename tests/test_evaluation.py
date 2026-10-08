"""Task 38: evaluation sanitizer, bounded trace cache, and sink behavior.
Task 45 additions: EvaluationService.record_event()'s sink-failure
isolation, and the same guarantee for /api/feedback's missing-trace
fallback path."""

import json
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from evaluation.sanitizer import session_hash, trace_to_event
from evaluation.schemas import EvaluationEvent
from evaluation.service import EvaluationService
from evaluation.sinks.local_dev import LocalDevSink
from pipeline.classifier import QueryTrace, RetrievalCandidate, SESSION_TRACES


def test_session_hash_is_deterministic_and_not_reversible():
    a = session_hash("session-123")
    b = session_hash("session-123")
    c = session_hash("session-456")
    assert a == b
    assert a != c
    assert "session-123" not in a


def test_session_hash_empty_for_missing_session():
    assert session_hash(None) == ""
    assert session_hash("") == ""


def test_trace_to_event_strips_full_prompt_and_raw_output_by_default():
    trace = QueryTrace(
        query_id="Q0001", query="What is the refund policy?", tab="document",
        intent_type="document", routing_path="explicit document selection",
        full_prompt="<the real prompt with private document text inside it>",
        raw_llm_output="<raw model output>",
        final_answer="The refund policy is...",
        response_state="paragraph", threshold_passed=True,
        candidates=[RetrievalCandidate(idx=0, title="doc.pdf", mentor="p.1", year="", department="Uploaded document", faiss_score=0.71)],
    )
    event = trace_to_event(trace, session_id="real-session-id")
    assert event.query_text is None
    assert event.answer_text is None
    assert "private document text" not in str(vars(event))
    assert event.retrieval_top_score == 0.71
    assert event.retrieval_candidate_count == 1
    assert event.session_hash != "real-session-id"


def test_trace_to_event_include_text_retains_only_query_and_answer():
    trace = QueryTrace(
        query_id="Q0002", query="Are candles allowed?", tab="housing",
        intent_type="housing", full_prompt="<full private prompt>",
        final_answer="No, candles are not allowed.",
    )
    event = trace_to_event(trace, session_id="s1", include_text=True, feedback="down", feedback_comment="wrong")
    assert event.query_text == "Are candles allowed?"
    assert event.answer_text == "No, candles are not allowed."
    assert event.feedback == "down"
    assert event.feedback_comment == "wrong"
    # full_prompt must never leak through even with include_text=True
    assert "full private prompt" not in (event.query_text or "") + (event.answer_text or "")


def test_evaluation_service_writes_sanitized_event_to_local_dev_sink():
    trace = QueryTrace(query_id="Q0003", query="test query", tab="cos", intent_type="cos", final_answer="test answer")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "evaluation_events.jsonl"
        service = EvaluationService(sinks=[LocalDevSink(path)])
        service.record_trace(trace, session_id="sess-abc", feedback="up")
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["query_id"] == "Q0003"
        assert record["feedback"] == "up"
        assert "full_prompt" not in record
        assert "sess-abc" not in json.dumps(record)


def test_evaluation_service_tolerates_none_trace():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        service = EvaluationService(sinks=[LocalDevSink(path)])
        service.record_trace(None, session_id="s1")  # must not raise
        assert not path.exists()


def test_record_event_isolates_a_failing_sink_from_a_working_one():
    class BrokenSink:
        def record(self, event):
            raise RuntimeError("simulated sink outage")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "events.jsonl"
        working = LocalDevSink(path)
        service = EvaluationService(sinks=[BrokenSink(), working])
        # Must not raise, and the working sink must still receive the event
        # even though the sink ordered before it failed.
        service.record_event(EvaluationEvent(
            event_id="e1", timestamp_utc="2026-10-08T00:00:00+00:00",
            query_id="Q9001", session_hash="", activity="ask", source_mode="",
        ))
        lines = path.read_text().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["query_id"] == "Q9001"


def test_feedback_endpoint_missing_trace_survives_a_broken_sink_with_utc_timestamp():
    """Task 45 (external review 2026-10-08): this fallback path used to call
    sink.record() directly in its own unprotected loop, bypassing
    EvaluationService's isolation -- a broken sink turned a feedback
    submission into an unhandled exception instead of a 200 with the
    failure merely logged. It also used a naive (non-UTC) timestamp."""
    import server.api_server as api_server
    from core.assistant_service import AssistantService
    from core.orchestrator import set_assistant_service
    from core.runtime_manager import RuntimeManager

    class BrokenSink:
        def record(self, event):
            raise RuntimeError("simulated sink outage")

    class CapturingSink:
        def __init__(self):
            self.events = []

        def record(self, event):
            self.events.append(event)

    set_assistant_service(AssistantService(RuntimeManager()))
    capturing = CapturingSink()
    original_sinks = api_server._evaluation_service.sinks
    api_server._evaluation_service.sinks = [BrokenSink(), capturing]
    try:
        client = TestClient(api_server.app)
        response = client.post("/api/feedback", json={
            "reaction": "down", "scope": "single",
            # query_id deliberately not in SESSION_TRACES -- forces the
            # missing-trace fallback path.
            "query_id": "Q-definitely-not-cached-9999",
            "session_id": "test-session",
        })
    finally:
        api_server._evaluation_service.sinks = original_sinks

    assert response.status_code == 200, response.text
    assert len(capturing.events) == 1
    event = capturing.events[0]
    assert event.query_id == "Q-definitely-not-cached-9999"
    assert event.feedback == "down"
    # A naive isoformat() string has no "+" offset and doesn't end in "Z".
    assert "+" in event.timestamp_utc or event.timestamp_utc.endswith("Z")


def test_session_traces_is_bounded():
    assert SESSION_TRACES.maxlen == 500
    before = len(SESSION_TRACES)
    for i in range(510):
        SESSION_TRACES.append(QueryTrace(query_id=f"BND{i}"))
    assert len(SESSION_TRACES) == 500
    # oldest entries were evicted, not kept forever
    assert SESSION_TRACES[0].query_id != "BND0"
