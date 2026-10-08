"""Task 38: evaluation sanitizer, bounded trace cache, and sink behavior."""

import json
import tempfile
from pathlib import Path

from evaluation.sanitizer import session_hash, trace_to_event
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


def test_session_traces_is_bounded():
    assert SESSION_TRACES.maxlen == 500
    before = len(SESSION_TRACES)
    for i in range(510):
        SESSION_TRACES.append(QueryTrace(query_id=f"BND{i}"))
    assert len(SESSION_TRACES) == 500
    # oldest entries were evicted, not kept forever
    assert SESSION_TRACES[0].query_id != "BND0"
