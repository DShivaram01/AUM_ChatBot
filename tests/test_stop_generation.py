"""Stop-mid-generation: a per-query_id flag checked by a StoppingCriteria
(transformers' own documented way to interrupt model.generate() -- you
can't safely kill a thread mid-CUDA-op), plus the per-iteration checks
quiz's two generators need since they make several SEPARATE generate()
calls per logical answer, not one."""

import json

import numpy as np
import torch
from fastapi.testclient import TestClient

import server.api_server as api_server
from core.assistant_service import AssistantService
from core.orchestrator import set_assistant_service
from core.runtime_manager import RuntimeManager
from pipeline.answer import (
    _StopRequested, clear_stop, is_stop_requested, request_stop,
)


def test_is_stop_requested_false_until_asked():
    assert is_stop_requested("SUM0099") is False


def test_request_stop_marks_the_exact_query_id():
    request_stop("SUM0001")
    try:
        assert is_stop_requested("SUM0001") is True
    finally:
        clear_stop("SUM0001")


def test_request_stop_also_matches_a_sub_call_id():
    # generate_streaming() is called with sub-ids like "SUM0001-summary"
    # or "QZ0001-basic-q3" -- a stop request names the bare base id the
    # client was shown, never a sub-id it never saw, so this must match
    # by prefix, not exact equality.
    request_stop("QZ0001")
    try:
        assert is_stop_requested("QZ0001-basic-q3") is True
        assert is_stop_requested("QZ0001-q2a1") is True
    finally:
        clear_stop("QZ0001")


def test_request_stop_does_not_match_an_unrelated_id_with_a_shared_prefix():
    # "QZ0001" must not accidentally match "QZ00010" (a different,
    # later query) just because it shares a string prefix -- the match
    # requires either exact equality or a "-" boundary.
    request_stop("QZ0001")
    try:
        assert is_stop_requested("QZ00010") is False
        assert is_stop_requested("QZ00010-basic-q1") is False
    finally:
        clear_stop("QZ0001")


def test_clear_stop_removes_the_flag():
    request_stop("DOC0001")
    clear_stop("DOC0001")
    assert is_stop_requested("DOC0001") is False


def test_clear_stop_on_an_id_never_requested_is_a_safe_no_op():
    clear_stop("GP9999")  # must not raise


def test_stop_requested_criteria_delegates_to_is_stop_requested():
    """_StopRequested is what's actually passed to model.generate() as a
    StoppingCriteria -- it's checked after every generated token."""
    criteria = _StopRequested("SUM0002")
    dummy_ids = torch.tensor([[1, 2, 3]])
    dummy_scores = torch.zeros(1, 10)

    assert criteria(dummy_ids, dummy_scores) is False
    request_stop("SUM0002")
    try:
        assert criteria(dummy_ids, dummy_scores) is True
    finally:
        clear_stop("SUM0002")


# ---------- /api/ask/stop and the "meta" event that makes it targetable ----------

class WordOverlapEmbedder:
    def encode(self, texts, **_kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)


class FakeTok:
    def encode(self, text):
        return list(range(len(text)))


def setup_function(_fn):
    set_assistant_service(AssistantService(RuntimeManager(embedder=WordOverlapEmbedder(), llm_tok=FakeTok())))
    api_server._model_loaded = True


def teardown_function(_fn):
    api_server._model_loaded = False


def test_ask_stop_endpoint_marks_the_query_id():
    client = TestClient(api_server.app)
    assert is_stop_requested("SUM0050") is False
    response = client.post("/api/ask/stop", json={"query_id": "SUM0050"})
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    try:
        assert is_stop_requested("SUM0050") is True
    finally:
        clear_stop("SUM0050")


def test_ask_stop_on_an_unknown_query_id_is_not_an_error():
    """A stop can always race the response finishing first -- this must
    stay a harmless no-op, never a 404/500 the client has to handle."""
    client = TestClient(api_server.app)
    response = client.post("/api/ask/stop", json={"query_id": "SUM9999-already-done"})
    assert response.status_code == 200


def test_ask_stream_summarize_announces_query_id_in_a_meta_event_before_done():
    """The whole point of "meta": the client must learn the real
    query_id WHILE the answer is still streaming, not only once it's
    already finished (data.query_id from the "done" event, as it worked
    before this feature) -- otherwise there is nothing left to stop by
    the time a Stop button could ever send it."""
    client = TestClient(api_server.app)
    upload = client.post(
        "/api/documents", data={"session_id": "stop-meta-test"},
        files={"file": ("notes.txt", b"The mitochondria is the powerhouse of the cell.", "text/plain")},
    )
    document_id = upload.json()["document_id"]

    import core.assistant_service as assistant_service_module
    real_classify = assistant_service_module.classify_summarize_intent
    assistant_service_module.classify_summarize_intent = lambda question, *_a, **_kw: True

    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    summarize_module.generate_streaming = lambda prompt, *_a, **_kw: iter(["partial", "partial text."])
    try:
        with client.stream("POST", "/api/ask/stream", json={
            "question": "summarize this document",
            "topic": "auto", "session_id": "stop-meta-test",
            "document_ids": [document_id],
        }) as response:
            raw = "".join(response.iter_text())
    finally:
        assistant_service_module.classify_summarize_intent = real_classify
        summarize_module.generate_streaming = real_generate_streaming

    events = raw.strip().split("\n\n")
    meta_index = next(i for i, e in enumerate(events) if e.startswith("event: meta"))
    done_index = next(i for i, e in enumerate(events) if e.startswith("event: done"))
    assert meta_index < done_index  # announced BEFORE the stream finishes, not just at the end

    meta_payload = json.loads(events[meta_index].split("data: ", 1)[1])
    done_payload = json.loads(events[done_index].split("data: ", 1)[1])
    assert meta_payload["query_id"]
    assert meta_payload["query_id"] == done_payload["query_id"]
