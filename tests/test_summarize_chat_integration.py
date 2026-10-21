"""A natural-language "summarize this document" request typed into
ordinary chat (/api/ask, /api/ask/stream) must reach
AssistantService.summarize_document() via the Mistral-classified intent
check, not fall through to document_chat()'s per-question Q&A path.
Integration-level (real FastAPI app via TestClient); the classifier and
generation calls are mocked, same convention as
tests/test_quiz_chat_integration.py -- this project's standing rule that
default tests never load real model weights."""

import json

import numpy as np
from fastapi.testclient import TestClient

import server.api_server as api_server
from core.assistant_service import AssistantService
from core.orchestrator import set_assistant_service
from core.runtime_manager import RuntimeManager


class WordOverlapEmbedder:
    def encode(self, texts, **_kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)


class FakeTok:
    """generate_summary_basic calls llm_tok.encode() once to size its
    output-token budget -- see tests/test_summarize.py's own FakeTok."""
    def encode(self, text):
        return list(range(len(text)))


def setup_function(_fn):
    set_assistant_service(AssistantService(RuntimeManager(embedder=WordOverlapEmbedder(), llm_tok=FakeTok())))
    api_server._model_loaded = True


def teardown_function(_fn):
    api_server._model_loaded = False


def _upload_doc(client, session_id):
    upload = client.post(
        "/api/documents", data={"session_id": session_id},
        files={"file": ("notes.txt", b"The mitochondria is the powerhouse of the cell.", "text/plain")},
    )
    assert upload.status_code == 200, upload.text
    return upload.json()["document_id"]


def test_ask_summarize_request_reaches_summarize_document_not_quiz_or_qa():
    client = TestClient(api_server.app)
    document_id = _upload_doc(client, "summarize-chat-test")

    import core.assistant_service as assistant_service_module
    real_classify = assistant_service_module.classify_summarize_intent
    assistant_service_module.classify_summarize_intent = lambda question, *_a, **_kw: True

    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    summarize_module.generate_streaming = lambda prompt, *_a, **_kw: iter(["a plain-text summary of the document"])
    try:
        response = client.post("/api/ask", json={
            "question": "Can you summarize this document for me?",
            "topic": "auto", "session_id": "summarize-chat-test",
            "document_ids": [document_id],
        })
    finally:
        assistant_service_module.classify_summarize_intent = real_classify
        summarize_module.generate_streaming = real_generate_streaming

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "answer"
    assert body["topic_used"] == "summarize"
    assert body["quiz"] is None
    assert body["answer"] == "a plain-text summary of the document"
    assert body["query_id"], "query_id must be non-empty, or the feedback button stays disabled"


def test_ask_without_a_document_never_calls_the_summarize_classifier():
    """The classifier call costs a real LLM round trip -- it must not run
    at all when no document is attached, regardless of phrasing."""
    client = TestClient(api_server.app)

    import core.assistant_service as assistant_service_module
    real_classify = assistant_service_module.classify_summarize_intent
    calls = []
    assistant_service_module.classify_summarize_intent = lambda question, *_a, **_kw: calls.append(question) or True

    import core.assistant_service as service_mod
    real_generate_streaming = service_mod.generate_streaming
    service_mod.generate_streaming = lambda prompt, *_a, **_kw: iter(["it is not a summarize request"])
    try:
        response = client.post("/api/ask", json={
            "question": "summarize this for me",
            "topic": "general", "session_id": "summarize-chat-test-2",
        })
    finally:
        assistant_service_module.classify_summarize_intent = real_classify
        service_mod.generate_streaming = real_generate_streaming

    assert response.status_code == 200, response.text
    assert calls == []  # classifier never invoked -- no document was attached


def test_ask_stream_summarize_emits_answer_kind_in_done_event():
    client = TestClient(api_server.app)
    document_id = _upload_doc(client, "summarize-stream-test")

    import core.assistant_service as assistant_service_module
    real_classify = assistant_service_module.classify_summarize_intent
    assistant_service_module.classify_summarize_intent = lambda question, *_a, **_kw: True

    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    summarize_module.generate_streaming = lambda prompt, *_a, **_kw: iter(["streamed summary text"])
    try:
        with client.stream("POST", "/api/ask/stream", json={
            "question": "give me a summary of the pdf",
            "topic": "auto", "session_id": "summarize-stream-test",
            "document_ids": [document_id],
        }) as response:
            assert response.status_code == 200
            raw = "".join(response.iter_text())
    finally:
        assistant_service_module.classify_summarize_intent = real_classify
        summarize_module.generate_streaming = real_generate_streaming

    assert "event: done" in raw
    done_line = next(line for line in raw.splitlines() if line.startswith("data:") and '"kind"' in line)
    payload = json.loads(done_line[len("data:"):].strip())
    assert payload["kind"] == "answer"
    assert payload["topic_used"] == "summarize"
    assert payload["query_id"]


def test_ask_stream_summarize_delivers_multiple_real_deltas_not_one_blocking_chunk():
    """The actual bug report: summarize used to go silent for the whole
    generation and deliver everything in a single SSE event right
    before "done". This proves real incremental streaming through the
    full HTTP/SSE stack -- a multi-step fake generation (simulating
    token-by-token growth) must arrive as multiple separate delta
    events, not one, before "done"."""
    client = TestClient(api_server.app)
    document_id = _upload_doc(client, "summarize-multi-delta-test")

    import core.assistant_service as assistant_service_module
    real_classify = assistant_service_module.classify_summarize_intent
    assistant_service_module.classify_summarize_intent = lambda question, *_a, **_kw: True

    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    summarize_module.generate_streaming = lambda prompt, *_a, **_kw: iter(
        ["The", "The cell", "The cell is", "The cell is the basic unit of life."],
    )
    try:
        with client.stream("POST", "/api/ask/stream", json={
            "question": "summarize this document",
            "topic": "auto", "session_id": "summarize-multi-delta-test",
            "document_ids": [document_id],
        }) as response:
            assert response.status_code == 200
            raw = "".join(response.iter_text())
    finally:
        assistant_service_module.classify_summarize_intent = real_classify
        summarize_module.generate_streaming = real_generate_streaming

    events = raw.strip().split("\n\n")
    # Excludes "done" (the completion payload) and "meta" (the early
    # query_id announcement added for the Stop feature -- its own
    # behavior is covered in tests/test_stop_generation.py) -- neither
    # is a text delta.
    delta_events = [
        e for e in events
        if not e.startswith("event: done") and not e.startswith("event: meta")
    ]
    # One SSE event per incremental yield (minus the "done" event) -- if
    # this were still the old blocking behavior, there would be exactly
    # one delta event carrying the whole final answer, not four.
    assert len(delta_events) == 4
    reassembled = "".join(
        e[len("data: "):] for e in delta_events if e.startswith("data: ")
    )
    assert reassembled == "The cell is the basic unit of life."
