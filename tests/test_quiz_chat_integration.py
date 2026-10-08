"""Task 53 Phase 1-2: a natural-language quiz request typed into ordinary
chat (/api/ask, /api/ask/stream) must reach the same AssistantService.quiz()
pipeline the Quiz button already uses, not an unstructured free-text
answer. Integration-level (real FastAPI app via TestClient), with the LLM
mocked exactly like tests/test_quiz.py already does -- this project's
standing rule that default tests never load real model weights."""

import json

import numpy as np
from fastapi.testclient import TestClient

import server.api_server as api_server
from core.assistant_service import AssistantService
from core.orchestrator import set_assistant_service
from core.runtime_manager import RuntimeManager


class WordOverlapEmbedder:
    def encode(self, texts, **_kwargs):
        values = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            values[row, row % 4] = 1.0
        return values


def _fake_single_question_response(number):
    return json.dumps({
        "question": f"Question {number}?",
        "options": ["A", "B", "C", "D"],
        "correct_index": number % 4,
        "explanation": "because",
        "evidence_refs": [],
    })


def setup_function(_fn):
    set_assistant_service(AssistantService(RuntimeManager(embedder=WordOverlapEmbedder())))
    api_server._model_loaded = True


def teardown_function(_fn):
    api_server._model_loaded = False


def test_ask_detects_quiz_intent_and_returns_structured_quiz_not_prose():
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    call_count = {"n": 0}

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
        call_count["n"] += 1
        yield _fake_single_question_response(call_count["n"])

    quiz_module.generate_streaming = fake_generate_streaming
    try:
        client = TestClient(api_server.app)
        response = client.post("/api/ask", json={
            "question": "Create 3 MCQs on computational biology",
            "topic": "general",  # Open-ended toggle on -- required for a pretrained fallback
            "session_id": "quiz-chat-test",
        })
    finally:
        quiz_module.generate_streaming = real_generate_streaming

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "quiz"
    assert body["topic_used"] == "quiz"
    assert body["quiz"] is not None
    assert len(body["quiz"]["questions"]) == 3
    assert call_count["n"] == 3  # one model call per question, same as the button path


def test_ask_quiz_request_without_a_source_asks_rather_than_guessing():
    client = TestClient(api_server.app)
    response = client.post("/api/ask", json={
        "question": "Create 5 MCQs on computational biology",
        "topic": "auto",  # Open-ended NOT enabled, no document attached, no housing hint
        "session_id": "quiz-chat-test",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "answer"
    assert body["topic_used"] == "quiz_unavailable"
    assert "Open-ended" in body["answer"]


def test_ask_ordinary_question_is_not_hijacked_into_a_quiz():
    # Not a quiz request -- falls through to the real general_chat() path,
    # which needs generate_streaming mocked too (patched where
    # core/assistant_service.py actually imported it, same convention
    # tests/test_quiz.py already documents: patching pipeline.answer's
    # copy would not intercept the call this module actually makes).
    import core.assistant_service as assistant_service_module
    real_generate_streaming = assistant_service_module.generate_streaming

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
        yield "it is not a quiz"

    assistant_service_module.generate_streaming = fake_generate_streaming
    try:
        client = TestClient(api_server.app)
        response = client.post("/api/ask", json={
            "question": "What time is my quiz?",
            "topic": "general",
            "session_id": "quiz-chat-test",
        })
    finally:
        assistant_service_module.generate_streaming = real_generate_streaming

    assert response.status_code == 200, response.text
    assert response.json()["kind"] == "answer"
    assert response.json()["topic_used"] != "quiz"


def test_ask_stream_detects_quiz_intent_and_emits_quiz_in_done_event():
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    call_count = {"n": 0}

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
        # detect_quiz_intent defaults to 5 questions when no count is
        # stated ("quiz me on X") -- each needs genuinely distinct text or
        # generate_quiz()'s own duplicate-question rejection kicks in.
        call_count["n"] += 1
        yield _fake_single_question_response(call_count["n"])

    quiz_module.generate_streaming = fake_generate_streaming
    try:
        client = TestClient(api_server.app)
        with client.stream("POST", "/api/ask/stream", json={
            "question": "quiz me on computational biology",
            "topic": "general",
            "session_id": "quiz-chat-test",
        }) as response:
            assert response.status_code == 200
            raw = "".join(response.iter_text())
    finally:
        quiz_module.generate_streaming = real_generate_streaming

    assert "event: done" in raw
    done_line = next(line for line in raw.splitlines() if line.startswith("data:") and '"kind"' in line)
    payload = json.loads(done_line[len("data:"):].strip())
    assert payload["kind"] == "quiz"
    assert len(payload["quiz"]["questions"]) == 5
