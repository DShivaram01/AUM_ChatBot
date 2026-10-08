"""Task 53 Phase 1-2 (Task 55: now the ONLY way to reach quiz generation,
the Quiz button and its /api/quiz endpoint having been removed): a
natural-language quiz request typed into ordinary chat (/api/ask,
/api/ask/stream) must reach AssistantService.quiz(), not an unstructured
free-text answer. Integration-level (real FastAPI app via TestClient),
with the LLM mocked exactly like tests/test_quiz.py already does -- this
project's standing rule that default tests never load real model
weights."""

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
    api_server._session_last_quiz.clear()


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


# ---------- Task 54: quiz follow-ups ----------

def _generate_a_chat_quiz(client, session_id, count=3):
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    call_count = {"n": 0}

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
        call_count["n"] += 1
        yield _fake_single_question_response(call_count["n"])

    quiz_module.generate_streaming = fake_generate_streaming
    try:
        response = client.post("/api/ask", json={
            "question": f"Create {count} MCQs on computational biology",
            "topic": "general", "session_id": session_id,
        })
    finally:
        quiz_module.generate_streaming = real_generate_streaming
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "quiz"
    return body["quiz"]


def test_explain_question_followup_after_a_chat_originated_quiz():
    client = TestClient(api_server.app)
    _generate_a_chat_quiz(client, "followup-test-1", count=3)

    response = client.post("/api/ask", json={
        "question": "explain question 2",
        "topic": "general", "session_id": "followup-test-1",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "answer"
    assert "Question 2?" in body["answer"]
    assert "because" in body["answer"]  # the stored explanation, looked up not regenerated


def test_explain_question_followup_works_regardless_of_how_the_quiz_was_recorded():
    # Task 55 removed the Quiz button and its dedicated /api/quiz endpoint
    # -- chat-originated generation (_try_quiz_from_chat) is the only
    # thing that calls _record_last_quiz() now. This still confirms the
    # follow-up handler itself only cares about _session_last_quiz's
    # contents, not which code path populated it -- a real property of
    # the design worth keeping covered even with one fewer caller.
    api_server._record_last_quiz(
        "followup-test-2",
        {"title": "Computational Biology Quiz", "questions": [
            {"question_id": "q1", "question": "Question 1?", "options": ["A", "B", "C", "D"],
             "correct_index": 1, "explanation": "because", "evidence_ids": []},
        ]},
        "computational biology", 1, "pretrained", None,
    )

    client = TestClient(api_server.app)
    response = client.post("/api/ask", json={
        "question": "why is question 1 correct?",
        "topic": "general", "session_id": "followup-test-2",
    })
    assert response.status_code == 200, response.text
    assert "Question 1?" in response.json()["answer"]


def test_explain_question_out_of_range_gives_a_clear_message_not_a_crash():
    client = TestClient(api_server.app)
    _generate_a_chat_quiz(client, "followup-test-3", count=2)

    response = client.post("/api/ask", json={
        "question": "explain question 9",
        "topic": "general", "session_id": "followup-test-3",
    })
    assert response.status_code == 200, response.text
    assert "2 question" in response.json()["answer"]


def test_regenerate_followup_reuses_the_same_params():
    client = TestClient(api_server.app)
    first_quiz = _generate_a_chat_quiz(client, "followup-test-4", count=2)

    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    call_count = {"n": 100}  # distinct text from the first batch

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
        call_count["n"] += 1
        yield _fake_single_question_response(call_count["n"])

    quiz_module.generate_streaming = fake_generate_streaming
    try:
        response = client.post("/api/ask", json={
            "question": "try again",
            "topic": "general", "session_id": "followup-test-4",
        })
    finally:
        quiz_module.generate_streaming = real_generate_streaming

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["kind"] == "quiz"
    assert len(body["quiz"]["questions"]) == 2  # same count as the original
    assert body["quiz"] != first_quiz  # genuinely regenerated, not the cached copy


def test_followup_phrase_without_a_prior_quiz_does_not_misfire():
    # "explain question 3" with nothing generated yet in this session --
    # must not crash or fabricate a quiz answer; falls through to normal
    # chat routing (which, with no LLM mocked, fails safely with a 503
    # here since _model_loaded gates it -- the real assertion is just
    # that _try_quiz_followup_from_chat itself returns None).
    from server.api_server import _try_quiz_followup_from_chat
    from server.api_server import ChatRequest
    req = ChatRequest(question="explain question 3", topic="general", session_id="never-had-a-quiz")
    assert _try_quiz_followup_from_chat(req) is None
