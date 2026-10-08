"""Task 54: server-side session history store (_get_history/_record_turn)
and that general_chat/document_chat actually inject it into the prompt
they build -- not just that format_history() itself works in isolation."""

import numpy as np

import server.api_server as api_server
from core.assistant_service import AssistantService
from core.runtime_manager import RuntimeManager


class WordOverlapEmbedder:
    def encode(self, texts, **_kwargs):
        values = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts):
            values[row, row % 4] = 1.0
        return values


def setup_function(_fn):
    api_server._session_history.clear()


def test_record_turn_then_get_history_round_trips():
    api_server._record_turn("s1", "What is the guest policy?", "Up to 3 nights.")
    history = api_server._get_history("s1")
    assert history == [{"question": "What is the guest policy?", "answer": "Up to 3 nights."}]


def test_get_history_is_empty_for_unknown_or_missing_session():
    assert api_server._get_history("never-seen") == []
    assert api_server._get_history(None) == []


def test_record_turn_ignores_blank_question_or_answer():
    api_server._record_turn("s2", "", "an answer")
    api_server._record_turn("s2", "a question", "")
    assert api_server._get_history("s2") == []


def test_history_is_bounded_per_session():
    for i in range(10):
        api_server._record_turn("s3", f"Q{i}", f"A{i}")
    history = api_server._get_history("s3")
    assert len(history) == api_server._HISTORY_MAX_TURNS
    assert history[0]["question"] == "Q6"  # oldest kept
    assert history[-1]["question"] == "Q9"  # most recent


def test_history_is_isolated_per_session():
    api_server._record_turn("s4", "session four question", "answer four")
    api_server._record_turn("s5", "session five question", "answer five")
    assert api_server._get_history("s4") != api_server._get_history("s5")
    assert "session four" not in str(api_server._get_history("s5"))


def test_general_chat_injects_history_into_the_prompt():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    import core.assistant_service as assistant_service_module
    real_generate_streaming = assistant_service_module.generate_streaming
    seen_prompts = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
        seen_prompts.append(prompt)
        yield "an answer"

    assistant_service_module.generate_streaming = fake_generate_streaming
    try:
        history = [{"question": "What is 2+2?", "answer": "4."}]
        list(service.general_chat("And what is that times 10?", history))
    finally:
        assistant_service_module.generate_streaming = real_generate_streaming

    assert "What is 2+2?" in seen_prompts[0]
    assert "Earlier in this conversation:" in seen_prompts[0]


def test_general_chat_with_no_history_omits_the_history_block():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    import core.assistant_service as assistant_service_module
    real_generate_streaming = assistant_service_module.generate_streaming
    seen_prompts = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
        seen_prompts.append(prompt)
        yield "an answer"

    assistant_service_module.generate_streaming = fake_generate_streaming
    try:
        list(service.general_chat("a fresh question with no prior turns"))
    finally:
        assistant_service_module.generate_streaming = real_generate_streaming

    assert "Earlier in this conversation:" not in seen_prompts[0]
