"""Basic, single-shot document summarization: Mistral-classified intent,
no map-reduce, no validation -- see pipeline/summarize.py's module
docstring for why map-reduce was rejected (exp/quiz-map-reduce benchmark)."""

import numpy as np

from core.assistant_service import AssistantService
from core.runtime_manager import RuntimeManager
from pipeline.summarize import _trim_to_last_sentence, generate_summary_basic


class WordOverlapEmbedder:
    def encode(self, texts, **_kwargs):
        return np.zeros((len(texts), 4), dtype=np.float32)


class FakeTok:
    """Stand-in for the real tokenizer's .encode() -- generate_summary_basic
    calls it once to size the output-token budget (pipeline/summarize.py's
    _OUTPUT_TOKENS_PER_INPUT_TOKEN formula). One "token" per character is
    plenty for exercising that formula without loading real weights."""
    def encode(self, text):
        return list(range(len(text)))


# ---------- pipeline.summarize._trim_to_last_sentence ----------

def test_trim_to_last_sentence_cuts_a_mid_word_cutoff_back_to_the_last_full_sentence():
    text = "First sentence is fine. Second one too! Third got cut off mid-wo"
    assert _trim_to_last_sentence(text) == "First sentence is fine. Second one too!"


def test_trim_to_last_sentence_leaves_a_complete_ending_untouched():
    text = "First sentence is fine. Second one too!"
    assert _trim_to_last_sentence(text) == text


def test_trim_to_last_sentence_leaves_text_with_no_punctuation_untouched():
    # No sentence boundary anywhere to trim back to -- returning the
    # whole thing unchanged beats silently deleting all of it.
    text = "just a fragment with no terminal punctuation at all"
    assert _trim_to_last_sentence(text) == text


def test_trim_to_last_sentence_handles_empty_text():
    assert _trim_to_last_sentence("") == ""


def test_generate_summary_basic_trims_mid_sentence_cutoff_from_the_real_model_output():
    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    summarize_module.generate_streaming = lambda prompt, *_a, **_kw: iter([
        "Paragraph one is complete. Paragraph two starts but gets cut off mid-wo"
    ])
    try:
        answer = generate_summary_basic(["chunk"], FakeTok(), None, "SUMtest")
    finally:
        summarize_module.generate_streaming = real_generate_streaming
    assert answer == "Paragraph one is complete."


def test_generate_summary_basic_prompt_tells_the_model_not_to_enumerate():
    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    seen_prompts = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, **_kwargs):
        seen_prompts.append(prompt)
        yield "a summary."

    summarize_module.generate_streaming = fake_generate_streaming
    try:
        generate_summary_basic(["chunk"], FakeTok(), None, "SUMtest")
    finally:
        summarize_module.generate_streaming = real_generate_streaming

    assert "do not list or enumerate every item" in seen_prompts[0].lower()


# ---------- pipeline.summarize.generate_summary_basic ----------

def test_generate_summary_basic_passes_through_raw_output_without_validation():
    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    seen_prompts = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, **_kwargs):
        seen_prompts.append(prompt)
        yield "not structured at all, just prose -- exactly as the model said it"

    summarize_module.generate_streaming = fake_generate_streaming
    try:
        answer = generate_summary_basic(["chunk one text", "chunk two text"], FakeTok(), None, "SUMtest")
    finally:
        summarize_module.generate_streaming = real_generate_streaming

    assert len(seen_prompts) == 1  # single-shot: exactly one model call, not one per chunk
    assert "chunk one text" in seen_prompts[0] and "chunk two text" in seen_prompts[0]
    assert answer == "not structured at all, just prose -- exactly as the model said it"


def test_generate_summary_basic_handles_no_chunks_without_crashing():
    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    summarize_module.generate_streaming = lambda prompt, *_a, **_kw: iter(["(nothing to summarize)"])
    try:
        answer = generate_summary_basic([], FakeTok(), None, "SUMtest")
    finally:
        summarize_module.generate_streaming = real_generate_streaming
    assert answer == "(nothing to summarize)"


def test_generate_summary_basic_scales_output_budget_with_input_size():
    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    seen_budgets = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, **_kwargs):
        seen_budgets.append(max_new_tokens)
        yield "ok"

    summarize_module.generate_streaming = fake_generate_streaming
    try:
        generate_summary_basic(["x" * 100], FakeTok(), None, "SUMtest")  # short -> floor of 400
        generate_summary_basic(["x" * 30000], FakeTok(), None, "SUMtest")  # long -> scales past the floor
    finally:
        summarize_module.generate_streaming = real_generate_streaming

    assert seen_budgets[0] == 400  # _MIN_OUTPUT_TOKENS floor, not 100 // 15
    assert seen_budgets[1] > 400  # 30000 "tokens" // 15 = 2000, well past the floor


# ---------- AssistantService.summarize_document / detect_summarize_intent ----------

def test_summarize_document_requires_attached_document():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    result = service.summarize_document(document_ids=None, session_id="s1")
    assert "error" in result and "attach a document" in result["error"]


def test_summarize_document_requires_a_session():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    result = service.summarize_document(document_ids=["doc1"], session_id=None)
    assert "error" in result and "session is required" in result["error"]


def test_summarize_document_returns_raw_answer_with_a_real_query_id():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder(), llm_tok=FakeTok()))

    class FakeChunk:
        def __init__(self, text):
            self.text = text

    class FakeDocument:
        chunks = [FakeChunk("alpha"), FakeChunk("beta")]

    service.documents.store.get = lambda document_id, session_id: FakeDocument()

    import pipeline.summarize as summarize_module
    real_generate_streaming = summarize_module.generate_streaming
    summarize_module.generate_streaming = lambda prompt, *_a, **_kw: iter(["the raw summary, unvalidated"])
    try:
        result = service.summarize_document(document_ids=["doc1"], session_id="s1")
    finally:
        summarize_module.generate_streaming = real_generate_streaming

    assert "error" not in result
    assert result["answer"] == "the raw summary, unvalidated"
    assert result["query_id"]  # non-empty, needed for the feedback button


def test_detect_summarize_intent_delegates_to_the_mistral_classifier():
    """detect_summarize_intent() just forwards to
    pipeline.summarize.classify_summarize_intent with this service's own
    runtime model/tokenizer -- patched here via core.assistant_service's
    own imported name, same convention as pipeline/quiz.py's
    generate_streaming patching (patching pipeline.summarize's copy would
    not intercept the call this module actually makes)."""
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))

    import core.assistant_service as assistant_service_module
    real_classify = assistant_service_module.classify_summarize_intent
    seen = []

    def fake_classify(question, llm_tok, llm_model):
        seen.append(question)
        return question == "summarize this please"

    assistant_service_module.classify_summarize_intent = fake_classify
    try:
        assert service.detect_summarize_intent("summarize this please") is True
        assert service.detect_summarize_intent("what time is it") is False
    finally:
        assistant_service_module.classify_summarize_intent = real_classify
    assert seen == ["summarize this please", "what time is it"]
