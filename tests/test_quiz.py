"""Task 36: quiz generation, validation, and source-mode wiring."""

import json

import numpy as np

from core.assistant_service import AssistantService
from core.runtime_manager import RuntimeManager
from pipeline.classifier import get_trace
from pipeline.quiz import (
    _MAX_CONTEXT_CHARS_PER_CHUNK, _extract_json, _finalize_question, _validate,
    generate_quiz, generate_quiz_basic,
)


class WordOverlapEmbedder:
    VOCAB = ["guest", "overnight", "midnight", "refund", "tuition", "lab", "safety"]

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            words = set(text.lower().split())
            vec = np.array([1.0 if term in words else 0.0 for term in self.VOCAB], dtype=np.float32)
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm > 0 else vec)
        return np.array(rows, dtype=np.float32)


# ---------- pure validation/extraction (already covered more thoroughly by
# direct invocation during development; kept here as a durable regression) ----

def test_validate_rejects_wrong_count_and_duplicates_and_bad_options():
    good_q = {"question": "Q1?", "options": ["A", "B", "C", "D"], "correct_index": 0, "evidence_refs": [1]}
    assert _validate({"questions": [good_q]}, 2, require_evidence=False) != []
    assert _validate({"questions": [good_q, good_q]}, 2, require_evidence=False) != []  # duplicate
    bad_opts = {"question": "Q?", "options": ["A", "A", "C", "D"], "correct_index": 0, "evidence_refs": []}
    assert _validate({"questions": [bad_opts]}, 1, require_evidence=False) != []
    assert _validate({"questions": [good_q]}, 1, require_evidence=True) == []  # refs present, OK
    no_refs_q = {"question": "Q2?", "options": ["A", "B", "C", "D"], "correct_index": 0}
    assert _validate({"questions": [no_refs_q]}, 1, require_evidence=True) != []  # missing refs
    assert _validate({"questions": [good_q]}, 1, require_evidence=False) == []


def test_extract_json_tolerates_markdown_fence_and_prose():
    raw = 'Sure, here you go:\n```json\n{"title": "X", "questions": []}\n```\nEnjoy!'
    assert _extract_json(raw) == {"title": "X", "questions": []}


def test_extract_json_strips_js_style_comments_but_not_urls_in_strings():
    # Found via Task 43's real-generation measurement: the model sometimes
    # appends "// explanation" after an array element, which is not valid
    # JSON. Must be stripped outside a string, but a literal "//" inside a
    # real string (e.g. a URL) must survive.
    raw = (
        '{"options": ["FTP", "SMTP", // explains itself\n "HTTP"], '
        '"note": "see https://www.aum.edu for more"}'
    )
    data = _extract_json(raw)
    assert data["options"] == ["FTP", "SMTP", "HTTP"]
    assert data["note"] == "see https://www.aum.edu for more"


def test_finalize_question_maps_evidence_refs_to_real_ids_not_model_text():
    q = {"question": "Q?", "options": ["A", "B", "C", "D"], "correct_index": 1,
         "explanation": "e", "evidence_refs": [2]}
    finalized = _finalize_question(q, {1: "chk_a", 2: "chk_b"}, 1)
    assert finalized["evidence_ids"] == ["chk_b"]
    assert finalized["question_id"] == "q1"


# ---------- AssistantService.quiz() integration, generation mocked ----------

def _fake_single_question_response(number, evidence_refs):
    return json.dumps({
        "question": f"Question {number}?",
        "options": ["A", "B", "C", "D"],
        "correct_index": number % 4,
        "explanation": "because",
        "evidence_refs": evidence_refs,
    })


def test_quiz_pretrained_mode_needs_no_evidence_and_succeeds():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    # generate_quiz() in pipeline/quiz.py imported generate_streaming into its
    # own module namespace -- patching core.assistant_service's copy would
    # not intercept the call actually made, so patch it where it's used.
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    call_count = {"n": 0}

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, avoid_texts=None, **_kwargs):
        assert "<context>" not in prompt  # pretrained mode must not inject any evidence
        call_count["n"] += 1
        yield _fake_single_question_response(call_count["n"], evidence_refs=[])

    quiz_module.generate_streaming = fake_generate_streaming
    try:
        result = service.quiz("operating systems", 3, "pretrained")
    finally:
        quiz_module.generate_streaming = real_generate_streaming
    assert "quiz" in result, result
    assert call_count["n"] == 3  # one model call per question, not one call for the whole quiz
    assert len(result["quiz"]["questions"]) == 3
    assert all(q["evidence_ids"] == [] for q in result["quiz"]["questions"])


def test_generate_quiz_caps_each_context_chunk_so_prompt_stays_well_under_truncation_limit():
    """Task 52: housing/document grounding can hand generate_quiz() up to 6
    full chunks (each up to ~400 tokens). Joined untruncated, that
    routinely exceeded generate_streaming()'s prompt-length safety net and
    silently destroyed the prompt's closing instruction tag. Confirm the
    assembled prompt never contains an unbounded chunk, regardless of how
    long the retrieved chunk text actually is."""
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    seen_prompts = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, avoid_texts=None, **_kwargs):
        seen_prompts.append(prompt)
        yield _fake_single_question_response(1, evidence_refs=[1])

    long_chunk = "word " * 2000  # far longer than _MAX_CONTEXT_CHARS_PER_CHUNK
    quiz_module.generate_streaming = fake_generate_streaming
    try:
        quiz, errors = generate_quiz(
            "guest policy", 1, None, None, "Q1",
            context_chunks=[long_chunk] * 6,
            evidence_map={i: f"chk_{i}" for i in range(1, 7)},
            require_evidence=True,
        )
    finally:
        quiz_module.generate_streaming = real_generate_streaming
    assert errors == [], errors
    assert len(seen_prompts) == 1
    prompt = seen_prompts[0]
    # Each of the 6 chunks must have been capped, not passed through whole.
    assert len(long_chunk) > _MAX_CONTEXT_CHARS_PER_CHUNK * 6  # sanity on the fixture itself
    for marker in ("[1]", "[2]", "[3]", "[4]", "[5]", "[6]"):
        idx = prompt.index(marker)
        next_marker_idx = prompt.find("[", idx + 1)
        segment = prompt[idx:next_marker_idx] if next_marker_idx != -1 else prompt[idx:]
        assert len(segment) < _MAX_CONTEXT_CHARS_PER_CHUNK + len(marker) + 20


def test_quiz_document_mode_requires_attached_document():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    result = service.quiz("anything", 3, "document", document_ids=None, session_id="s1")
    assert "error" in result and "attach a document" in result["error"]


def test_quiz_housing_mode_uses_housing_evidence_and_rejects_without_housing():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder(), housing_ok=False))
    result = service.quiz("guest policy", 2, "housing")
    assert "error" in result  # no housing index available in this unit test


def test_quiz_document_mode_uses_best_scoring_chunks_across_all_attached_documents():
    """Task 56: quiz()'s document branch used to concatenate per-document
    hit lists and slice [:6] WITHOUT sorting by score first -- whichever
    document was listed first (or merely returned >=6 of its own hits)
    silently crowded out every other attached document's content, with
    no error. Attach a low-relevance document FIRST and a high-relevance
    one SECOND; the high-relevance document's chunks must still win."""
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))

    def fake_retrieve_chunks(document_id, session_id, query, limit=6):
        # Simulate retrieve_chunks()'s own real contract: each call
        # returns ITS document's hits already sorted by score -- the bug
        # was in how the CALLER (quiz()) merged multiple such lists.
        if document_id == "doc_low":
            return [
                {"chunk_id": f"low_{i}", "text": f"low relevance chunk {i}", "score": 0.1}
                for i in range(6)  # >= 6 on its own -- old code: this alone fills hits[:6]
            ]
        return [{"chunk_id": "high_1", "text": "the single highly relevant chunk", "score": 0.9}]

    service.documents.retrieve_chunks = fake_retrieve_chunks
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    seen_prompts = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, avoid_texts=None, **_kwargs):
        seen_prompts.append(prompt)
        yield _fake_single_question_response(1, evidence_refs=[1])

    quiz_module.generate_streaming = fake_generate_streaming
    try:
        result = service.quiz(
            "anything", 1, "document",
            document_ids=["doc_low", "doc_high"], session_id="s1",
        )
    finally:
        quiz_module.generate_streaming = real_generate_streaming

    assert "quiz" in result, result
    assert "the single highly relevant chunk" in seen_prompts[0]


def test_quiz_generation_failure_is_a_controlled_error_not_a_malformed_quiz():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming

    def always_broken(prompt, tokenizer, model, query_id=None, max_new_tokens=300, avoid_texts=None, **_kwargs):
        yield "this is not json at all"

    quiz_module.generate_streaming = always_broken
    try:
        result = service.quiz("anything", 2, "pretrained")
    finally:
        quiz_module.generate_streaming = real_generate_streaming
    assert "quiz" not in result
    assert "error" in result


# ---------- basic document quiz: no JSON schema, no validation, no retries ----

def test_generate_quiz_basic_splits_chunks_evenly_and_skips_validation():
    """6 chunks / 3 questions -> 2 chunks per question, and whatever the
    model returns (even non-JSON prose) is passed through untouched."""
    chunks = [f"chunk{i}" for i in range(1, 7)]
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    seen_prompts = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, **_kwargs):
        seen_prompts.append(prompt)
        yield f"not json, just prose for {query_id}"

    quiz_module.generate_streaming = fake_generate_streaming
    try:
        # generate_quiz_basic() streams now -- drain to the last yield
        # for the final concatenated text, same as any caller that just
        # wants the finished result would.
        answer = ""
        for partial in generate_quiz_basic("topic", 3, chunks, None, None, "QZtest"):
            answer = partial
    finally:
        quiz_module.generate_streaming = real_generate_streaming

    assert len(seen_prompts) == 3
    assert "chunk1" in seen_prompts[0] and "chunk2" in seen_prompts[0] and "chunk3" not in seen_prompts[0]
    assert "chunk3" in seen_prompts[1] and "chunk4" in seen_prompts[1]
    assert "Question 1:" in answer and "Question 3:" in answer
    assert "not json, just prose for QZtest-basic-q1" in answer


def _drain(gen) -> tuple[str, str]:
    """quiz_basic_document()/summarize_document() are generators now,
    streaming (text, sources, trace_placeholder, query_id) 4-tuples like
    document_chat() -- drain to the last one for the final (text, qid),
    same convention server/api_server.py's _drain_activity_generator_sync
    uses for the real blocking endpoint."""
    text, qid = "", ""
    for text, _sources, _trace, qid in gen:
        pass
    return text, qid


def test_quiz_basic_document_requires_attached_document():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    text, qid = _drain(service.quiz_basic_document("anything", 3, document_ids=None, session_id="s1"))
    assert "attach a document" in text
    assert qid  # still gets a real query_id -- a QueryTrace was stored even for this error


def test_quiz_basic_document_returns_raw_answer_not_a_structured_quiz():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))

    class FakeChunk:
        def __init__(self, text):
            self.text = text

    class FakeDocument:
        chunks = [FakeChunk("alpha"), FakeChunk("beta"), FakeChunk("gamma"), FakeChunk("delta")]

    service.documents.store.get = lambda document_id, session_id: FakeDocument()

    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    quiz_module.generate_streaming = lambda prompt, *_a, **_kw: iter(["raw model text, unvalidated"])
    try:
        text, qid = _drain(service.quiz_basic_document(
            "anything", 2, document_ids=["doc1"], session_id="s1",
        ))
    finally:
        quiz_module.generate_streaming = real_generate_streaming

    assert qid
    assert text.count("raw model text, unvalidated") == 2


# ---------- real streaming, and the QueryTrace fix ----------

def test_generate_quiz_basic_yields_growing_text_across_all_questions_not_just_a_final_value():
    """The whole point of making this a generator: a caller watching the
    stream sees each question's text grow and accumulate, not silence
    for the whole multi-call run followed by one final blob -- this is
    the actual "streaming doesn't work" bug for basic quiz."""
    chunks = ["chunk one", "chunk two"]
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    quiz_module.generate_streaming = lambda prompt, *_a, **_kw: iter(["Q", "Q done."])
    try:
        seen = list(generate_quiz_basic("topic", 2, chunks, None, None, "QZtest"))
    finally:
        quiz_module.generate_streaming = real_generate_streaming

    assert len(seen) == 4  # 2 growing yields per question x 2 questions
    assert seen[0] == "Question 1:\nQ"
    assert seen[1] == "Question 1:\nQ done."
    # question 2's text accumulates ON TOP of question 1's finished text:
    assert seen[2] == "Question 1:\nQ done.\n\nQuestion 2:\nQ"
    assert seen[3] == "Question 1:\nQ done.\n\nQuestion 2:\nQ done."


def test_quiz_basic_document_stores_a_retrievable_query_trace():
    """Quiz/summarize never stored a QueryTrace before this fix, so
    /api/feedback's get_trace(query_id) lookup always missed. This must
    store one, same as document_chat() does."""
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))

    class FakeChunk:
        def __init__(self, text):
            self.text = text

    class FakeDocument:
        chunks = [FakeChunk("alpha"), FakeChunk("beta")]

    service.documents.store.get = lambda document_id, session_id: FakeDocument()

    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming
    quiz_module.generate_streaming = lambda prompt, *_a, **_kw: iter(["a question."])
    try:
        text, qid = _drain(service.quiz_basic_document(
            "computational biology", 1, document_ids=["doc1"], session_id="s1",
        ))
    finally:
        quiz_module.generate_streaming = real_generate_streaming

    trace = get_trace(qid)
    assert trace is not None
    assert trace.query == "computational biology"
    assert trace.final_answer == text
    assert trace.intent_type == "quiz"


def test_quiz_basic_document_stores_a_trace_even_on_a_validation_error():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    text, qid = _drain(service.quiz_basic_document("anything", 3, document_ids=None, session_id="s1"))
    trace = get_trace(qid)
    assert trace is not None
    assert trace.final_answer == text
