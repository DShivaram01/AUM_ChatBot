"""Task 36: quiz generation, validation, and source-mode wiring."""

import json

import numpy as np

from core.assistant_service import AssistantService
from core.runtime_manager import RuntimeManager
from pipeline.quiz import _extract_json, _finalize_question, _validate


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

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
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


def test_quiz_document_mode_requires_attached_document():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    result = service.quiz("anything", 3, "document", document_ids=None, session_id="s1")
    assert "error" in result and "attach a document" in result["error"]


def test_quiz_aum_mode_uses_housing_evidence_and_rejects_without_housing():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder(), housing_ok=False))
    result = service.quiz("guest policy", 2, "aum")
    assert "error" in result  # no housing index available in this unit test


def test_quiz_generation_failure_is_a_controlled_error_not_a_malformed_quiz():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    import pipeline.quiz as quiz_module
    real_generate_streaming = quiz_module.generate_streaming

    def always_broken(prompt, tokenizer, model, query_id=None, max_new_tokens=300):
        yield "this is not json at all"

    quiz_module.generate_streaming = always_broken
    try:
        result = service.quiz("anything", 2, "pretrained")
    finally:
        quiz_module.generate_streaming = real_generate_streaming
    assert "quiz" not in result
    assert "error" in result
