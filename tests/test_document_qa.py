"""Task 35: grounded Document QA -- evidence gating, citations, privacy."""

import re

import numpy as np
import pymupdf
import pytest

import core.course_service as course_service
from core.assistant_service import AssistantService
from core.document_service import DocumentService
from core.runtime_manager import RuntimeManager
from pipeline.classifier import SESSION_TRACES


class WordOverlapEmbedder:
    """Deterministic, CPU-only stand-in for MiniLM: cosine similarity by
    shared-word fraction against a fixed small vocabulary. Good enough to
    tell a genuinely on-topic query apart from an unrelated one, without
    loading the real embedding model."""

    VOCAB = [
        "refund", "tuition", "withdraw", "semester", "week", "policy",
        "lab", "safety", "goggles", "chemistry", "capital", "peru",
    ]

    def encode(self, texts, **_kwargs):
        rows = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            vec = np.array([1.0 if term in words else 0.0 for term in self.VOCAB], dtype=np.float32)
            norm = np.linalg.norm(vec)
            rows.append(vec / norm if norm > 0 else vec)
        return np.array(rows, dtype=np.float32)


def _refund_pdf_bytes():
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text(
        (72, 72),
        "Refund Policy\nStudents who withdraw within the first two weeks of "
        "a semester receive a full tuition refund. No refund is issued "
        "after week six.",
    )
    data = pdf.tobytes()
    pdf.close()
    return data


def _service_with_ingested_doc(session_id="qa-session-a"):
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    metadata = service.ingest_document("refund.pdf", _refund_pdf_bytes(), session_id)
    return service, metadata["document_id"]


def test_document_chat_abstains_on_unrelated_question():
    service, document_id = _service_with_ingested_doc()
    traces_before = len(SESSION_TRACES)
    items = list(service.document_chat(
        "What is the capital of Peru?", [document_id], "qa-session-a",
    ))
    answer = items[-1][0]
    assert answer == (
        "I could not find enough information in the uploaded document to answer that."
    )
    assert len(SESSION_TRACES) == traces_before + 1
    trace = SESSION_TRACES[-1]
    assert trace.response_state == "not_found"
    assert trace.tab == "document"
    assert "redacted" in trace.full_prompt
    assert "week six" not in trace.full_prompt


def test_document_chat_grounded_answer_cites_page_and_stays_private():
    service, document_id = _service_with_ingested_doc()
    import core.assistant_service as assistant_module
    real_generate_streaming = assistant_module.generate_streaming

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, trace=None):
        assert "week six" in prompt  # the real chunk text IS in the prompt sent to the model
        yield "Full refund within two weeks; none after week six. (refund.pdf, p.1)"

    assistant_module.generate_streaming = fake_generate_streaming
    try:
        items = list(service.document_chat(
            "What is the refund policy for withdrawing students?",
            [document_id], "qa-session-a",
        ))
    finally:
        assistant_module.generate_streaming = real_generate_streaming

    answer = items[-1][0]
    assert "week six" in answer
    trace = SESSION_TRACES[-1]
    assert trace.response_state == "paragraph"
    assert trace.threshold_passed is True
    assert trace.candidates and trace.candidates[0].title == "refund.pdf"
    # The prompt sent to the model DOES contain chunk text (checked above via
    # the fake), but the STORED trace must never carry it.
    assert "week six" not in trace.full_prompt
    assert "redacted" in trace.full_prompt


def test_document_chat_with_multiple_documents_does_not_cite_irrelevant_ones():
    """Task 56: found live with 3 real attached documents -- the evidence
    gate only checks the TOP pooled score, so a confidently-relevant top
    hit let unrelated low-scoring chunks from OTHER attached documents
    through into both the model's context and the citation string the
    prompt tells it to append verbatim. A real generation cited a parking
    PDF and a dining-services PDF for a capstone-grading fact. Confirm
    both the context and the citation list are filtered to hits that
    individually clear the evidence floor, not just the pooled top-5."""
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    service.documents = DocumentService(WordOverlapEmbedder())

    def fake_retrieve_chunks(document_id, session_id, query, limit=5):
        if document_id == "doc_relevant":
            return [{
                "chunk_id": "rel_1", "document_id": "doc_relevant", "filename": "capstone.pdf",
                "page": 1, "page_end": 1, "heading_path": [], "score": 0.48,
                "text": "The oral defense counts for 25 percent of the capstone grade.",
            }]
        return [{
            "chunk_id": f"{document_id}_1", "document_id": document_id,
            "filename": f"{document_id}.pdf", "page": 1, "page_end": 1,
            "heading_path": [], "score": 0.05, "text": "unrelated filler content",
        }]

    service.documents.retrieve_chunks = fake_retrieve_chunks
    import core.assistant_service as assistant_module
    real_generate_streaming = assistant_module.generate_streaming
    seen_prompts = []

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, trace=None):
        seen_prompts.append(prompt)
        yield "The oral defense counts for 25 percent. (capstone.pdf, p.1)"

    assistant_module.generate_streaming = fake_generate_streaming
    try:
        list(service.document_chat(
            "What percentage is the oral defense worth?",
            ["doc_relevant", "doc_filler1", "doc_filler2"], "qa-session-a",
        ))
    finally:
        assistant_module.generate_streaming = real_generate_streaming

    prompt = seen_prompts[0]
    assert "oral defense counts for 25 percent" in prompt
    assert "unrelated filler content" not in prompt
    assert "doc_filler1.pdf" not in prompt
    assert "doc_filler2.pdf" not in prompt


# ---------- start_assignment_chat() -- "can a student converse with the
# assignment document" -- ingests the real assignment content through the
# exact same document_chat() path above, with a "contact your professor"
# fallback swapped in for the generic "uploaded document" one. ----------

def _service_with_assignment_chat_started(session_id="asg-session-a"):
    courses = course_service.list_courses()
    course_id = courses[0]["course_id"]
    assignment_id = course_service.list_assignments(course_id)[0]["assignment_id"]
    content = course_service.get_assignment(course_id, assignment_id)["content"]
    # A real, distinctive word straight from the assignment's own content,
    # not a hardcoded assumption about fixture wording -- robust to
    # data/courses_data.json changing independently of this test.
    content_word = max(re.findall(r"[a-zA-Z]{6,}", content), key=len).lower()

    embedder = WordOverlapEmbedder()
    embedder.VOCAB = [content_word, "qwertyzzyxnotpresent"]
    service = AssistantService(RuntimeManager(embedder=embedder))
    started = service.start_assignment_chat(course_id, assignment_id, session_id)
    return service, started, content_word


def test_start_assignment_chat_returns_an_attachable_document_id():
    service, started, _word = _service_with_assignment_chat_started()
    assert started["document_id"].startswith("doc_")
    assert started["filename"].endswith(".txt")
    # Same shape /api/documents' own upload response has (document_id,
    # filename) -- the client attaches it to a chat the identical way.
    metadata = service.get_document(started["document_id"], "asg-session-a")
    assert metadata["document_id"] == started["document_id"]


def test_start_assignment_chat_is_idempotent_for_the_same_session():
    service, first, _word = _service_with_assignment_chat_started()
    course_id = course_service.list_courses()[0]["course_id"]
    assignment_id = course_service.list_assignments(course_id)[0]["assignment_id"]
    second = service.start_assignment_chat(course_id, assignment_id, "asg-session-a")
    assert first["document_id"] == second["document_id"]


def test_start_assignment_chat_raises_key_error_for_an_unknown_assignment():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    with pytest.raises(KeyError):
        service.start_assignment_chat("NOT_A_REAL_COURSE", "whatever", "asg-session-a")


def test_start_assignment_chat_requires_a_session_id():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    course_id = course_service.list_courses()[0]["course_id"]
    assignment_id = course_service.list_assignments(course_id)[0]["assignment_id"]
    with pytest.raises(ValueError):
        service.start_assignment_chat(course_id, assignment_id, "")


def test_assignment_chat_grounded_answer_comes_only_from_the_assignment():
    service, started, content_word = _service_with_assignment_chat_started()
    import core.assistant_service as assistant_module
    real_generate_streaming = assistant_module.generate_streaming

    def fake_generate_streaming(prompt, tokenizer, model, query_id=None, max_new_tokens=300, trace=None):
        assert "Do NOT use outside knowledge" in prompt
        assert content_word in prompt.lower()  # the real assignment chunk IS in the prompt
        yield f"The assignment covers {content_word}. (cited)"

    assistant_module.generate_streaming = fake_generate_streaming
    try:
        items = list(service.document_chat(
            f"What does the assignment say about {content_word}?",
            [started["document_id"]], "asg-session-a",
        ))
    finally:
        assistant_module.generate_streaming = real_generate_streaming

    answer = items[-1][0]
    assert content_word in answer


def test_assignment_chat_abstention_suggests_contacting_the_professor():
    """The whole point of the feature request: when the assignment itself
    doesn't have enough information, the student is pointed at their
    instructor instead of getting document_chat()'s generic "uploaded
    document" abstention."""
    service, started, _content_word = _service_with_assignment_chat_started()
    items = list(service.document_chat(
        "qwertyzzyxnotpresent in the assignment?",
        [started["document_id"]], "asg-session-a",
    ))
    answer = items[-1][0]
    assert "contact" in answer.lower()
    assert "assignment" in answer.lower()
    assert "uploaded document" not in answer


def test_document_chat_denies_cross_session_access():
    service, document_id = _service_with_ingested_doc(session_id="qa-session-a")
    items = list(service.document_chat(
        "What is the refund policy?", [document_id], "qa-session-intruder",
    ))
    answer = items[-1][0]
    assert "not available in this session" in answer


def test_document_chat_requires_document_and_session():
    service = AssistantService(RuntimeManager(embedder=WordOverlapEmbedder()))
    no_doc = list(service.document_chat("anything", [], "some-session"))
    assert "attach a document" in no_doc[-1][0]
    no_session = list(service.document_chat("anything", ["doc_x"], None))
    assert "session is required" in no_session[-1][0]
