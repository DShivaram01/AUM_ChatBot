"""Task 35: grounded Document QA -- evidence gating, citations, privacy."""

import re

import numpy as np
import pymupdf

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
