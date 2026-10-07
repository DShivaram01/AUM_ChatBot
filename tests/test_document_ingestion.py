import io

import pymupdf
import numpy as np
from fastapi.testclient import TestClient

from core.assistant_service import AssistantService
from core.orchestrator import set_assistant_service
from core.runtime_manager import RuntimeManager
from ingestion.document_parser import structured_chunks
from server.api_server import app


class FakeEmbedder:
    def encode(self, texts, **_kwargs):
        values = np.zeros((len(texts), 4), dtype=np.float32)
        for row, text in enumerate(texts): values[row, row % 4] = 1.0
        return values


def test_structured_chunks_keep_correct_page_span():
    pages = [
        {"metadata": {"page_number": 1}, "text": "# Section A\nfirst\nsecond"},
        {"metadata": {"page_number": 2}, "text": "continued\n# Section B\nthird"},
        {"metadata": {"page_number": 3}, "text": "fourth"},
    ]
    chunks = structured_chunks(pages)
    assert [(c["page"], c["page_end"]) for c in chunks] == [(1, 2), (2, 3)]


def _pdf_bytes():
    pdf = pymupdf.open()
    page = pdf.new_page(); page.insert_text((72, 72), "Section One\nPrivate alpha content")
    page = pdf.new_page(); page.insert_text((72, 72), "Section Two\nPrivate beta content")
    data = pdf.tobytes(); pdf.close(); return data


def test_document_api_isolation_deletion_and_private_metadata():
    set_assistant_service(AssistantService(RuntimeManager(embedder=FakeEmbedder())))
    client = TestClient(app)
    content = _pdf_bytes()
    from pipeline.classifier import SESSION_TRACES
    traces_before = len(SESSION_TRACES)
    response = client.post("/api/documents", data={"session_id": "session-a"}, files={"file": ("notes.pdf", content, "application/pdf")})
    assert response.status_code == 200, response.text
    metadata = response.json(); document_id = metadata["document_id"]
    assert "Private alpha content" not in str(metadata)
    listing = client.get("/api/documents", params={"session_id": "session-a"}).json()
    assert len(listing) == 1 and "Private" not in str(listing)
    assert len(SESSION_TRACES) == traces_before
    repeated = client.post("/api/documents", data={"session_id": "session-a"}, files={"file": ("notes.pdf", content, "application/pdf")})
    assert repeated.json()["document_id"] == document_id
    assert client.get(f"/api/documents/{document_id}", params={"session_id": "session-b"}).status_code == 403
    service = __import__("core.orchestrator", fromlist=["get_assistant_service"]).get_assistant_service()
    hits = service.retrieve_document(document_id, "session-a", "alpha")
    assert hits and set(hits[0]) == {"chunk_id", "document_id", "page", "page_end", "heading_path", "score"}
    try:
        service.retrieve_document(document_id, "session-b", "alpha")
    except PermissionError:
        pass
    else:
        raise AssertionError("cross-session retrieval was allowed")
    assert client.delete(f"/api/documents/{document_id}", params={"session_id": "session-a"}).status_code == 200
    assert client.get(f"/api/documents/{document_id}", params={"session_id": "session-a"}).status_code == 404
    invalid = client.post("/api/documents", data={"session_id": "session-a"}, files={"file": ("notes.txt", b"hello", "text/plain")})
    assert invalid.status_code == 400


def test_expired_document_is_evicted():
    from datetime import datetime, timedelta, timezone
    from core.schemas import DocumentVersion, SourceRecord
    from stores.document_store import DocumentStore, StoredDocument
    now = datetime.now(timezone.utc)
    source = SourceRecord(
        source_id="src_expired", source_type="user_upload", owner_scope="session",
        created_at=now - timedelta(hours=2), expires_at=now - timedelta(hours=1),
    )
    version = DocumentVersion("doc_expired", "src_expired", "ver_expired", "hash", now, 1, "test")
    store = DocumentStore(); store.put(StoredDocument(source, version, "session-a"))
    try:
        store.get("doc_expired", "session-a")
    except KeyError as exc:
        assert "expired" in str(exc).lower()
    else:
        raise AssertionError("expired document remained accessible")
    assert store.list("session-a") == []
