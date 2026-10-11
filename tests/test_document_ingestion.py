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
    invalid = client.post("/api/documents", data={"session_id": "session-a"}, files={"file": ("notes.xlsx", b"PK\x03\x04", "application/octet-stream")})  # .txt/.docx are supported since Task 57
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


# ---------- Task 57-B: TXT / Markdown ----------

def test_markdown_upload_keeps_headings_and_txt_treats_hash_lines_as_text():
    from core.document_service import DocumentService
    service = DocumentService(FakeEmbedder())
    md = service.ingest("notes.md", b"# Housing\nGuests may stay two nights.\n## Pets\nNo pets.", "s1")
    assert [c.heading_path for c in md.chunks] == [("Housing",), ("Housing", "Pets")]
    assert md.source.mime_type == "text/markdown"
    txt = service.ingest("notes.txt", b"# not a heading\nplain body", "s1")
    assert len(txt.chunks) == 1 and txt.chunks[0].heading_path == ()
    assert "# not a heading" in txt.chunks[0].text and txt.source.mime_type == "text/plain"


def test_text_upload_rejects_binary_non_utf8_empty_and_unknown_types():
    from core.document_service import DocumentService
    service = DocumentService(FakeEmbedder())
    for name, data, message in [
        ("a.txt", b"abc\x00def", "text document"), ("a.txt", "caf\u00e9".encode("latin-1"), "UTF-8"),
        ("a.md", b"   \n\n", "extractable text"), ("a.docx", b"PK\x03\x04", "valid Office document"), ("a.xlsx", b"PK", "Only PDF, TXT"),
        ("a.pdf", b"not a pdf", "Only PDF, TXT"),
    ]:
        try:
            service.ingest(name, data, "s1")
        except ValueError as exc:
            assert message in str(exc), (name, exc)
        else:
            raise AssertionError(f"{name} should have been rejected")


def test_markdown_upload_through_api_and_document_chat_sources():
    set_assistant_service(AssistantService(RuntimeManager(embedder=FakeEmbedder())))
    client = TestClient(app)
    response = client.post("/api/documents", data={"session_id": "md-s"},
                           files={"file": ("guide.md", b"# Intro\nPrivate gamma content", "text/markdown")})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["filename"] == "guide.md" and body["chunks"] == 1 and body["pages"] == 1


# ---------- Task 57 phase 4: CSV / DOCX / PPTX ----------

def _zip(files):
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


_WNS = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
_ANS = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'


def _docx():
    def para(text, style=None):
        props = '<w:pPr><w:pStyle w:val="%s"/></w:pPr>' % style if style else ""
        return f"<w:p>{props}<w:r><w:t>{text}</w:t></w:r></w:p>"
    body = (para("Parking Rules", "Heading1") + para("Permits cost $90 per year.") + para("Towing", "Heading2")
            + para("Cars in fire lanes are towed.")
            + "<w:tbl><w:tr><w:tc><w:p><w:r><w:t>Lot</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Hours</w:t></w:r></w:p></w:tc></w:tr>"
              "<w:tr><w:tc><w:p><w:r><w:t>Lot C</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>24/7</w:t></w:r></w:p></w:tc></w:tr></w:tbl>")
    return _zip({"word/document.xml": f"<w:document {_WNS}><w:body>{body}</w:body></w:document>"})


def _pptx():
    slide = lambda *ps: f"<p:sld xmlns:p='x' {_ANS}>" + "".join(f"<a:p><a:r><a:t>{p}</a:t></a:r></a:p>" for p in ps) + "</p:sld>"  # noqa: E731
    return _zip({"ppt/slides/slide1.xml": slide("Welcome", "Orientation week"),
                 "ppt/slides/slide2.xml": slide("Library", "Open until midnight"),
                 "ppt/slides/slide10.xml": slide("Contacts", "Call ext. 2200"),
                 "ppt/slides/slide11.xml": f"<p:sld xmlns:p='x' {_ANS}><a:p><a:r><a:t>Market</a:t></a:r><a:br/>"
                                           "<a:r><a:t>Trends</a:t></a:r></a:p></p:sld>"})


def test_docx_headings_tables_and_pptx_slides_as_pages():
    from core.document_service import DocumentService
    service = DocumentService(FakeEmbedder())
    doc = service.ingest("rules.docx", _docx(), "s1")
    assert [c.heading_path for c in doc.chunks] == [("Parking Rules",), ("Parking Rules", "Towing")]
    assert "Permits cost $90" in doc.chunks[0].text and "Lot C | 24/7" in doc.chunks[1].text
    deck = service.ingest("orientation.pptx", _pptx(), "s1")
    # slide10 sorts after slide2 (numeric, not lexical); short slides merge
    # into one chunk spanning p.1-3, exactly like short PDF pages do
    assert deck.version.page_count == 4
    assert [(c.page, c.page_end) for c in deck.chunks] == [(1, 4)]
    assert "Market Trends" in deck.chunks[0].text  # <a:br/> is a space, not glued
    text = deck.chunks[0].text
    assert text.index("Welcome") < text.index("Library") < text.index("Call ext. 2200")


def test_csv_rows_are_self_describing():
    from core.document_service import DocumentService
    doc = DocumentService(FakeEmbedder()).ingest("rooms.csv", b"Room,Capacity\nE214,30\n\nB101,120\n", "s1")
    assert doc.chunks[0].text == "Room: E214; Capacity: 30\n\nRoom: B101; Capacity: 120"
    assert doc.source.mime_type == "text/csv"


def test_office_zip_bomb_and_missing_parts_are_rejected():
    from core.document_service import DocumentService
    from ingestion import document_parser
    service = DocumentService(FakeEmbedder())
    for name, data, message in [("x.docx", _zip({"other.xml": "<a/>"}), "valid Office"),
                                ("x.docx", _zip({"word/document.xml": "<not-closed"}), "valid Office")]:
        try:
            service.ingest(name, data, "s1")
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(name)
    old = document_parser._MAX_UNZIPPED_BYTES
    document_parser._MAX_UNZIPPED_BYTES = 100
    try:
        service.ingest("big.docx", _zip({"word/document.xml": "x" * 1000}), "s1")
    except ValueError as exc:
        assert "too large" in str(exc)
    else:
        raise AssertionError("zip bomb accepted")
    finally:
        document_parser._MAX_UNZIPPED_BYTES = old
