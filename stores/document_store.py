"""Private, session-scoped uploaded-document lifecycle store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from core.schemas import Chunk, DocumentVersion, SourceRecord


@dataclass
class StoredDocument:
    source: SourceRecord
    version: DocumentVersion
    owner_session_id: str
    chunks: list[Chunk] = field(default_factory=list)
    index: object | None = None
    embeddings: object | None = None


class DocumentStore:
    """In-memory private documents; never mutates AUM retrieval stores."""

    def __init__(self) -> None:
        self._documents: dict[str, StoredDocument] = {}

    def put(self, document: StoredDocument) -> None:
        self._documents[document.version.document_id] = document

    def get(self, document_id: str, session_id: str) -> StoredDocument:
        document = self._documents.get(document_id)
        if document is None:
            raise KeyError("Document not found")
        if document.owner_session_id != session_id:
            raise PermissionError("Document is not available in this session")
        if document.source.expires_at and document.source.expires_at <= datetime.now(timezone.utc):
            self._documents.pop(document_id, None)
            raise KeyError("Document has expired")
        return document

    def delete(self, document_id: str, session_id: str) -> None:
        self.get(document_id, session_id)
        del self._documents[document_id]

    def list(self, session_id: str) -> list[StoredDocument]:
        now = datetime.now(timezone.utc)
        return [
            document for document in self._documents.values()
            if document.owner_session_id == session_id
            and (document.source.expires_at is None or document.source.expires_at > now)
        ]
