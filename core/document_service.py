"""Private PDF ingestion and per-document vector indexing."""

import hashlib
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import faiss
import numpy as np

from core.schemas import Chunk, DocumentVersion, SourceRecord
from ingestion.document_parser import content_hash, extract_pdf_pages, structured_chunks, validate_pdf
from stores.document_store import DocumentStore, StoredDocument


class DocumentService:
    PARSER_NAME = "pymupdf4llm"
    PARSER_VERSION = "1.28.2-task34-v1"

    def __init__(self, embedder, store=None):
        self.embedder = embedder
        self.store = store or DocumentStore()

    def ingest(self, filename, content, session_id):
        if not session_id.strip(): raise ValueError("session_id is required")
        validate_pdf(filename, content)
        digest = content_hash(content)
        identity = hashlib.sha256(f"{session_id}:{digest}".encode()).hexdigest()[:16]
        document_id, source_id = f"doc_{identity}", f"src_{identity}"
        version_id = f"ver_{digest[:16]}"
        now = datetime.now(timezone.utc)
        with tempfile.NamedTemporaryFile(suffix=".pdf") as uploaded:
            uploaded.write(content); uploaded.flush()
            pages = extract_pdf_pages(Path(uploaded.name))
        records = structured_chunks(pages)
        if not records: raise ValueError("The PDF did not contain extractable text")
        chunks = []
        for position, record in enumerate(records):
            text = record["text"]
            chunks.append(Chunk(
                chunk_id=f"chk_{identity}_{position:04d}", document_id=document_id,
                document_version_id=version_id, heading_path=tuple(record["heading_path"]),
                page=record["page"], page_end=record["page_end"], text=text,
                content_hash=content_hash(text.encode()), token_count=len(text.split()),
            ))
        embeddings = self.embedder.encode([c.text for c in chunks], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        index = faiss.IndexFlatIP(embeddings.shape[1]); index.add(embeddings)
        source = SourceRecord(
            source_id=source_id, source_type="user_upload", owner_scope="session",
            original_filename=filename, mime_type="application/pdf", content_hash=digest,
            parser_name=self.PARSER_NAME, parser_version=self.PARSER_VERSION,
            created_at=now, expires_at=now + timedelta(hours=24),
        )
        version = DocumentVersion(document_id, source_id, version_id, digest, now, len(pages), self.PARSER_VERSION)
        document = StoredDocument(source, version, session_id, chunks, index, embeddings)
        self.store.put(document)
        return document

    def metadata(self, document_id, session_id):
        return self.describe(self.store.get(document_id, session_id))

    def retrieve(self, document_id, session_id, query, limit=5):
        document = self.store.get(document_id, session_id)
        query_embedding = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True,
        ).astype(np.float32)
        scores, positions = document.index.search(query_embedding, min(limit, len(document.chunks)))
        return [
            {
                "chunk_id": document.chunks[position].chunk_id,
                "document_id": document_id,
                "page": document.chunks[position].page,
                "page_end": document.chunks[position].page_end,
                "heading_path": list(document.chunks[position].heading_path),
                "score": float(score),
            }
            for score, position in zip(scores[0], positions[0]) if position >= 0
        ]

    def list(self, session_id):
        return [self.describe(document) for document in self.store.list(session_id)]

    def delete(self, document_id, session_id):
        self.store.delete(document_id, session_id)

    @staticmethod
    def describe(document):
        return {
            "document_id": document.version.document_id,
            "document_version_id": document.version.document_version_id,
            "filename": document.source.original_filename,
            "content_hash": document.version.content_hash,
            "pages": document.version.page_count, "chunks": len(document.chunks),
            "status": "ready",
        }
