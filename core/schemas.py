"""Typed contracts shared by the assistant service and future source stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    source_type: Literal["aum", "user_upload"]
    owner_scope: Literal["public", "session", "user", "project"]
    original_filename: str | None = None
    mime_type: str | None = None
    content_hash: str = ""
    parser_name: str = ""
    parser_version: str = ""
    access_classification: str = "USER_PRIVATE"
    created_at: datetime | None = None
    expires_at: datetime | None = None


@dataclass(frozen=True)
class DocumentVersion:
    document_id: str
    source_id: str
    document_version_id: str
    content_hash: str
    uploaded_at: datetime
    page_count: int
    parser_version: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    document_version_id: str
    heading_path: tuple[str, ...] = ()
    page: int | None = None
    page_end: int | None = None
    content_hash: str = ""
    token_count: int = 0
    text: str = field(repr=False, default="")
