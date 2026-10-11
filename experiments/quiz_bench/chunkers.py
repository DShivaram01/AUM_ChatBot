"""Chunker registry. Every chunker maps production-extracted pages to a list of
{"text", "page", "page_end", "heading_path"} dicts (the production shape).
Experiment branches register additional chunkers here."""

from typing import Callable

CHUNKERS: dict[str, Callable[[list[dict]], list[dict]]] = {}


def register(name: str):
    def wrap(fn):
        CHUNKERS[name] = fn
        return fn
    return wrap


@register("structured400")
def structured400(pages: list[dict]) -> list[dict]:
    """Production baseline: ingestion.document_parser.structured_chunks."""
    from ingestion.document_parser import structured_chunks
    return structured_chunks(pages)
