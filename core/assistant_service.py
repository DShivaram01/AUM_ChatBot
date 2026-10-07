"""Service-owned COS, Housing, routing, and pretrained answer execution."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

import config
from core.document_service import DocumentService
from core.runtime_manager import RuntimeManager
from pipeline.answer import (
    build_cos_answer_streaming,
    build_housing_answer_streaming,
    generate_streaming,
    render_trace_md,
)
from pipeline.classifier import classify_query, classify_topic, next_query_id
from pipeline.memory import logger
from pipeline.retrieval import retrieve_cos_rrf, retrieve_housing_logged


class AssistantService:
    """The application boundary used by FastAPI and the debug Gradio client."""

    selection_re = re.compile(r"^\s*(\d{1,2})\s*$")

    def __init__(self, runtime: RuntimeManager, *, debug: bool = False):
        self.runtime = runtime
        self.debug = debug
        self.documents = DocumentService(runtime.embedder)

    def ingest_document(self, filename: str, content: bytes, session_id: str) -> dict:
        document = self.documents.ingest(filename, content, session_id)
        return self.documents.describe(document)

    def get_document(self, document_id: str, session_id: str) -> dict:
        return self.documents.metadata(document_id, session_id)

    def list_documents(self, session_id: str) -> list[dict]:
        return self.documents.list(session_id)

    def retrieve_document(self, document_id: str, session_id: str, query: str) -> list[dict]:
        return self.documents.retrieve(document_id, session_id, query)

    def delete_document(self, document_id: str, session_id: str) -> None:
        self.documents.delete(document_id, session_id)

    def classify_topic(self, question: str, routing_path: list[str] | None = None) -> str:
        r = self.runtime
        return classify_topic(
            question, r.embedder, r.cos_index, r.housing_index, r.housing_ok,
            r.llm_tok, r.llm_model, routing_path,
        )

    def cos_chat(self, message: str, history: list, pending_cands: list) -> Iterator[tuple]:
        message = message.strip()
        qid = next_query_id("COS")
        logger.info(f"[{qid}] NEW COS QUERY: {message!r}")
        if not message:
            yield "Please type a question about AUM research projects.", pending_cands, "", qid
            return
        selected_match = self.selection_re.match(message)
        if selected_match and pending_cands:
            selected_number = int(selected_match.group(1))
            if not 1 <= selected_number <= len(pending_cands):
                yield f"Please enter a number between 1 and {len(pending_cands)}.", pending_cands, "", qid
                return
            selected = pending_cands[selected_number - 1]
            if selected.get("_person_choice"):
                yield from self.cos_chat(
                    f"projects associated with {selected['_person_choice']}", history, []
                )
                return
            selected["_selection_n"] = selected_number
            qinfo = {
                "type": "TYPE_TOPIC", "person_hints": [], "person_hint": None,
                "year_hint": None, "dept_hint": None, "is_broad": False,
            }
            for partial, _, trace in build_cos_answer_streaming(
                f"Tell me about: {selected['meta'].get('title', '')}", [], qinfo,
                self.runtime.llm_tok, self.runtime.llm_model, qid, selected_cand=selected,
            ):
                yield partial, [], render_trace_md(trace) if self.debug else "", qid
            return
        qinfo = classify_query(message)
        if qinfo.get("person_ambiguous"):
            names = qinfo["person_hints"]
            prompt = "I found multiple people matching that name. Please choose one:\n\n" + "\n".join(
                f"{i}. {name}" for i, name in enumerate(names, 1)
            ) + "\n\nReply with a number to continue."
            yield prompt, [{"_person_choice": name} for name in names], "", qid
            return
        r = self.runtime
        candidates = retrieve_cos_rrf(
            message, qinfo, r.embedder, r.cos_index, r.cos_embeddings,
            r.cos_metadata, r.cos_texts, r.cos_bm25, r.reranker, query_id=qid,
        )
        pending = []
        for partial, returned, trace in build_cos_answer_streaming(
            message, candidates, qinfo, r.llm_tok, r.llm_model, qid
        ):
            if returned is not None:
                pending = returned
            yield partial, pending, render_trace_md(trace) if self.debug else "", qid

    def housing_chat(self, message: str, history: list, _pending: list) -> Iterator[tuple]:
        message = message.strip()
        qid = next_query_id("HSG")
        if not message:
            yield "Please type a question about AUM Housing policy.", [], "", qid
            return
        r = self.runtime
        if not r.housing_ok:
            yield f"Housing PDF not found: {config.HOUSING_PDF}", [], "", qid
            return
        started = time.time()
        hits = retrieve_housing_logged(
            message, r.embedder, r.housing_index, r.housing_embeddings,
            r.housing_chunks, query_id=qid, reranker=r.reranker,
        )
        for partial, trace in build_housing_answer_streaming(
            message, hits, r.llm_tok, r.llm_model, qid,
            search_ms=(time.time() - started) * 1000,
        ):
            yield partial, [], render_trace_md(trace) if self.debug else "", qid

    def general_chat(self, question: str) -> Iterator[tuple]:
        qid = next_query_id("GP")
        prompt = (
            "<s>[INST] Answer the user's question directly, clearly, and accurately. "
            "This is a general-purpose answer and is not grounded in the AUM source "
            "collections. Do not claim an AUM source, citation, policy, or factual "
            "basis unless the user supplied it. If you are uncertain, say so.\n\n"
            f"User question: {question}\n[/INST]"
        )
        for partial in generate_streaming(
            prompt, self.runtime.llm_tok, self.runtime.llm_model, query_id=qid, max_new_tokens=350,
        ):
            yield partial, [], "", qid
