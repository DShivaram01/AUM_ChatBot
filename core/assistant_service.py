"""Service-owned COS, Housing, routing, and pretrained answer execution."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator

from datetime import datetime

import config
from core.document_service import DocumentService
from core.runtime_manager import RuntimeManager
from pipeline.answer import (
    HOUSING_EVIDENCE_SCORE_FLOOR,
    SIGMA,
    _relative_threshold,
    _trunc,
    build_cos_answer_streaming,
    build_housing_answer_streaming,
    generate_streaming,
    render_trace_md,
)
from pipeline.classifier import (
    QueryTrace,
    RetrievalCandidate,
    classify_query,
    classify_topic,
    next_query_id,
    store_trace,
)
from pipeline.memory import logger
from pipeline.retrieval import retrieve_cos_rrf, retrieve_housing_logged

_DOCUMENT_INSUFFICIENT_EVIDENCE = (
    "I could not find enough information in the uploaded document to answer that."
)


def _document_cite(hit: dict) -> str:
    start, end = hit["page"], hit["page_end"]
    page = str(start) if end in (None, start) else f"{start}-{end}"
    heading = " > ".join(hit["heading_path"]) if hit["heading_path"] else "document"
    return f'({hit["filename"]}, {heading}, p.{page})'


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

    def document_chat(
        self, question: str, document_ids: list[str], session_id: str | None,
    ) -> Iterator[tuple]:
        """Grounded Q&A restricted to the caller's own uploaded document(s).

        Mistral answers only from retrieved chunk evidence; insufficient
        evidence returns the fixed abstention message rather than silently
        falling back to pretrained knowledge. Traces never carry raw document
        text (only IDs/pages/scores) -- see the full_prompt redaction below.
        """
        qid = next_query_id("DOC")
        question = question.strip()
        if not question:
            yield "Please type a question about the uploaded document.", [], "", qid
            return
        if not document_ids:
            yield "Please attach a document before asking a document-grounded question.", [], "", qid
            return
        if not session_id:
            yield "A session is required to use uploaded documents.", [], "", qid
            return

        hits: list[dict] = []
        try:
            for document_id in document_ids:
                hits.extend(self.documents.retrieve_chunks(document_id, session_id, question))
        except (KeyError, PermissionError) as exc:
            yield str(exc), [], "", qid
            return
        hits.sort(key=lambda h: h["score"], reverse=True)
        hits = hits[:5]

        trace = QueryTrace(
            query_id=qid, query=question, tab="document",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            intent_type="document",
            routing_path=f"explicit document selection: {document_ids}",
        )
        for rank, hit in enumerate(hits, 1):
            page_label = f"p.{hit['page']}"
            if hit["page_end"] not in (None, hit["page"]):
                page_label = f"p.{hit['page']}-{hit['page_end']}"
            trace.candidates.append(RetrievalCandidate(
                idx=rank - 1, title=hit["filename"], mentor=page_label,
                year="", department="Uploaded document",
                faiss_score=hit["score"], faiss_rank=rank, final_rank=rank,
            ))
        trace.faiss_hits = len(trace.candidates)

        evidence_scores = [{"rerank": hit["score"]} for hit in hits]
        is_strong, top, mean, std = _relative_threshold(
            evidence_scores, absolute_floor=HOUSING_EVIDENCE_SCORE_FLOOR,
        )
        trace.threshold_top, trace.threshold_mean, trace.threshold_std = top, mean, std
        trace.threshold_cutoff = max(HOUSING_EVIDENCE_SCORE_FLOOR, mean + SIGMA * std)
        trace.threshold_passed = is_strong

        if not hits or not is_strong:
            trace.response_state = "not_found"
            trace.final_answer = _DOCUMENT_INSUFFICIENT_EVIDENCE
            trace.full_prompt = "<redacted: evidence gate rejected, no prompt generated>"
            store_trace(trace)
            logger.info(f"[{qid}] Document evidence gate rejected (top={top:.3f})")
            yield _DOCUMENT_INSUFFICIENT_EVIDENCE, [], "", qid
            return

        ctx = "\n\n---\n\n".join(_trunc(hit["text"], 500) for hit in hits)
        cits = "  ".join(_document_cite(hit) for hit in hits)
        prompt = (
            "<s>[INST] You are a document assistant. "
            "Use ONLY the document text inside <context> tags. "
            "Do NOT use outside knowledge. Do NOT invent facts. "
            "Write ONE paragraph. No bullet points. End with the citations.\n\n"
            f"Question: {question}\n\n"
            f"<context>\n{ctx}\n</context>\n\n"
            f"Write one paragraph answering the question. End with: {cits} "
            "[/INST]"
        )
        trace.response_state = "paragraph"
        full = ""
        for partial in generate_streaming(
            prompt, self.runtime.llm_tok, self.runtime.llm_model, qid, max_new_tokens=400,
        ):
            full = partial
            yield partial, [], "", qid
        trace.final_answer = full
        trace.full_prompt = (
            f"<redacted: document-grounded prompt, {len(hits)} chunk(s) from "
            f"document(s) {sorted(set(hit['document_id'] for hit in hits))}>"
        )
        store_trace(trace)
