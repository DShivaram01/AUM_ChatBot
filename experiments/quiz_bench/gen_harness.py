"""Shared generation harness for the quiz experiments (run in the `qzgen`
conda env -- a clone of production aum_env -- during an approved maintenance
window, with the production server stopped so Mistral can use the GPU).

- MistralLM: loads Mistral exactly like models/loader.load_llm (fp16,
  device_map="auto") and generates with the production repetition guard.
  `generate(..., truncate=True)` reproduces production's 3200-char prompt cap;
  experiment arms call it with truncate=False and their own token budgets.
- Evidence: deterministic offline ingest (same chunk ids as the server) with
  the production embedder, plus dense retrieval.
- score_quiz: shared scoring for every arm.

Generated questions quote the benchmark document: write them only under
fixture.bench_dir(); commit metrics only.
"""

import json
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from experiments.quiz_bench.fixture import gold, norm, paper_bytes

EMBEDDED_QUIZ_HEADING = "quiz benchmark questions"


@dataclass
class CallLog:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0
    truncated_prompts: int = 0
    records: list = field(default_factory=list)


class MistralLM:
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import config
        t = time.perf_counter()
        self.tok = AutoTokenizer.from_pretrained(config.LLM_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(config.LLM_ID, device_map="auto", torch_dtype=torch.float16)
        self.model.eval()
        self.load_seconds = time.perf_counter() - t
        self.log = CallLog()
        self.lock = threading.Lock()

    def ntok(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def generate(self, prompt: str, max_new_tokens: int, avoid_texts=None, truncate: bool = False,
                 prefix_allowed_tokens_fn=None, logits_processors=(), in_answer_guard: bool = True) -> str:
        """in_answer_guard=False disables the production in-answer 4-gram ban and
        repetition penalty (still bans re-writing avoid_texts). Multi-question
        JSON must repeat its keys ("options": [ ...) once per question, which the
        in-answer ban forbids; batch/structured arms record this setting."""
        import torch
        from transformers import LogitsProcessorList
        from pipeline.answer import GeneratedOnlyRepetitionGuard, _PROMPT_MAX_CHARS, _truncate_prompt
        if truncate and len(prompt) > _PROMPT_MAX_CHARS:
            prompt = _truncate_prompt(prompt, _PROMPT_MAX_CHARS)
            self.log.truncated_prompts += 1
        inputs = self.tok(prompt, return_tensors="pt", truncation=True, max_length=8192).to(self.model.device)
        n_in = int(inputs["input_ids"].shape[1])
        guard = GeneratedOnlyRepetitionGuard(
            n_in, banned_sequences=[self.tok(x, add_special_tokens=False)["input_ids"] for x in (avoid_texts or [])],
            **({} if in_answer_guard else {"ngram": 10**6, "penalty": 1.0}))
        t = time.perf_counter()
        with self.lock, torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                logits_processor=LogitsProcessorList([guard, *logits_processors]),
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                eos_token_id=self.tok.eos_token_id, pad_token_id=self.tok.eos_token_id)
        secs = time.perf_counter() - t
        new = out[0][n_in:]
        text = self.tok.decode(new, skip_special_tokens=True).strip()
        self.log.calls += 1
        self.log.input_tokens += n_in
        self.log.output_tokens += int(new.shape[0])
        self.log.seconds += secs
        self.log.records.append({"in": n_in, "out": int(new.shape[0]), "s": round(secs, 2)})
        return text

    def production_streaming_shim(self):
        """A drop-in for pipeline.answer.generate_streaming so production
        pipeline.quiz.generate_quiz runs unchanged against this model and is
        counted in self.log (it truncates exactly like production)."""
        def shim(prompt, tokenizer, model, query_id="Q", max_new_tokens=300, trace=None, avoid_texts=None):
            yield self.generate(prompt, max_new_tokens, avoid_texts=avoid_texts, truncate=True)
        return shim


@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    import config
    return SentenceTransformer(config.EMBED_MODEL_ID, device="cpu")


@lru_cache(maxsize=1)
def document():
    """(chunks, chunk_ids, matrix): production ingest of the fixture document."""
    from core.document_service import DocumentService
    svc = DocumentService(embedder())
    doc = svc.ingest("benchmark.pdf", paper_bytes(), "eval-qz-offline")
    chunks = [{"chunk_id": c.chunk_id, "text": c.text, "page": c.page, "page_end": c.page_end,
               "heading_path": list(c.heading_path)} for c in doc.chunks]
    return chunks, doc.embeddings


def retrieve(query: str, k: int) -> list[tuple[dict, float]]:
    chunks, m = document()
    q = embedder().encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
    s = m @ q
    return [(chunks[i], float(s[i])) for i in np.argsort(-s)[:k]]


def chapter_of(chunk: dict) -> str:
    for h in chunk["heading_path"]:
        m = re.match(r"chapter\s+(\d+)", norm(h))
        if m:
            return f"ch{int(m.group(1))}"
    return "front"


def is_embedded_quiz_box(chunk: dict) -> bool:
    return any(EMBEDDED_QUIZ_HEADING in norm(h) for h in chunk["heading_path"])


@lru_cache(maxsize=1)
def embedded_questions() -> list[str]:
    chunks, _ = document()
    out = []
    for c in chunks:
        if is_embedded_quiz_box(c):
            out += [q.strip() for q in re.findall(r"^\s*\d+\.\s*(.+)$", c["text"], re.M)]
    return out


def score_quiz(questions: list[dict], request: dict, topic_terms: list[str] | None = None) -> dict:
    """questions: [{"question","options","correct_index","evidence_ids"}]."""
    from sentence_transformers import util
    g = gold()
    chunks, _ = document()
    by_id = {c["chunk_id"]: c for c in chunks}
    emb = embedder()
    out = {"n_questions": len(questions)}
    if not questions:
        return out
    texts = [q["question"] for q in questions]
    E = emb.encode(texts)
    sims = util.cos_sim(E, E)
    out["near_duplicate_pairs"] = sum(float(sims[i][j]) >= 0.85 for i in range(len(texts)) for j in range(i + 1, len(texts)))
    answers = [norm(q["options"][q["correct_index"]]) for q in questions]
    out["repeated_correct_answers"] = len(answers) - len(set(answers))
    emb_q = embedded_questions()
    if emb_q:
        S = util.cos_sim(E, emb.encode(emb_q))
        out["copied_embedded_questions"] = int(sum(float(S[i].max()) >= 0.85 for i in range(len(texts))))
    supported, touched, pages, chapters, uncited = 0, set(), set(), set(), 0
    for q in questions:
        ev_chunks = [by_id[e] for e in q.get("evidence_ids", []) if e in by_id]
        if not ev_chunks:
            uncited += 1
        ev = norm(" ".join(c["text"] for c in ev_chunks))
        pages |= {c["page"] for c in ev_chunks}
        chapters |= {chapter_of(c) for c in ev_chunks}
        ans = norm(q["options"][q["correct_index"]])
        words = [w for w in re.findall(r"[a-z0-9().+*/^|=-]+", ans) if len(w) > 2 or any(ch.isdigit() for ch in w)]
        supported += bool(ev) and bool(words) and sum(w in ev for w in words) / len(words) >= 0.5
        qa = norm(q["question"] + " " + q["options"][q["correct_index"]])
        for c in g["concepts"]:
            if any(norm(a) in ev for a in c["anchors"]) and any(w in qa for w in norm(c["answer"]).split() if len(w) > 3):
                touched.add(c["id"])
    out.update(lexically_supported=supported, uncited_questions=uncited, gold_concepts_touched=sorted(touched),
               n_gold_concepts=len(touched), cited_pages=sorted(pages), cited_chapters=sorted(chapters))
    if request.get("concepts"):
        out["target_concepts_touched"] = f"{len(touched & set(request['concepts']))}/{len(request['concepts'])}"
    if topic_terms:
        out["questions_mentioning_topic"] = sum(any(t in norm(q["question"]) for t in topic_terms) for q in questions)
    return out


def gpu_peak_gib():
    import torch
    return round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None


def save_private(name: str, payload) -> str:
    from experiments.quiz_bench.fixture import bench_dir
    path = bench_dir() / f"gen_outputs_{name}.json"
    path.write_text(json.dumps(payload, indent=1))
    path.chmod(0o600)
    return str(path)
