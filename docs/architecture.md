# AUM Chatbot — Architecture

Snapshot as of 2026-08-18, describing `/home/gh204/Desktop/aum_chatbot/` — the
reorganised, current codebase (see `workspace.md` TASK 11). The original flat
`backend.py` / `api_server.py` on the Desktop are historical; this document
describes the modular version everyone should build on going forward.

This is written so someone with no prior context on this project can read it
top to bottom and understand how a message travels from a user to an answer,
what each file is responsible for, and where the rough edges are.

---

## 1. What this system does

A two-domain question-answering chatbot for Auburn University Montgomery:

- **COS** — undergraduate research symposium projects (mentors, presenters,
  departments, abstracts, papers).
- **Housing** — the AUM Housing and Community Standards policy document.

Two client surfaces talk to the same backend logic:

- A **Gradio web UI** (`main.py` → `server/gradio_ui.py`), with one tab per
  domain plus a "Pipeline Inspector" debug tab.
- An **Electron desktop app** (`desktop/client/`), which talks over HTTP to a
  **FastAPI server** (`server/api_server.py`) — this is the one with an
  "auto" topic mode that has to guess which domain a question belongs to,
  since there's no tab to click.

The core design rule, carried through every rewrite so far: **the LLM never
invents facts.** Titles, years, departments, presenters, mentors, and
citations are always inserted by template code from the retrieved metadata.
The LLM (Mistral-7B-Instruct-v0.3) is only ever asked to paraphrase an
abstract into plain English, or to write one grounded paragraph from Housing
policy text it's given verbatim — never to generate names, dates, or facts
on its own.

---

## 2. Project layout

```
aum_chatbot/
├── config.py              # every path/model-id/threshold constant, one place
├── main.py                # load_everything() + standalone Gradio launch (<=80 lines)
├── models/
│   └── loader.py          # load_embedder(), load_reranker(), load_llm(), warmup()
├── pipeline/
│   ├── memory.py          # logger setup, FlushFileHandler, format_history() (stub)
│   ├── retrieval.py       # data loading, BM25+FAISS+RRF retrieval, name index
│   ├── classifier.py      # classify_query(), classify_topic(), QueryTrace/SESSION_TRACES
│   └── answer.py          # templates, LLM prompts, generate_streaming(), render_trace_md()
├── server/
│   ├── gradio_ui.py       # cos_chat()/housing_chat(), the Gradio Blocks UI
│   └── api_server.py      # FastAPI app: /api/health, /api/ask, /api/ask/stream
├── desktop/client/        # Canonical Electron app source + packaged build — talks to api_server.py
├── data/                  # cos_data.jsonl, Housing PDF, emb_store/ (cached indices), scratch/
├── logs/                  # one session_*.log per process start
├── archive/               # historical copies of backend.py, the notebook, laptop fork
└── tests/                 # empty — no automated tests exist yet
```

---

## 3. Request flow

### 3a. Gradio UI path

```
Browser tab (COS or Housing)
  → msg_box.submit / send_btn.click
  → _submit() [server/gradio_ui.py] — a generator, so Gradio streams tokens live
  → cos_chat() or housing_chat() [server/gradio_ui.py]
  → classify_query() / retrieve_cos_rrf() / build_cos_answer_streaming()  (or the Housing equivalents)
  → yields (partial_text, pending_cands, debug_md) repeatedly
  → Gradio updates the chat bubble on every yield
```

Each tab owns its own `gr.State` for `pending_cands` (the numbered-list
selection flow — "reply with a number"). Tabs are completely independent:
there is no cross-tab routing, so a message typed in the COS tab can never
reach `housing_chat`.

### 3b. HTTP API path (what the Electron client actually uses)

```
POST /api/ask  { question, topic: "cos"|"housing"|"auto", session_id }
  → _resolve_topic(req)  [server/api_server.py]
      topic != "auto"                          → use it directly
      pending COS selection + bare number reply → "cos" (skips the router entirely)
      person-name in the question               → "cos" (classify_query()'s NAME_INDEX match, zero LLM cost)
      otherwise                                  → Mistral intent router (classify_topic(), TASK 12)
                                                    falls back to an embedding+keyword heuristic
                                                    if the LLM output doesn't parse cleanly
  → _run_chat_sync(topic, question, session_id)
  → gradio_ui.cos_chat(...) or gradio_ui.housing_chat(...)  — the SAME generator functions the Gradio UI uses
  → drains the generator fully, returns the final answer as one JSON response
```

`/api/ask/stream` exists (Server-Sent Events, same generators, sends deltas
as they arrive) but **the Electron client does not use it** — `renderer.js`
only calls `/api/ask`. See §6 for why that matters.

`_session_pending` (a plain in-memory dict keyed by `session_id`) is the
HTTP equivalent of Gradio's `gr.State` — it's what lets "reply with 3" work
over the API, since there's no browser tab to hold state in.

---

## 4. COS query pipeline

1. **`classify_query()`** (`pipeline/classifier.py`) — decides one of:
   - `TYPE_PERSON`: any token/phrase in the query matches `NAME_INDEX`
     (built once from every mentor/presenter/author in the corpus). This
     check runs *last* and overrides everything else — a person match is
     always trusted.
   - `TYPE_BROAD`: a year, a department keyword, or "list"-shaped phrasing.
   - `TYPE_TOPIC`: the default — a focused, single-subject question.
2. **`retrieve_cos_rrf()`** (`pipeline/retrieval.py`):
   - Person path: exact metadata scan (`_exact_person_cands`) over mentor +
     lead_presenters first, falling back to other_authors only if that finds
     nothing — then reranked.
   - Broad/Topic path: BM25 (`build_bm25`/`bm25_search`) and FAISS
     (`cos_index`, `all-MiniLM-L6-v2` embeddings) run independently, fused
     with Reciprocal Rank Fusion, then reranked with a CrossEncoder
     (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
3. **`build_cos_answer_streaming()`** (`pipeline/answer.py`) picks the
   response shape:
   - No candidates → deterministic "not found" message.
   - A previously-shown numbered item was selected → `_explain_one_project()`
     (template header/footer + one LLM-paraphrased paragraph of the abstract).
   - `TYPE_PERSON` → `_format_person_answer()`, template only, **zero LLM
     calls** — this path cannot hallucinate.
   - Broad, or the top rerank score isn't clearly ahead of the pack
     (`_relative_threshold`, `SIGMA=0.8` standard-deviations above the mean)
     → `_format_chooser_list()`, a numbered list, template only.
   - A strong single topic hit → `_explain_one_project()` on the top result.

## 5. Housing query pipeline

Simpler and with one notable gap (see §6): `retrieve_housing_logged()` always
returns the top 4 FAISS hits against the 181 chunked sections of the policy
PDF — there's no relevance threshold. `build_housing_answer_streaming()`
only short-circuits to "not found" when literally zero hits come back
(essentially never, since retrieval always returns up to 4). Otherwise the
LLM is given the retrieved chunk text verbatim inside `<context>` tags and
told to answer only from it, ending with citations.

## 6. Models

`models/loader.py` — pure functions, no import-time side effects (a
deliberate change from the original flat script, which loaded everything the
moment it was imported):

| Function | Loads | Notes |
|---|---|---|
| `load_embedder()` | `sentence-transformers/all-MiniLM-L6-v2` | |
| `load_reranker()` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | |
| `load_llm()` | `mistralai/Mistral-7B-Instruct-v0.3` | float16/GPU or bfloat16/CPU |
| `warmup()` | — | one throwaway generate() call so the first real request isn't slow |

All four are called once, in that order, by `main.load_everything()`, which
both `main.py`'s own entrypoint and `server/api_server.py`'s FastAPI startup
event call (idempotent — a `_loaded` guard prevents double-loading if both
happen to run in the same process, though in practice they're separate
processes).

## 7. Data & caching

- `data/cos_data.jsonl` (94 records) → embedded once, cached as a FAISS
  `IndexFlatIP` + a BM25 corpus. A manifest (`embeddings_manifest.json`,
  hash of the corpus text) gates whether the cache is trusted or rebuilt.
- `data/AUM-Housing-Community-Standards.pdf` → chunked by detected section
  headings (`extract_housing_chunks`), embedded, cached the same way.
- `data/scratch/` is a fast local copy of `data/emb_store/` that the app
  actually reads from at runtime (`scratch_copy()`); `emb_store/` is the
  "source of truth" copy that gets written back to when the cache rebuilds.

## 8. Session / state model

- **Gradio UI**: `gr.State` per tab, in the Gradio server process's memory.
  Gone when the browser tab closes or the server restarts.
- **HTTP API**: `_session_pending` dict in `server/api_server.py`, keyed by
  the client-supplied `session_id`. In-memory, single-process — lost on
  restart, and would silently give wrong answers if this process ever ran
  behind a multi-worker deployment (each worker would have its own dict).
- **`SESSION_TRACES`** (`pipeline/classifier.py`): every query's full
  pipeline trace (classification, retrieval scores, the exact prompt sent
  to the LLM, raw output) is kept in memory for the Pipeline Inspector tab.
  It is *never* cleared — grows for the lifetime of the process.

## 9. Intent routing (TASK 12)

For `topic="auto"` requests, in order of trust:

1. **Pending COS selection** — a bare number reply when `_session_pending`
   already holds a list → `"cos"`, zero LLM cost.
2. **Person-name override** — any `NAME_INDEX` match → `"cos"`, zero LLM
   cost (NAME_INDEX only contains COS people, so this is always safe).
3. **Mistral router** (`_classify_topic_llm`) — a 5-token-capped, greedy
   `[INST]` prompt asking for exactly `"cos"` or `"housing"`.
4. **Heuristic fallback** (`_classify_topic_heuristic`) — the original
   embedding-similarity + keyword-bias logic — used only if the LLM call
   raises or its output doesn't parse to exactly one of the two words.

The router itself is cheap (observed: same-second as the next pipeline
stage in the logs). It is *not* what makes requests slow — see §10.

---

## 10. Known issues / where this needs improvement

Ranked roughly by user-facing impact:

1. **RESOLVED, with a surprise — generation speed.** An earlier version of
   this doc reported 124-134 seconds per non-streaming answer and blamed
   missing flash-attention (Mistral-7B assumed to be running naive "eager"
   attention). TASK 13 investigated this properly and found both parts
   wrong: `transformers` was already auto-selecting PyTorch's built-in
   `sdpa` (a fused, hardware-optimized kernel), not eager attention — and a
   controlled A/B test (isolated benchmark, long-context benchmark, and the
   real HTTP API, all three) showed `flash_attention_2` performs
   *identically* to `sdpa` on this hardware. The original 124-134s figure
   turned out not to be reproducible at all: the same query now measures
   ~10 seconds regardless of attention implementation. The leading
   (unconfirmed) explanation is residual system state from the concurrency
   bug in #3 below, deliberately triggered earlier in the same testing
   session that produced the 124-134s measurement. flash_attention_2 is
   still wired in (`models/loader.py`, CUDA branch only) since it's correct
   and harmless, just not the fix it looked like. Full investigation:
   `issues_fixes.md` #11, `workspace.md` Entry 017.

2. **The Electron desktop client only calls `/api/ask` (non-streaming),
   never `/api/ask/stream`** (verified in `renderer.js`). Real users still
   wait the full generation time per message with a static "pending"
   indicator and zero progressive feedback — even though the underlying
   pipeline (and the Gradio UI) already supports true token-by-token
   streaming, and even though that wait is now ~10s instead of 124s+.
   Switching the Electron client to consume `/api/ask/stream` (Server-Sent
   Events) is still probably the single highest-value client-side change
   available, and is now also tangled up with TASK 18 (which backend the
   client should point at) — see `workspace.md`.

3. **RESOLVED — lock around `model.generate()`.** `server/api_server.py`
   previously had no lock; two overlapping requests could both call
   `generate()` on the same GPU model instance concurrently, directly
   observed to stall a request for 90+ seconds with the GPU pinned at
   19.5/20.5GB and 83% utilization. TASK 14 added a process-wide
   `asyncio.Lock()` (`_generate_lock`) covering every code path that
   reaches `generate()` — both the answer-generation calls and the Mistral
   router's classification call (`_resolve_topic()`, added by TASK 12
   after this gap was originally found). Verified by reproducing the
   original bug directly: two genuinely concurrent requests now correctly
   queue (10.5s then 23.9s total, i.e. the second waited its turn) instead
   of racing or stalling. One implementation detail worth preserving if
   this code changes: the lock in the streaming endpoint has to be
   acquired *inside* the `token_stream()` async generator (spanning its
   `yield`s), not in `ask_stream()` itself, which returns before any
   generation happens. Full detail: `issues_fixes.md` #9, `workspace.md`
   Entry 018.

4. **GPU VRAM headroom is tight** (~5GB free after the Mistral-7B upgrade,
   down from ~13GB with Phi-3-mini). Less of a concern now that #3 is
   fixed (no more concurrent generate() calls piling up memory pressure),
   but still worth watching under heavier load than this project has
   tested.

5. **No relevance threshold on the Housing path.** COS gates broad/weak
   results behind a numbered list instead of a confident answer
   (`_relative_threshold`, §4); Housing has no equivalent — any query, even
   one entirely unrelated to housing policy, retrieves the 4 nearest chunks
   and the LLM is asked to answer from them. There's no way for Housing to
   say "I don't have information on that."

6. **Five-way routing is implemented (TASK 16 and TASK 26).** The classifier
   now distinguishes COS, Housing, general AUM, general-purpose, and declined
   requests. COS/Housing use grounded retrieval. General-AUM stays
   deterministic because no authoritative AUM-wide collection is connected.
   General-purpose requests use the already-loaded Mistral model with a
   separate prompt, no retrieval, and no AUM-source or citation claim.

7. **In-memory state doesn't survive a restart or scale past one process**
   — `_session_pending` and `SESSION_TRACES` (§8). Fine for the current
   single-process dev deployment; would need Redis/a DB if this ever runs
   with more than one worker.

8. **No structured `sources` field.** `ChatResponse.sources` always returns
   `[]` — citations exist but are embedded in the answer text, not
   machine-parseable. A client that wanted to show "sources" separately
   from the answer prose can't, today.

9. **`classify_query()`'s person/broad/topic detection is still fully
   regex + corpus-lookup**, not LLM-based. This was a deliberate scope
   decision in TASK 12 (the user explicitly chose *not* to extend the
   Mistral router to this layer), not an oversight — but it's the natural
   next candidate if regex/corpus-match edge cases start showing up.

10. **No automated tests** — `tests/` is empty. Every verification so far
    has been manual (live launches, live HTTP requests, log inspection).

11. **`config.MIN_FREE_RAM_FOR_LLM_GB` is a placeholder (`None`)** — no such
    constant or RAM-pressure guard ever existed in the original codebase;
    it was added to `config.py` only because a later task asked for it by
    name. There is no actual RAM-checking logic anywhere in this system.

12. **The Electron client's packaged build isn't automated.**
    `electron-builder`'s `npm run dist` does not refresh
    `dist/squashfs-root/` (the directory the desktop launcher actually
    runs) — that has to be re-extracted from the fresh AppImage by hand
    every time (`--appimage-extract`). Easy to forget, and was in fact
    the reason TASK 8 was blocked for a while.

---

## 11. What could be added next

Two of §10's original gaps (flash-attention investigation, the generation
lock) are now resolved — see items 1 and 3 above. Remaining gaps worth
closing: an Electron streaming switch, a Housing relevance threshold, and
a connected authoritative general-AUM collection for TASK 16. Other directions worth
considering:

- **Knowledge Graph (TASK 15, queued as `pipeline/graph.py`)** — the current
  retrieval is flat (BM25/FAISS over independent documents); a graph over
  mentors, presenters, departments, and projects would let the system answer
  relationship questions the current pipeline can't ("who has Kursun
  co-authored with", "what other projects came out of the same department
  in the same year") without relying on the LLM to infer connections it
  wasn't shown.
- **Structured sources**: parse `trace.candidates` / retrieval hits into
  `ChatResponse.sources` instead of leaving it always empty.
- **Shared/persistent session store** (Redis or a small DB) so
  `_session_pending` survives restarts and works correctly if this is ever
  deployed with more than one worker process.
- **A real test suite** — even a handful of pytest cases around
  `classify_query()`, the retrieval RRF math, and the selection-flow regex
  would catch regressions that today are only caught by manual live testing.
- **Automate the Electron packaging step** (rebuild + re-extract
  `squashfs-root` in one script) so this class of bug can't recur.
- **A RAM/VRAM pressure guard** before loading or generating, now that GPU
  headroom is measurably tight — could use `config.MIN_FREE_RAM_FOR_LLM_GB`
  for real instead of leaving it `None`.

---

## 12. How to run this

```bash
conda activate aum_env
cd /home/gh204/Desktop/aum_chatbot

# Gradio UI (http://127.0.0.1:7860)
python main.py

# HTTP API for the Electron client (picks a free port, prints it)
python server/api_server.py
```

See `workspace.md` for the full task-by-task history of how this codebase
got here, including every bug found and fixed along the way, and for
current task status beyond what's captured here (last updated after TASK
14, 2026-08-24).
