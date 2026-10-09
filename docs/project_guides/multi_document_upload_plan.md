# Multi-document upload — plan, implementation, and real findings

**Date:** 2026-10-09
**Status:** DONE and verified live against the real server/model (this document)
**Origin:** direct user instruction: "Go ahead, build the multi-file input... Look into the attached file, choose which changes you want to do... Evaluate whether our current pipeline could handle it or silently gets crashed. Sometimes, too many filters can give 0 input." The attached file (`AUM_CHATBOT_QUIZ_RELIABILITY_PERSISTENCE_MULTIFORMAT_PLAN_2026-10-08.md`) is a much larger proposal (persistence, resumable jobs, multi-format document adapters); this document covers only the slice actually asked for this pass -- multi-file upload -- plus what that work's own risk evaluation surfaced. The rest of that larger plan is logged as future tasks, not built here (see Section 5).
**Tracking:** `workspace.md` TASK 56, Entry 066.

## 1. What was already there vs. what was missing

Checked before writing anything, not assumed:

- **Backend was already multi-document-ready.** `document_chat()` and `AssistantService.quiz()`'s document branch both already looped over `document_ids: list[str]`, and `DocumentStore` has no per-session document count limit.
- **The Electron client was not.** `desktop/client/renderer.js` had a literal comment -- *"One PDF per chat for this first pass"* -- and uploading a file did `activeChat.documentIds = [metadata.document_id]`, which overwrites rather than appends. The attachment UI only ever rendered a single chip.

So the actual gap was entirely client-side: a single-file `<input>`, overwrite-not-append state, and a one-chip UI. That's what "build the multi-file input" meant concretely.

## 2. What was built

- `desktop/client/index.html`: `documentFileInput` gained the `multiple` attribute; the static single `#attachmentChip`/`#removeAttachmentBtn` pair was replaced with an empty `#attachmentRow` container, populated dynamically.
- `desktop/client/renderer.js`:
  - `MAX_ATTACHED_DOCUMENTS = 5` -- a deliberate, modest cap, not derived from a measured limit. Justification: `AssistantService.quiz()`'s document branch pools up to 6 chunks *per* document before picking its best 6 overall, and `document_chat()` runs its own per-document FAISS search per attached file -- more documents is real additional retrieval work per question, not free. Revisit with real multi-document latency numbers if 5 turns out wrong in either direction.
  - `documentFileInput`'s change handler now iterates every selected file (bounded by remaining room under the cap), uploads each via the same existing single-file `/api/documents` call (looped, not batched -- the endpoint's contract didn't need to change), and **appends** to `documentIds`/`attachedDocuments` instead of overwriting. One failed upload in a multi-file selection doesn't abort the others.
  - `renderAttachmentRow()` now renders one chip per attached document, each with its own remove button (`removeAttachedDocument(documentId)`), instead of a single static pair.
- `desktop/client/styles.css`: `.attachment-row` gained `flex-wrap: wrap`; a new `.attachment-chip-group` class pairs each chip with its own remove button.

## 3. "Evaluate whether the pipeline could handle it or silently gets crashed" -- real findings, not a guess

Tested live against the real server with three real uploaded PDFs in one session (one genuinely relevant to the test question, two deliberately irrelevant filler documents about parking and dining services) before writing any fix, then again after. Two real, previously-latent bugs surfaced -- neither is a crash; both are the "silent wrong answer" failure mode that's worse than a crash because nothing signals it went wrong.

### 3.1 Confirmed bug: quiz's document branch could silently drop entire documents

`AssistantService.quiz()`'s document-source branch concatenated each attached document's own (already internally sorted) hit list and then sliced `[:6]` -- **without a final sort across documents first**. Whichever document was listed first, if it alone returned 6 hits (common for any document with enough content), completely crowded out every other attached document's content from the quiz, silently, with no error. `document_chat()` already did this correctly (sorted before truncating); only `quiz()`'s document branch had the bug.

**Fixed**: added the same `hits.sort(key=lambda h: h["score"], reverse=True)` before truncating, in `core/assistant_service.py`. Regression test added (`tests/test_quiz.py::test_quiz_document_mode_uses_best_scoring_chunks_across_all_attached_documents`) that attaches a low-relevance document with six of its own hits *first* and a single high-relevance document *second*, and confirms the high-relevance content still reaches the model.

### 3.2 Confirmed bug, found live: `document_chat()` cited documents it never used

With all three test documents attached and a question only the relevant one could answer, the evidence gate correctly passed (the top score, 0.482, cleared both the absolute floor and the relative threshold against the pooled candidates -- see Section 4 on the "too many filters" question specifically). But the real generated answer ended with: *"(doc_pertinent.pdf, p.1) (doc_filler1.pdf document, p1) (doc_filler2.pdf Section 1: Meal Plan options, p1)"* -- citing the parking and dining-services documents for a capstone-grading fact they have nothing to do with.

**Root cause**: `document_chat()` pools the top 5 hits across *all* attached documents, but built both the model's context (`ctx`) and the citation string (`cits`) from the full pooled top-5 regardless of each individual hit's own score -- so low-scoring filler hits from unrelated documents rode along into the prompt purely because they happened to be among the top 5 by rank, even though their actual relevance (score 0.151, 0.124, 0.036) was far below the 0.482 of the real answer and well under the document evidence floor (0.35). With a single attached document this was invisible (every hit necessarily came from the one document); with multiple documents it produces visibly wrong-looking citations.

**Fixed**: both `ctx` and `cits` are now built only from hits that individually clear `DOCUMENT_EVIDENCE_SCORE_FLOOR`, not just the pooled top-5 by rank. The top hit always clears this by construction (that's what made the evidence gate pass in the first place), so the filtered list can never be empty. Regression test added (`tests/test_document_qa.py::test_document_chat_with_multiple_documents_does_not_cite_irrelevant_ones`). **Re-verified live after the fix, same three real documents, same question**: the answer now cites only the relevant document, with no mention of the filler ones -- and, as a side effect, the stated percentages became fully accurate too (the model was evidently also distracted by the filler content it no longer sees).

### 3.3 The user's specific caution: "too many filters can give 0 input"

Tested for this directly rather than assuming it either happens or doesn't. In the live 3-document test above, the real score gap (0.482 vs. 0.151/0.124/0.036) was wide enough that the relative-threshold evidence gate passed without issue -- adding two irrelevant documents did **not**, in this test, dilute the mean/std enough to reject a genuinely strong top match.

This is a real, measured data point, not a guarantee. The gate's statistic (`top >= mean + SIGMA*std` over the pooled top-5) means the risk is real in principle whenever the irrelevant documents' chunks score only moderately lower than the relevant one, rather than clearly lower -- the closer the "noise" scores sit to the "signal" score, the more the threshold calculation tightens and the more plausible an outright rejection becomes, even when the right chunk was retrieved at rank 1. This exact failure mode was already confirmed once this session in a different context (TASK 54's Housing follow-up finding: a correctly-retrieved top candidate rejected because the surrounding candidate pool pulled the relative threshold up). `MAX_ATTACHED_DOCUMENTS = 5` bounds how bad this can get, but does not eliminate the underlying risk.

**Not fixed, logged as an open risk**: calibrating or redesigning the evidence gate specifically for the multi-document case (e.g., a per-document floor check before pooling, instead of one threshold over the merged pool) would need real measurement across many more document/topic combinations than one live test provides -- consistent with this project's standing rule against tuning thresholds from anecdotal examples. Flagged in `workspace.md` for whoever picks up TASK 49/51-adjacent evaluation work next.

### 3.4 Found and fixed along the way, not originally in scope: raw prompt/output logging

Reading the uploaded plan's own Section 2 claim ("`generate_streaming()` logs the first 500 characters of the assembled prompt... must be fixed") led to re-checking `pipeline/answer.py` directly rather than trusting the claim or dismissing it. It was accurate, and it was worse than a one-line note suggested: `generate_streaming()` -- the one function every pipeline (COS/Housing/Document/General/Quiz) funnels through -- logged the real first 500 characters of the assembled prompt and the real first 600 characters of the model's output, unconditionally, for every single generation call. Task 44 (earlier this session) fixed several other raw-text logging call sites but never touched this shared one, so uploaded document content and Housing policy text have been reaching the plaintext application log on every query this whole time, not just on failure.

**Fixed**: both lines now log `len=...` and a non-reversible content fingerprint (`pipeline.memory.query_fingerprint`, the same helper Task 44 introduced) instead of the real text. Re-verified live: a fresh Housing query's log shows `== LLM PROMPT == len=2718 fp=b04209adce` and `== LLM OUTPUT == len=1406 fp=f4f14b9e7d` -- no real content, still useful for correlating repeated identical prompts/outputs across log lines.

## 4. Verification

- Unit tests: `tests/test_quiz.py` (merge-sort fix) and `tests/test_document_qa.py` (citation/context filtering fix), both new, both passing. Full regression suite: 86/88 (the 2 failures are the same pre-existing harness artifacts logged in every entry this session, unrelated to this work).
- `renderer.js` syntax-checked after every edit via the project's bundled Electron binary run as plain Node (`ELECTRON_RUN_AS_NODE=1`) -- no system Node available in this environment.
- Live, end-to-end, real server + real Mistral model + real uploaded PDFs: confirmed the quiz merge-sort fix via a targeted unit test (mocked retrieval, since reproducing a 6-chunk-per-document scenario with tiny real test PDFs wasn't practical); confirmed the citation fix by reproducing the exact failure with three real documents, then re-running the identical request after the fix and seeing the wrong citations disappear.

## 5. Explicitly not built this pass

The uploaded plan's other phases -- interactive quiz persistence across Electron/app restarts, a durable backend quiz/session store (SQLite), resumable generation jobs, and multi-format document adapters (TXT/MD/DOCX/PPTX/CSV/XLSX) -- are real, well-reasoned proposals but a much larger initiative than "multi-file input," and were not started. Logged in `workspace.md` as a condensed set of future tasks pointing back at the uploaded plan document for full detail, rather than re-specifying all of it here. The plan's own Section 7 note ("add multiple-file attachment support only after one-document recovery is reliable") was read and is worth the project owner knowing about directly: it recommends the opposite sequencing from what was actually done this entry, on the reasoning that persistence should come first. The user's own explicit instruction ("go ahead, build the multi-file input") was followed as given rather than re-litigated.
