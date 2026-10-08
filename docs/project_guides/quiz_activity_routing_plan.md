# Quiz as a cross-entry-point activity — design & implementation record

**Date:** 2026-10-08
**Status:** Phase 1-2 DONE and verified live (this document); Phase 3-4 NOT started (see Scope cuts below)
**Origin:** `AUM_CHATBOT_INTELLIGENT_QUIZ_ROUTING_AND_RELIABILITY_PLAN_2026-10-08.md` (uploaded design proposal, validated against the codebase before any of this was built) + the user's direct bug report that motivated it: "I wasnt able to generate quiz using quiz button... when i tried open ended mode and just asked a query to generate 5 mcqs on a subject, it gave me 5 questions with options but unstructured."
**Tracking:** `workspace.md` TASK 52 (the underlying generation bug, fixed first) and TASK 53 (this feature), Entries 062-063.

## 1. Problem

Two separate entry points existed for the same underlying capability:

- **Quiz button** → a modal with its own topic/count/source inputs → `POST /api/quiz` → `AssistantService.quiz()` → `pipeline/quiz.py`'s structured, validated, one-question-per-call generator.
- **Ordinary chat, Open-ended mode on** → `POST /api/ask/stream` with `topic: "general"` → `general_chat()` → a single free-text Mistral call with no structure, no validation, no retry.

A user typing "generate 5 MCQs on X" into the chat box got routed to the second path — unstructured prose formatted as if it were a quiz, not an actual quiz. The two paths never shared code.

## 2. Research findings (before any line of this was written)

- `pipeline/classifier.py`'s router has five labels (`cos`, `housing`, `general_aum`, `general`, `out_of_scope`) with no quiz concept at all, and no architectural hook for one -- confirming the uploaded plan's own framing that quiz is an *activity* orthogonal to topic/domain, not a sixth label.
- `AssistantService.quiz()` (`core/assistant_service.py`) already supports exactly three source modes -- `pretrained`, `housing`, `document` -- with no COS or combined-source retrieval path implemented anywhere in the codebase.
- `desktop/client/renderer.js`'s composer always calls `/api/ask/stream`; the Open-ended toggle only changes the `topic` field it sends (`"general"` vs. the active tab). The Quiz modal is a fully separate code path calling `/api/quiz` directly.
- `server/api_server.py`'s `_resolve_topic()` already has a precedent for exactly this kind of pre-routing check: explicit `document_ids` and an explicit non-`"auto"` topic are both decided *before* any domain classification runs. Quiz detection was added at the same point in the request lifecycle, for the same reason.
- `consumeSSE()` (client) already parses the SSE `done` event as arbitrary JSON and returns it to the caller -- extending that JSON's shape (rather than inventing a new SSE event type) was the minimal-footprint integration point.

## 3. What was implemented (Phase 1-2)

### 3.1 Activity intent detection — `pipeline/quiz_intent.py`

Deliberately rule-based (regex), not an LLM classifier. A false positive here silently hijacks an ordinary question into a multi-call quiz generation instead of answering it -- a mistake worth preventing with simple, auditable rules. `detect_quiz_intent(text, has_attached_document=False) -> ActivityIntent` recognizes:

- `(create|make|generate|build|write|give me|prepare|provide) [a/an/some] [N] (MCQs|multiple-choice questions|quiz(zes)|practice questions|test questions)`
- the same, with bare `questions` instead, **only** when a count number is present (`"10 questions"` triggers; `"questions about X"` alone does not -- too generic a word to trust without that extra signal)
- `quiz me [on|about] ...`

Topic is extracted via the connector word (`on`/`about`/`from`/`regarding`/`covering`/`for`/`using`) following the trigger phrase; count defaults to 5 and is clamped to `[1, 20]` (the same bounds `AssistantService.quiz()` already enforces).

Every positive and negative example named in the source plan (Sections 3, 6, 9) is a regression test in `tests/test_quiz_intent.py` and passes, including the ones that are easy to get wrong by accident: `"how do I create a quiz app"`, `"what time is my quiz"`, `"quiz schedule"`, `"summarize my quiz results"` all correctly do **not** trigger.

One real bug found and fixed during this: an early version used a Python conditional regex group (`(?(count)questions?)`) to make the bare-"questions" case count-gated. A conditional group with no explicit "no" branch matches an **empty string** when its condition is false, not a failure -- which silently made the entire quiz-noun requirement optional whenever no digit was present, and `"create a quiz app"` matched. Fixed by replacing it with three explicit, mutually exclusive alternatives (see the module's own comment for detail). Caught by the plan's own negative-example test, not discovered by luck.

### 3.2 Source resolution — `resolve_quiz_source()` (same file)

Mirrors the plan's Section 3 precedence, bounded to the three sources that actually have a retrieval implementation:

1. An attached document always wins (matches `_resolve_topic()`'s own existing rule for ordinary chat).
2. Otherwise, infer Housing from topic content (`_HOUSING_HINT_WORDS`, reused from `pipeline/classifier.py` rather than duplicated).
3. Otherwise, fall back to pretrained generation **only if** the request already opted into Open-ended mode (`req.topic == "general"`) -- the same signal `_resolve_topic()` already trusts for GENERAL's own opt-in gate. This is a direct reuse of this project's standing rule that open-ended/ungrounded generation is never an automatic fallback, applied to quiz instead of invented fresh.
4. Otherwise: return a message asking the user to attach a document, mention a housing topic, or turn on Open-ended mode. **Never guesses.**

COS is deliberately not resolved to anything here -- there is no COS retrieval path for quiz to call, so a topic that would ideally route to COS falls through to "ask for a source" rather than silently misrouting to Housing or fabricating COS grounding that doesn't exist.

### 3.3 Server wiring — `server/api_server.py`

`_try_quiz_from_chat(req)` runs the detection + resolution + (if resolved) the actual `AssistantService.quiz()` call, and is invoked from both `ask()` and `ask_stream()` **before** `_resolve_topic()` -- same position in the request lifecycle as `_resolve_topic()`'s own document/explicit-topic checks, for the same reason (quiz is a pre-routing activity decision, not a topic). `ChatResponse` gained `kind: "answer"|"quiz"`, `quiz: dict | None`, and `source_mode`, all optional/defaulted so existing callers are unaffected. The streaming endpoint emits one placeholder delta ("Generating your quiz…", since quiz generation makes several sequential model calls internally with no meaningful per-token stream of its own) and puts the full quiz in the `done` event's JSON rather than inventing a new SSE event type.

### 3.4 Client wiring — `desktop/client/renderer.js`

The composer's SSE completion handler checks `data.kind === 'quiz'`: if so, it removes the placeholder bot bubble and calls the **same** `renderQuizCard()` the Quiz button already uses, so a chat-originated quiz looks and behaves identically to a button-originated one (interactive options, "Check answers", explanations). `labelForTopic()` gained a label for the "ask for a source" case.

## 4. Verification

All of this is useless unverified against the real model -- every claim below was checked live against the running server (restarted on the fixed code) and the real Mistral-7B instance, not mocked, after a mocked-LLM unit/integration test pass first caught one real regex bug (3.1 above):

| Scenario | Request | Result |
|---|---|---|
| Open-ended + natural-language MCQ request (the user's own original report) | `"Generate 5 MCQs on computational biology"`, `topic: "general"` | Real structured 5-question quiz, `kind: "quiz"`, `source_mode: "pretrained"` |
| No source available | same question, `topic: "auto"`, no document | Clear guidance message, `topic_used: "quiz_unavailable"` -- no guess, no silent pretrained fallback |
| Housing inferred from topic | `"quiz me on the dorm guest policy"`, `topic: "auto"` | Real 5-question quiz grounded in Housing policy, `source_mode: "housing"` |
| Ordinary question mentioning "quiz" | `"What time is my quiz?"`, `topic: "general"` | Answered normally via `general_chat()` -- **not** hijacked into quiz generation |

Test suite: 18 new tests (`tests/test_quiz_intent.py`, `tests/test_quiz_chat_integration.py`), full regression sweep 62/64 (the 2 failures are pre-existing harness artifacts already logged in earlier `workspace.md` entries, unrelated to this change).

## 5. Scope cuts (explicit, not silent)

Everything below is from the source plan's Phases 2-4 and was **not** built in this pass. Each is a real gap, not an oversight:

- **Quiz follow-up detection** ("explain question 3", "make it harder"). Needs server-side quiz-state tracking per session that doesn't exist yet. A follow-up message today just falls through to ordinary chat routing -- a safe default (it answers conversationally rather than crashing or misfiring), but it won't reference the specific quiz.
- **COS quiz support.** No retrieval path exists; `resolve_quiz_source()` never routes here. Needs real work adapting `retrieve_cos_rrf()`'s output into quiz evidence chunks -- a retrieval-design task, not a routing one.
- **Combined-source mode** (document + KB in one quiz, per-question provenance). Same reason -- no implementation to route to yet.
- **Partial-quiz UX** (N-1 of N questions valid). This is a product decision (the source plan itself flags it as one, Section 12) -- not defaulted here.
- **Persisting a chat-originated quiz as reloadable/interactive.** It renders live and correctly, but `activeChat.messages` stores only a text summary for it today; reopening a saved chat later will show that summary line, not the interactive quiz. Flagged directly in the client code where this happens.

None of these block what was built: a user can now get a real, structured, validated quiz by typing a natural request into chat, from pretrained knowledge, Housing policy, or an attached document, exactly as reliably as using the Quiz button -- which was the actual bug report this work started from.
