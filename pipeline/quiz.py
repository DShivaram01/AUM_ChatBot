"""Task 36: structured, validated MCQ generation.

Quiz is an activity (plan Section 2), not a fourth chat pipeline -- this
module only knows how to turn retrieved/empty evidence plus a topic into a
validated quiz. Source selection (document/aum/pretrained) and retrieval
live in core/assistant_service.py, same split as document_chat().

Generates one question per model call, not the whole quiz in one shot.
Empirically (see workspace.md Task 36 log entry), asking Mistral-7B for
N questions in a single JSON blob produces increasingly inconsistent key
names and quoting as the output grows (observed: correct_Index,
correct_ Index, correct_idx, explanatons, explanaton, evidence_\\_refs,
evidence__refs, ... all within one 5-question generation). One question
per call is short enough to stay well-formed after repair/normalization,
and a bad question can be retried individually instead of discarding an
otherwise-good batch.
"""

from __future__ import annotations

import json
import re

from pipeline.answer import generate_streaming
from pipeline.memory import logger

REQUIRED_OPTIONS = 4
_MAX_ATTEMPTS_PER_QUESTION = 2  # one real attempt + one retry with feedback

_KEY_CANONICAL = {
    "title": "title",
    "questions": "questions",
    "question": "question",
    "options": "options",
    "option": "options",
    "correctindex": "correct_index",
    "correct_index": "correct_index",
    "explanation": "explanation",
    "explanations": "explanation",
    "evidencerefs": "evidence_refs",
    "evidence_refs": "evidence_refs",
}


def _repair_json_text(raw: str) -> str:
    """Repair the common non-strict-JSON patterns observed in practice:
    (1) a single-quoted value used where a double-quoted JSON string was
    required, e.g. a bare array element like 'Round Robin (RR)', and
    (2) backslash-escaped underscores (e.g. evidence\\_refs) that aren't
    valid JSON escapes.

    Case (1) must NOT touch a single quote used as nested emphasis inside
    an already-valid double-quoted string, e.g. "the 'time quantum'
    concept" or "'Least Laxity First' algorithm" -- both observed in real
    generations. A blind regex can't tell the two apart; this scans the
    text tracking whether we're inside a double-quoted string and only
    treats a single-quote span as a mistaken delimiter when it appears
    OUTSIDE one. Also strips literal markdown bold markers (observed
    injected mid-array, e.g. right before the model's chosen "correct"
    option) and JavaScript-style `//` line comments the model sometimes
    appends after an option to explain its reasoning (found via Task 43's
    real-generation measurement, not anticipated in advance) -- neither
    is ever valid inside real JSON, both safe to remove, and the comment
    strip uses the same "only outside a double-quoted string" rule as the
    quote repair, so a literal `//` inside real content (e.g. a URL) is
    left alone."""
    raw = raw.replace("**", "")
    raw = re.sub(r'\\([^"\\/bfnrtu])', r"\1", raw)

    out: list[str] = []
    in_double = False
    i, n = 0, len(raw)
    while i < n:
        ch = raw[i]
        if ch == '"' and (i == 0 or raw[i - 1] != "\\"):
            in_double = not in_double
            out.append(ch)
            i += 1
            continue
        if ch == "'" and not in_double:
            j = i + 1
            while j < n and raw[j] not in ("'", '"'):
                j += 1
            if j < n and raw[j] == "'":
                out.append('"' + raw[i + 1:j] + '"')
                i = j + 1
                continue
        if ch == "/" and not in_double and i + 1 < n and raw[i + 1] == "/":
            while i < n and raw[i] != "\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize_keys(obj):
    """Fold key variance (trailing whitespace, case, a few observed
    synonyms/plurals, stray internal spaces/underscores) onto the
    canonical keys this module reads. Unknown keys are kept, lowercased
    and stripped, rather than dropped -- _validate() then sees a clearly
    missing field instead of silently losing data to an unrecognized
    alias."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            normalized = re.sub(r"[\s_]+", "", str(k).strip().lower())
            canon = _KEY_CANONICAL.get(normalized, str(k).strip().lower())
            out[canon] = _normalize_keys(v)
        return out
    if isinstance(obj, list):
        return [_normalize_keys(v) for v in obj]
    return obj


def _extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("model output did not contain a JSON object")
    text = _repair_json_text(match.group(0))
    return _normalize_keys(json.loads(text))


def _validate_question(q: dict, require_evidence: bool) -> list[str]:
    errors: list[str] = []
    if not isinstance(q, dict):
        return ["not a JSON object"]
    qtext = str(q.get("question", "")).strip()
    if not qtext:
        errors.append("empty question text")

    options = q.get("options")
    if not isinstance(options, list) or len(options) != REQUIRED_OPTIONS:
        errors.append(f"must have exactly {REQUIRED_OPTIONS} options")
    else:
        cleaned = [str(o).strip() for o in options]
        if any(not o for o in cleaned):
            errors.append("empty option text")
        if len({o.lower() for o in cleaned}) != len(cleaned):
            errors.append("duplicate options")

    idx = q.get("correct_index")
    if not isinstance(idx, int) or not (0 <= idx < REQUIRED_OPTIONS):
        errors.append(f"correct_index must be an integer 0-{REQUIRED_OPTIONS - 1}")

    if require_evidence:
        refs = q.get("evidence_refs")
        if not isinstance(refs, list) or not refs:
            errors.append("missing evidence_refs for a grounded quiz")

    return errors


def _validate(data: dict, expected_count: int, require_evidence: bool) -> list[str]:
    """Whole-quiz validation, kept for direct unit testing and as the final
    shape check generate_quiz() itself always satisfies by construction."""
    questions = data.get("questions")
    if not isinstance(questions, list) or len(questions) != expected_count:
        got = len(questions) if isinstance(questions, list) else "none"
        return [f"expected exactly {expected_count} questions, got {got}"]

    errors: list[str] = []
    seen: set[str] = set()
    for i, q in enumerate(questions, 1):
        q_errors = _validate_question(q, require_evidence)
        errors.extend(f"question {i}: {e}" for e in q_errors)
        if isinstance(q, dict):
            key = str(q.get("question", "")).strip().lower()
            if key in seen:
                errors.append(f"question {i}: duplicate question text")
            seen.add(key)
    return errors


def _coerce_ref(ref) -> int | None:
    """The model sometimes emits evidence_refs as strings (e.g. "2" instead
    of 2) -- accept either rather than silently dropping real evidence."""
    if isinstance(ref, bool):
        return None
    if isinstance(ref, int):
        return ref
    if isinstance(ref, str) and ref.strip().lstrip("-").isdigit():
        return int(ref.strip())
    return None


def _finalize_question(q: dict, evidence_map: dict[int, str], number: int) -> dict:
    refs = q.get("evidence_refs") or []
    evidence_ids = [
        evidence_map[key] for key in (_coerce_ref(r) for r in refs)
        if key is not None and key in evidence_map
    ]
    return {
        "question_id": f"q{number}",
        "question": str(q["question"]).strip(),
        "options": [str(o).strip() for o in q["options"]],
        "correct_index": q["correct_index"],
        "explanation": str(q.get("explanation", "")).strip(),
        "evidence_ids": evidence_ids,
    }


def _question_prompt(
    topic: str, context: str, require_evidence: bool,
    avoid_questions: list[str], feedback: str | None,
) -> str:
    grounding_rule = (
        "Base the question ONLY on the material inside <context> tags. "
        "Do NOT use outside knowledge. Set \"evidence_refs\" to the context "
        "number(s) (e.g. [1] or [1,2]) that support it."
        if require_evidence else
        "Use your own general knowledge. Set \"evidence_refs\" to an empty "
        "list, since there is no source document."
    )
    context_block = f"\n\n<context>\n{context}\n</context>\n" if context else ""
    avoid_block = (
        "\n\nDo not repeat any of these already-used questions: "
        + "; ".join(avoid_questions)
        if avoid_questions else ""
    )
    retry_block = (
        f"\n\nYour previous attempt was invalid for this reason: {feedback}\nFix it and try again."
        if feedback else ""
    )
    return (
        "<s>[INST] You generate one multiple-choice quiz question as strict JSON, nothing else.\n\n"
        f"Topic: {topic}\n{grounding_rule}\n\n"
        "Output ONLY a JSON object, no prose before or after, in exactly this shape:\n"
        '{"question": "<text>", "options": ["<A>", "<B>", "<C>", "<D>"], '
        '"correct_index": <0-3>, "explanation": "<why>", '
        '"evidence_refs": [<context numbers, or [] if none>]}\n\n'
        "Exactly four distinct options and exactly one correct_index."
        f"{context_block}{avoid_block}{retry_block} [/INST]"
    )


def generate_quiz(
    topic: str, count: int, llm_tok, llm_model, query_id: str,
    context_chunks: list[str] | None = None,
    evidence_map: dict[int, str] | None = None,
    require_evidence: bool = False,
) -> tuple[dict | None, list[str]]:
    """Generate and validate a quiz, one question per model call. Returns
    (quiz, errors) -- quiz is None and errors is non-empty on a controlled
    failure (never a malformed quiz)."""
    context_chunks = context_chunks or []
    evidence_map = evidence_map or {}
    context = "\n\n---\n\n".join(f"[{i}] {c}" for i, c in enumerate(context_chunks, 1))

    questions: list[dict] = []
    seen_texts: list[str] = []
    for q_num in range(1, count + 1):
        feedback: str | None = None
        accepted: dict | None = None
        last_errors: list[str] = []
        for attempt in range(1, _MAX_ATTEMPTS_PER_QUESTION + 1):
            prompt = _question_prompt(topic, context, require_evidence, seen_texts, feedback)
            full = ""
            for partial in generate_streaming(
                prompt, llm_tok, llm_model,
                query_id=f"{query_id}-q{q_num}a{attempt}", max_new_tokens=220,
            ):
                full = partial
            try:
                data = _extract_json(full)
            except (ValueError, json.JSONDecodeError) as exc:
                last_errors = [f"could not parse model output as JSON: {exc}"]
                logger.warning(
                    f"[{query_id}] Quiz q{q_num} attempt {attempt} unparseable: {exc}; raw={full[:300]!r}"
                )
                feedback = last_errors[0]
                continue
            q_errors = _validate_question(data, require_evidence)
            qtext_key = str(data.get("question", "")).strip().lower()
            if not q_errors and qtext_key in {t.lower() for t in seen_texts}:
                q_errors = ["duplicate of an already-used question"]
            if not q_errors:
                finalized = _finalize_question(data, evidence_map, q_num)
                # A syntactically-present evidence_refs can still fail to map
                # to any real retrieved chunk (wrong number, stray string
                # type already coerced and tried in _finalize_question) --
                # don't silently ship an ungrounded "grounded" question.
                if require_evidence and not finalized["evidence_ids"]:
                    q_errors = ["evidence_refs did not match any retrieved context number"]
                else:
                    accepted = (data, finalized)
                    break
            last_errors = q_errors
            logger.warning(f"[{query_id}] Quiz q{q_num} attempt {attempt} invalid: {q_errors}")
            feedback = "; ".join(q_errors)

        if accepted is None:
            return None, [f"question {q_num}: {'; '.join(last_errors)}"]
        raw_data, finalized = accepted
        questions.append(finalized)
        seen_texts.append(str(raw_data["question"]).strip())

    title = f"{topic.strip().title()} Quiz" if topic.strip() else "Quiz"
    return {"title": title, "questions": questions}, []
