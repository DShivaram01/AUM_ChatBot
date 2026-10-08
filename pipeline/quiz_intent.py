"""Task 53 Phase 1-2: deterministic activity-intent detection for Quiz, so
a natural-language chat request ("generate 5 MCQs on X") reaches the exact
same AssistantService.quiz() pipeline the Quiz button already uses,
instead of a separate, unstructured free-text answer from general_chat()
-- this was the user's own original bug report (Task 52 log entry: "open
ended mode... gave me 5 questions with options but unstructured").

Deliberately rule-based, not an LLM classifier: the trigger phrases are a
closed, well-known set ("quiz me on X", "generate N MCQs on X", ...), and
a false positive here silently hijacks an ordinary question into a
multi-call quiz generation instead of answering it -- worth preventing
with simple, auditable rules rather than a model call that could itself
misfire. See the design document this implements
(AUM_CHATBOT_INTELLIGENT_QUIZ_ROUTING_AND_RELIABILITY_PLAN_2026-10-08.md,
Section 6) and workspace.md TASK 53/Entry 063 for the validation and
scope notes -- this module intentionally does NOT implement "quiz
follow-up" detection (e.g. "explain question 3") or COS/combined source
resolution; both are out of scope for this pass (see that entry).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from pipeline.classifier import _HOUSING_HINT_WORDS

_DEFAULT_COUNT = 5
_MIN_COUNT, _MAX_COUNT = 1, 20

_TRIGGER_VERBS = r"(?:create|make|generate|build|write|give me|prepare|provide)"

# Matches without needing a preceding count -- these nouns are
# unambiguously about quiz content on their own.
_QUIZ_NOUN_UNQUALIFIED = (
    r"(?:mcqs?|multiple[- ]choice questions?|"
    r"quiz(?:zes)?(?!\s+(?:app|application|website|program|system|schedule|time|date))|"
    r"practice questions?|test questions?)"
)

# Bare "questions" is too generic a word to trust on its own (ordinary
# chat says "I have questions about X" constantly) -- only counts as a
# quiz-content noun when a count preceded it ("10 questions").
#
# NOTE: this used to be written with a single optional "(?:(?P<count>\d+)
# \s+)?" prefix followed by "(?:NOUN|(?(count)questions?))" -- a Python
# conditional group with no "no" branch matches an EMPTY STRING when the
# condition is false, not a failure. That silently made the entire
# quiz-noun requirement optional whenever no digit was present, so
# "create a quiz app" (and anything else matching a bare trigger verb)
# incorrectly matched. Three explicit alternatives instead, each either
# requiring a real quiz-noun or requiring both a count AND "questions".
_CREATE_RE = re.compile(
    rf"{_TRIGGER_VERBS}\s+(?:me\s+)?(?:a\s+|an\s+|some\s+)?"
    rf"(?:"
    rf"(?P<count_a>\d{{1,2}})\s+{_QUIZ_NOUN_UNQUALIFIED}"  # "5 MCQs"
    rf"|{_QUIZ_NOUN_UNQUALIFIED}"                            # "MCQs" (no count)
    rf"|(?P<count_b>\d{{1,2}})\s+questions?"                # "10 questions" (count required)
    rf")\b",
    re.IGNORECASE,
)

_QUIZ_ME_RE = re.compile(r"\bquiz\s+me\b(?:\s+(?:on|about))?", re.IGNORECASE)

# Connector separating the trigger phrase from the actual subject, e.g.
# "generate 5 MCQs **on** protein sequences" -> topic = "protein sequences".
_TOPIC_SPLIT_RE = re.compile(
    r"\b(?:on|about|from|regarding|covering|for|using)\b\s*(.+)$", re.IGNORECASE,
)


@dataclass
class ActivityIntent:
    activity: str  # "quiz_create" | "chat"
    topic: str | None = None
    count: int | None = None


def detect_quiz_intent(text: str, *, has_attached_document: bool = False) -> ActivityIntent:
    """Rule-based quiz-activity detector. Returns activity="chat" (the
    conservative default -- fall through to normal routing) unless the
    text clearly matches a known quiz-request pattern.

    has_attached_document only affects what happens when a quiz pattern
    matched but left no usable topic text (e.g. a bare "quiz me on this
    PDF" is caught by the topic split and works fine; a bare "quiz me"
    with nothing else does not, unless a document is attached to ask
    about)."""
    stripped = text.strip()
    if not stripped:
        return ActivityIntent(activity="chat")

    matched = _CREATE_RE.search(stripped) or _QUIZ_ME_RE.search(stripped)
    if not matched:
        return ActivityIntent(activity="chat")

    count = _DEFAULT_COUNT
    groups = matched.groupdict()
    count_group = groups.get("count_a") or groups.get("count_b")
    if count_group:
        count = max(_MIN_COUNT, min(_MAX_COUNT, int(count_group)))

    remainder = stripped[matched.end():].strip(" .,:;!?")
    topic_match = _TOPIC_SPLIT_RE.search(remainder) or _TOPIC_SPLIT_RE.search(stripped)
    topic = (topic_match.group(1) if topic_match else remainder).strip(" .,:;!?")

    if not topic:
        if has_attached_document:
            topic = "the main topics covered in the attached document"
        else:
            return ActivityIntent(activity="chat")

    return ActivityIntent(activity="quiz_create", topic=topic, count=count)


# No source matched and pretrained generation isn't allowed for this
# request -- ask rather than silently guessing (plan Section 5, "No
# sufficiently relevant source"; and this project's own standing rule
# that GENERAL/open-ended is an explicit opt-in, never an automatic
# fallback -- see server/api_server.py:_resolve_topic's own GENERAL
# handling, which this mirrors for quiz).
NO_SOURCE_MESSAGE = (
    "I can build that quiz from an attached document, from AUM Housing "
    "policy, or from Mistral's own general knowledge -- but I need to "
    "know which. Attach a document, mention a housing-related topic, or "
    "turn on Open-ended mode and ask again."
)


def resolve_quiz_source(
    topic: str, *, has_attached_document: bool, open_ended_enabled: bool,
) -> tuple[str | None, str | None]:
    """Decide which of AssistantService.quiz()'s three source modes a
    natural-language quiz request should use. Mirrors the source-selection
    precedence in the plan this implements (Section 3): explicit
    attachment always wins; otherwise infer from topic content; otherwise
    only fall back to pretrained knowledge if the user has already opted
    into Open-ended mode for this message (never silently).

    Returns (source_mode, None) on success, or (None, error_message) when
    no source can be determined -- the caller should surface the message
    rather than guess.

    Does NOT resolve a COS source or a combined source -- neither has a
    real retrieval implementation in AssistantService.quiz() yet (see
    workspace.md TASK 53's own scope notes); a topic that would ideally
    route to COS falls through to pretrained/ask-for-a-source instead of
    silently misrouting to Housing.
    """
    if has_attached_document:
        return "document", None

    tokens = set(re.findall(r"[a-z]+", topic.lower()))
    if tokens & _HOUSING_HINT_WORDS:
        return "housing", None

    if open_ended_enabled:
        return "pretrained", None

    return None, NO_SOURCE_MESSAGE


# ---------- Quiz follow-ups (Task 54) ----------
# Deliberately NOT checked unconditionally against every message -- the
# caller (server/api_server.py) only runs this when the session actually
# has a stored last-generated quiz. That's what keeps "explain question 3"
# from misfiring against, say, a document question that happens to
# mention "question 3" of some unrelated numbered list; without a quiz to
# explain, there's nothing for this to match against in the first place.

_QUESTION_NUM_RE = re.compile(r"\bquestion\s*(\d{1,2})\b", re.IGNORECASE)
_EXPLAIN_TRIGGER_RE = re.compile(r"\b(?:explain|why)\b", re.IGNORECASE)
_REGENERATE_RE = re.compile(
    r"\b(try again|regenerate|make another(?:\s+one|\s+quiz)?|"
    r"give me another(?:\s+quiz)?|another quiz(?:\s+on the same topic)?|"
    r"new quiz(?:\s+on the same topic)?|redo (?:the|this) quiz)\b",
    re.IGNORECASE,
)


def detect_quiz_followup(text: str) -> dict | None:
    """Returns {"kind": "explain", "question_number": N} for "explain
    question N" (in either word order, "why is question 2's answer X"
    also matches); {"kind": "regenerate"} for "try again"/"regenerate"/
    "make another one"/etc.; None otherwise."""
    stripped = text.strip()
    if not stripped:
        return None

    question_match = _QUESTION_NUM_RE.search(stripped)
    if question_match and _EXPLAIN_TRIGGER_RE.search(stripped):
        return {"kind": "explain", "question_number": int(question_match.group(1))}

    if _REGENERATE_RE.search(stripped):
        return {"kind": "regenerate"}

    return None
