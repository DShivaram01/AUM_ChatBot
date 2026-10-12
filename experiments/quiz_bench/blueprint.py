"""Model-independent quiz blueprint (plan Phase 7). No LLM calls.

A blueprint is a list of slots: {slot, chunk_id, chapter, page, section,
objective, coverage_group}. Generators (any model, any batching) fill slots;
they never choose evidence themselves.

Evidence eligibility: a chunk is eligible unless it is one of the document's
own "Quiz Benchmark Questions" boxes (questions, not facts) or has fewer than
MIN_CONTENT_TOKENS tokens (titles, clipped one-line code stubs).

- whole-document request: slots are dealt round-robin across chapters in
  document order (coverage_group = chapter), cycling each chapter's chunks,
  then objectives.
- topic request: relevant = eligible chunks scoring >= the production
  document evidence floor (config.DOCUMENT_EVIDENCE_SCORE_FLOOR, 0.35 -- reused,
  not tuned on this benchmark), plus the remaining eligible chunks of the top
  hit's chapter. No chunk above the floor -> unsupported (no quiz).
- capacity: at most len(OBJECTIVES) slots per chunk. If the request asks for
  more questions than that, the blueprint is shorter and says so -- it never
  invents slots to reach the requested count.
"""

from dataclasses import dataclass, asdict

from experiments.quiz_bench import gen_harness as H
from experiments.quiz_bench.fixture import norm

OBJECTIVES = ["recall a stated fact or definition", "apply or interpret it (e.g. a complexity, formula or example)",
              "compare or distinguish it from a related idea in the same passage"]
MIN_CONTENT_TOKENS = 40


@dataclass
class Slot:
    slot: int
    chunk_id: str
    chapter: str
    page: int
    section: str
    objective: str
    coverage_group: str


@dataclass
class Blueprint:
    scope: str
    requested: int
    slots: list
    supported: bool
    note: str = ""

    def as_dict(self):
        return {"scope": self.scope, "requested": self.requested, "planned": len(self.slots),
                "supported": self.supported, "note": self.note, "slots": [asdict(s) for s in self.slots]}


def eligible_chunks(ntok) -> list[dict]:
    chunks, _ = H.document()
    return [c for c in chunks if not H.is_embedded_quiz_box(c) and ntok(c["text"]) >= MIN_CONTENT_TOKENS]


def _slots_from(groups: list[list[dict]], n: int) -> list[Slot]:
    """Deal n slots round-robin over groups; within a group cycle chunks, then objectives."""
    slots, used = [], {}
    capacity = sum(len(g) for g in groups) * len(OBJECTIVES)
    pos = [0] * len(groups)
    while len(slots) < min(n, capacity):
        progressed = False
        for gi, g in enumerate(groups):
            if len(slots) >= n:
                break
            for _ in range(len(g)):
                c = g[pos[gi] % len(g)]
                pos[gi] += 1
                k = used.get(c["chunk_id"], 0)
                if k < len(OBJECTIVES):
                    used[c["chunk_id"]] = k + 1
                    slots.append(Slot(len(slots) + 1, c["chunk_id"], H.chapter_of(c), c["page"],
                                      " > ".join(c["heading_path"][1:]) or "(document)", OBJECTIVES[k], H.chapter_of(c)))
                    progressed = True
                    break
        if not progressed:
            break
    return slots


def build(scope: str, topic: str, n: int, ntok, floor: float | None = None) -> Blueprint:
    import config
    floor = config.DOCUMENT_EVIDENCE_SCORE_FLOOR if floor is None else floor
    elig = eligible_chunks(ntok)
    if scope == "whole":
        order = []
        for c in elig:
            if H.chapter_of(c) not in order:
                order.append(H.chapter_of(c))
        groups = [[c for c in elig if H.chapter_of(c) == ch] for ch in order]
        slots = _slots_from(groups, n)
        return Blueprint(scope, n, slots, True, "" if len(slots) == n else f"document supports only {len(slots)} distinct slots")
    ids = {c["chunk_id"] for c in elig}
    hits = [(c, s) for c, s in H.retrieve(topic, 50) if c["chunk_id"] in ids]
    above = [c for c, s in hits if s >= floor]
    if not above:
        return Blueprint(scope, n, [], False,
                         f"no passage scores above the evidence floor {floor} (best {hits[0][1]:.3f}); topic not covered by the document")
    top_chapter = H.chapter_of(above[0])
    relevant = above + [c for c in elig if H.chapter_of(c) == top_chapter and c not in above]
    slots = _slots_from([[c] for c in relevant], n)
    note = "" if len(slots) == n else f"the document supports only {len(slots)} reliable questions on this topic (requested {n})"
    return Blueprint(scope, n, slots, True, note)
