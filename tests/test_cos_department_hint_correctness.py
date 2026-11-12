"""Regression coverage for the two reported COS "no results" failures:

1. classify_query()'s department regex has zero typo tolerance, so a
   misspelled department name (e.g. "mathemetics") silently drops the
   department filter instead of applying it.
2. retrieve_cos_rrf() used to hard-filter candidates to dept_hint's
   department. When dept_hint came from a topic word that merely overlaps
   a department name (e.g. "computational biology" -> Biology and
   Environmental Science), that hard filter could throw away the single
   best, genuinely relevant candidate filed under a different department,
   leaving only weak same-department candidates and false-rejecting the
   whole query at the evidence floor.
"""

import numpy as np

from pipeline.answer import _relative_threshold
from pipeline.classifier import classify_query
from pipeline.retrieval import _cos_evidence_passes, build_bm25, retrieve_cos_rrf


# ── Fix 1: fuzzy department matching catches misspellings ────────────────

def test_misspelled_department_is_still_recognized():
    assert classify_query("provide research projects related to mathemetics department")["dept_hint"] == "Mathematics"
    assert classify_query("tell me about biologie research")["dept_hint"] == "Biology and Environmental Science"
    assert classify_query("projects in the chemistery department")["dept_hint"] == "Chemistry"
    assert classify_query("pyschology research at the symposium")["dept_hint"] == "Psychology"


def test_correctly_spelled_department_still_uses_the_exact_regex_path():
    # The fuzzy fallback must not interfere with the existing exact match.
    assert classify_query("mathematics department projects")["dept_hint"] == "Mathematics"
    assert classify_query("biology research")["dept_hint"] == "Biology and Environmental Science"


def test_fuzzy_department_matching_has_no_false_positives_on_common_words():
    for query in [
        "list some research projects",
        "what symposium projects were presented",
        "show me the mentor and presenters",
        "give me a project abstract",
    ]:
        assert classify_query(query)["dept_hint"] is None


# ── Fix 2: dept_hint is a soft preference with a hard-filter retry, not an
#    unconditional hard exclusion ─────────────────────────────────────────

def _make_corpus():
    # All three docs share "zzyx qvwp" so BM25 returns all of them with a
    # positive score regardless of department.
    meta = [
        {"title": "Zzyx Qvwp Weak Alpha", "mentor": "", "department": "Dept X",
         "departments": ["Dept X"], "category": "", "year": 2024, "keywords": []},
        {"title": "Zzyx Qvwp Strong Match", "mentor": "", "department": "Dept Y",
         "departments": ["Dept Y"], "category": "", "year": 2024, "keywords": []},
        {"title": "Zzyx Qvwp Weak Beta", "mentor": "", "department": "Dept X",
         "departments": ["Dept X"], "category": "", "year": 2024, "keywords": []},
    ]
    texts = [
        "Abstract: zzyx qvwp weak alpha filler text",
        "Abstract: zzyx qvwp strong match filler text",
        "Abstract: zzyx qvwp weak beta filler text",
    ]
    bm25 = build_bm25(meta, texts)
    return meta, texts, bm25


class _FakeEmbedder:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True):
        return np.zeros((1, 4), dtype=np.float32)


class _FakeIndex:
    def search(self, q_vec, k):
        return np.zeros((1, k), dtype=np.float32), -np.ones((1, k), dtype=int)


class _FakeReranker:
    """Scores "strong" documents clearly above the evidence floor and
    everything else clearly below it, regardless of which department the
    retrieval happens to have filtered down to."""
    def predict(self, pairs):
        return [5.0 if "strong" in text else -5.0 for _, text in pairs]


def test_dept_hint_falls_back_to_cross_department_match_when_own_dept_is_weak():
    meta, texts, bm25 = _make_corpus()
    qinfo = {"type": "TYPE_TOPIC", "person_hints": [], "year_hint": None,
             "dept_hint": "Dept X", "is_broad": True}
    cands = retrieve_cos_rrf(
        "zzyx qvwp", qinfo, _FakeEmbedder(), _FakeIndex(), None,
        meta, texts, bm25, _FakeReranker(), query_id="T1",
    )
    assert cands[0]["meta"]["title"] == "Zzyx Qvwp Strong Match"
    is_strong, top, *_ = _relative_threshold(cands)
    assert is_strong is True and top == 5.0


def test_dept_hint_keeps_its_own_weak_candidates_when_nothing_else_is_strong():
    meta, texts, bm25 = _make_corpus()
    # Rewrite doc1's text so nothing in the corpus contains "strong" --
    # every candidate, in or out of Dept X, now reads as weak evidence.
    texts = [t.replace("strong", "mid") for t in texts]
    bm25 = build_bm25(meta, texts)
    qinfo = {"type": "TYPE_TOPIC", "person_hints": [], "year_hint": None,
             "dept_hint": "Dept X", "is_broad": True}
    cands = retrieve_cos_rrf(
        "zzyx qvwp", qinfo, _FakeEmbedder(), _FakeIndex(), None,
        meta, texts, bm25, _FakeReranker(), query_id="T2",
    )
    # Every returned candidate must still belong to Dept X: no silent
    # substitution of an unrelated department's weak top hit.
    assert cands
    assert all("Dept X" in c["meta"]["departments"] for c in cands)
    assert _cos_evidence_passes(cands) is False
