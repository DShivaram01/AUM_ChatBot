"""Regression coverage for TASK 30 Tier-1 correctness fixes."""

from pathlib import Path

from pipeline.answer import _relative_threshold
from pipeline.classifier import classify_query
from pipeline.retrieval import (
    _exact_person_cands,
    bm25_search,
    build_bm25,
    build_name_index,
    canonicalize_departments,
    extract_housing_chunks,
    load_jsonl,
)
from server.api_server import _sse_event


ROOT = Path(__file__).resolve().parents[1]


def test_sse_multiline_uses_one_data_prefix_per_line():
    assert _sse_event("first\nsecond") == "data: first\ndata: second\n\n"
    assert _sse_event('{"query_id":"Q1"}', event="done") == (
        'event: done\ndata: {"query_id":"Q1"}\n\n'
    )


def test_housing_chunks_use_actual_hrl_sections_and_pages_without_toc():
    chunks = extract_housing_chunks(ROOT / "data/AUM-Housing-Community-Standards.pdf")
    by_code = {}
    for chunk in chunks:
        by_code.setdefault(chunk["section"][:8], []).append(chunk)
    assert min(chunk["page"] for chunk in by_code["HRL.0001"]) == 14
    assert min(chunk["page"] for chunk in by_code["HRL.0012"]) == 16
    assert min(chunk["page"] for chunk in by_code["HRL.0042"]) == 24
    assert all("table of contents" not in chunk["text"].lower() for chunk in chunks)
    assert {chunk["_chunk_version"] for chunk in chunks} == {4}
    assert all(not chunk["text"].lstrip().startswith("|") for chunk in chunks)


def test_name_disambiguation_keeps_full_name_and_flags_bare_surname():
    rows = load_jsonl(ROOT / "data/cos_data.jsonl")
    build_name_index([row["metadata"] for row in rows])
    assert classify_query("projects by Sutanu Bhattacharya")["person_hints"] == ["Sutanu Bhattacharya"]
    bare = classify_query("projects by Bhattacharya")
    assert bare["person_ambiguous"] is True
    assert len(bare["person_hints"]) > 1


def test_coauthors_are_merged_with_primary_matches():
    meta = [
        {"mentor": "Ada Primary", "lead_presenters": [], "other_authors": []},
        {"mentor": "Other Mentor", "lead_presenters": [], "other_authors": ["Ada Primary"]},
    ]
    cands = _exact_person_cands(["Ada Primary"], meta, ["one", "two"])
    assert {candidate["idx"] for candidate in cands} == {0, 1}


def test_department_mapping_and_bm25_punctuation_are_exact():
    assert canonicalize_departments("Department of Computer Science") == [
        "Computer Science and Computer Information Systems"
    ]
    assert canonicalize_departments("Biology & Environmental Sciences, College of Sciences") == [
        "Biology and Environmental Science"
    ]
    assert canonicalize_departments("1Chemistry and 2Computer Science") == [
        "Chemistry", "Computer Science and Computer Information Systems"
    ]
    bm25 = build_bm25([{"year": 2024}], ["Abstract: 2024 project"])
    assert bm25_search(bm25, "2024?")[1].tolist() == bm25_search(bm25, "2024")[1].tolist()


def test_evidence_gate_rejects_uniformly_low_scores():
    passed, top, _mean, _std = _relative_threshold([{"rerank": -3.0}, {"rerank": -3.0}])
    assert top == -3.0
    assert passed is False
    housing_passed, *_ = _relative_threshold(
        [{"rerank": 0.10}, {"rerank": 0.10}], absolute_floor=0.35
    )
    assert housing_passed is False
