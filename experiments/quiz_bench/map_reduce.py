"""Experiment 4: concept-inventory strategies for whole-document quizzes.

    python -m experiments.quiz_bench.map_reduce --out results/dsa_map_reduce.json

Separates the three stages the plan asks to keep distinct:
  1. evidence/concept SELECTION (this experiment's variable),
  2. concept extraction (LLM, only in the map arms),
  3. question generation (identical for every arm: one question per selected
     item, sequential, from that item's own evidence chunk).

Selection arms for a request of N questions:
  direct      production: top-6 chunks for the literal topic "this document"
              (questions dealt over those 6 chunks)
  blueprint   no-LLM chapter round-robin over eligible chunks (blueprint.py)
  map_section MAP: one LLM call per chapter extracts up to FACTS_PER_CALL
              quiz-worthy facts with evidence numbers; REDUCE: embedding dedupe
              (cos >= DEDUPE_COS) then chapter round-robin to N
  map_window  MAP over sliding windows of WINDOW consecutive eligible chunks
              (stride STRIDE) instead of chapters; same REDUCE

Inventory metrics: gold concepts whose anchor chunk is among the selected
evidence; for map arms also whether the extracted fact restates the concept.
"""

import argparse
import json
import re
import time

from experiments.quiz_bench import gen_harness as H
from experiments.quiz_bench.blueprint import build, eligible_chunks
from experiments.quiz_bench.fixture import gold, norm

FACTS_PER_CALL = 4
DEDUPE_COS = 0.85
WINDOW, STRIDE = 3, 2


def map_prompt(chunks):
    ev = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))
    return ("<s>[INST] Read the passages and list up to "
            f"{FACTS_PER_CALL} distinct facts that are explicitly stated in them and would make good multiple-choice "
            "quiz questions. Do not add anything that is not in the passages.\n\n"
            f"<passages>\n{ev}\n</passages>\n\n"
            'Output ONLY JSON: {"facts": [{"fact": "<one sentence>", "evidence": [<passage number>]}]} [/INST]')


def extract(lm, groups):
    from pipeline.quiz import _repair_json_text, _normalize_keys
    facts, failures = [], 0
    for group in groups:
        raw = lm.generate(map_prompt(group), 60 * FACTS_PER_CALL + 40, in_answer_guard=False)
        try:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = _normalize_keys(json.loads(_repair_json_text(m.group(0))))
            for f in data.get("facts", []):
                refs = [r for r in f.get("evidence", []) if isinstance(r, int) and 1 <= r <= len(group)]
                if str(f.get("fact", "")).strip() and refs:
                    c = group[refs[0] - 1]
                    facts.append({"fact": str(f["fact"]).strip(), "chunk_id": c["chunk_id"], "chapter": H.chapter_of(c)})
        except Exception:
            failures += 1
    return facts, failures


def reduce(facts, n):
    """Embedding dedupe, then chapter round-robin (document order) to n."""
    from sentence_transformers import util
    if not facts:
        return []
    E = H.embedder().encode([f["fact"] for f in facts])
    keep = []
    for i in range(len(facts)):
        if all(float(util.cos_sim(E[i], E[j])) < DEDUPE_COS for j in keep):
            keep.append(i)
    chapters = []
    for i in keep:
        if facts[i]["chapter"] not in chapters:
            chapters.append(facts[i]["chapter"])
    queues = {ch: [facts[i] for i in keep if facts[i]["chapter"] == ch] for ch in chapters}
    out = []
    while len(out) < n and any(queues.values()):
        for ch in chapters:
            if queues[ch] and len(out) < n:
                out.append(queues[ch].pop(0))
    return out, len(keep)


def question_prompt(item_text, chunk_text, avoid):
    avoid_block = ("\n\nDo not repeat these existing questions: " + "; ".join(avoid)) if avoid else ""
    return ("<s>[INST] Write ONE multiple-choice quiz question as strict JSON, nothing else. Base it ONLY on the "
            "passage; do not use outside knowledge.\n\n"
            f"Focus: {item_text}\n\n<passage>\n{chunk_text}\n</passage>{avoid_block}\n\n"
            'Output ONLY: {"question": "<text>", "options": ["<A>", "<B>", "<C>", "<D>"], "correct_index": <0-3>, '
            '"explanation": "<why>"} [/INST]')


def generate_from_items(lm, items, text_of):
    from pipeline.quiz import _extract_json, _validate_question
    qs, invalid = [], 0
    for it in items:
        avoid = [q["question"] for q in qs]
        for _ in range(2):
            raw = lm.generate(question_prompt(it["focus"], text_of[it["chunk_id"]], avoid), 320, avoid_texts=avoid)
            try:
                q = _extract_json(raw)
                if not _validate_question(q, require_evidence=False):
                    qs.append({"question": str(q["question"]).strip(), "options": [str(o).strip() for o in q["options"]],
                               "correct_index": q["correct_index"], "evidence_ids": [it["chunk_id"]]})
                    break
            except Exception:
                pass
            invalid += 1
    return qs, invalid


def inventory_coverage(items, by_id):
    g = gold()
    ev = {it["chunk_id"] for it in items}
    anchored = {c["id"] for c in g["concepts"] if any(all(norm(a) in norm(by_id[e]["text"]) for a in c["anchors"]) for e in ev)}
    stated = set()
    for it in items:
        if it.get("fact"):
            f = norm(it["fact"])
            for c in g["concepts"]:
                if all(norm(a) in norm(by_id[it["chunk_id"]]["text"]) for a in c["anchors"]) and \
                        sum(w in f for w in norm(c["answer"]).split() if len(w) > 3) >= 1:
                    stated.add(c["id"])
    return {"evidence_chunks": len(ev), "chapters": len({it["chapter"] for it in items}),
            "gold_concepts_in_evidence": len(anchored), "gold_concepts_stated_by_facts": len(stated) if any(i.get("fact") for i in items) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--counts", default="10,20")
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    lm = FakeMapLM() if a.fake else H.MistralLM()
    chunks, _ = H.document()
    by_id = {c["chunk_id"]: c for c in chunks}
    text_of = {c["chunk_id"]: c["text"] for c in chunks}
    elig = eligible_chunks(lm.ntok)
    chapters = []
    for c in elig:
        if H.chapter_of(c) not in chapters:
            chapters.append(H.chapter_of(c))
    section_groups = [[c for c in elig if H.chapter_of(c) == ch] for ch in chapters]
    window_groups = [elig[i:i + WINDOW] for i in range(0, max(1, len(elig) - WINDOW + 1), STRIDE)]

    out, private = {"arms": {}}, {}
    mapped = {}
    for arm, groups in (("map_section", section_groups), ("map_window", window_groups)):
        b = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens); t = time.perf_counter()
        facts, failures = extract(lm, groups)
        mapped[arm] = {"facts": facts, "map_calls": lm.log.calls - b[0], "map_input_tokens": lm.log.input_tokens - b[1],
                       "map_output_tokens": lm.log.output_tokens - b[2], "map_seconds": round(time.perf_counter() - t, 1),
                       "map_parse_failures": failures, "raw_facts": len(facts)}
    for n in [int(x) for x in a.counts.split(",")]:
        req = {"scope": "whole"}
        arms = {}
        direct = [c for c, _ in H.retrieve("this document", 6)]
        arms["direct"] = [{"chunk_id": direct[i % len(direct)]["chunk_id"], "chapter": H.chapter_of(direct[i % len(direct)]),
                           "focus": "the passage"} for i in range(n)]
        bp = build("whole", "this document", n, lm.ntok)
        arms["blueprint"] = [{"chunk_id": s.chunk_id, "chapter": s.chapter, "focus": s.objective} for s in bp.slots]
        for arm in ("map_section", "map_window"):
            sel, unique = reduce(mapped[arm]["facts"], n)
            arms[arm] = [{**f, "focus": f["fact"]} for f in sel]
            mapped[arm]["unique_facts"] = unique
        for arm, items in arms.items():
            b = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens); t = time.perf_counter()
            qs, invalid = generate_from_items(lm, items, text_of)
            rec = {"n_requested": n, "n_items": len(items), **inventory_coverage(items, by_id),
                   "gen_calls": lm.log.calls - b[0], "gen_input_tokens": lm.log.input_tokens - b[1],
                   "gen_output_tokens": lm.log.output_tokens - b[2], "gen_seconds": round(time.perf_counter() - t, 1),
                   "invalid_attempts": invalid, **H.score_quiz(qs, req)}
            if arm in mapped:
                rec.update({k: v for k, v in mapped[arm].items() if k != "facts"})
            out["arms"][f"{arm}@{n}"] = rec
            private[f"{arm}@{n}"] = {"items": items, "questions": qs}
            print(f"{arm}@{n}", json.dumps({k: rec.get(k) for k in ("n_items", "chapters", "gold_concepts_in_evidence",
                                                                    "gold_concepts_stated_by_facts", "map_calls", "map_seconds",
                                                                    "gen_seconds", "n_questions", "near_duplicate_pairs",
                                                                    "n_gold_concepts", "lexically_supported",
                                                                    "copied_embedded_questions")}), flush=True)
    out["gpu_peak_gib"] = H.gpu_peak_gib()
    json.dump(out, open(a.out, "w"), indent=1)
    if not a.fake:
        print("private outputs ->", H.save_private("map_reduce", private))


class FakeMapLM:
    def __init__(self):
        from transformers import AutoTokenizer
        import config
        self.tok = AutoTokenizer.from_pretrained(config.LLM_ID)
        self.log = H.CallLog()
        self.n = 0

    def ntok(self, t):
        return len(self.tok(t, add_special_tokens=False)["input_ids"])

    def generate(self, prompt, max_new_tokens, avoid_texts=None, **_):
        self.log.calls += 1
        self.n += 1
        if "list up to" in prompt:
            k = len(re.findall(r"^\[(\d+)\]", prompt, re.M))
            return json.dumps({"facts": [{"fact": f"Fact {self.n}-{i} about item {i * 13 + self.n}", "evidence": [i]} for i in range(1, k + 1)]})
        return json.dumps({"question": f"Fake question {self.n} on {self.n * 3}?", "options": ["a", "b", "c", "d"],
                           "correct_index": 0, "explanation": "x"})


if __name__ == "__main__":
    main()
