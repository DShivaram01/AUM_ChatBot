"""Experiment 6: blueprint-driven batch quiz generation.

    python -m experiments.quiz_bench.batch_gen --arms seq b5 b10 adaptive --out results/dsa_batch.json

Every arm fills the SAME blueprint (blueprint.py), so arms differ only in
how many slots go into one model call:
  seq       1 slot per call (blueprint + whole-chunk evidence, no batching)
  b5        batches of 5
  b10       batches of 10
  adaptive  add slots to a call until its evidence reaches ADAPTIVE_EVIDENCE_TOKENS
            or it holds ADAPTIVE_MAX_SLOTS slots

Prompts carry whole evidence chunks (each chunk once per call, numbered),
token-aware -- no 3200-char cap. Each question is validated with production
_validate_question (+ evidence ref must point at the slot's chunk); failed
slots are regenerated individually (targeted repair, up to REPAIR_ATTEMPTS),
valid questions are never discarded. A question semantically duplicating an
accepted one (cos >= 0.9) counts as invalid. Multi-question output disables
the in-answer n-gram guard (it forbids repeating JSON keys); earlier accepted
questions are passed as avoid_texts.
"""

import argparse
import json
import re
import time

from experiments.quiz_bench import gen_harness as H
from experiments.quiz_bench.blueprint import build
from experiments.quiz_bench.fixture import gold

REPAIR_ATTEMPTS = 2
ADAPTIVE_EVIDENCE_TOKENS = 1200
ADAPTIVE_MAX_SLOTS = 6
OUT_TOKENS_PER_Q = 190
DUP_COS = 0.9


def batch_prompt(slots, chunk_text, avoid):
    ids = []
    for s in slots:
        if s.chunk_id not in ids:
            ids.append(s.chunk_id)
    num = {cid: i for i, cid in enumerate(ids, 1)}
    evidence = "\n\n".join(f"[{num[cid]}] {chunk_text[cid]}" for cid in ids)
    specs = "\n".join(f"- slot {s.slot}: use evidence [{num[s.chunk_id]}]; objective: {s.objective}" for s in slots)
    avoid_block = ("\n\nDo not repeat or rephrase these existing questions:\n" + "\n".join(f"- {a}" for a in avoid)) if avoid else ""
    n = len(slots)
    return (
        "<s>[INST] You write multiple-choice quiz questions as strict JSON, nothing else.\n"
        "Base every question ONLY on its assigned evidence passage. Do NOT use outside knowledge. "
        "Each question must test a different fact; four distinct options; exactly one correct.\n\n"
        f"<evidence>\n{evidence}\n</evidence>\n\n"
        f"Write exactly {n} question{'s' if n > 1 else ''}, one per slot:\n{specs}{avoid_block}\n\n"
        'Output ONLY: {"questions": [{"slot": <slot number>, "question": "<text>", "options": ["<A>", "<B>", "<C>", "<D>"], '
        '"correct_index": <0-3>, "explanation": "<why, citing the evidence>", "evidence_refs": [<evidence number>]}]} [/INST]'
    ), num


def parse_questions(raw):
    """Accept {"questions":[...]}, a bare list, or a single question object."""
    from pipeline.quiz import _repair_json_text, _normalize_keys
    m = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
    if not m:
        raise ValueError("no JSON in output")
    data = _normalize_keys(json.loads(_repair_json_text(m.group(0))))
    if isinstance(data, dict) and "questions" in data:
        data = data["questions"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("JSON is not a question list")
    return data


class SlotFiller:
    def __init__(self, lm, decode=None, in_answer_guard=False):
        self.lm, self.decode, self.in_answer_guard = lm, decode, in_answer_guard
        chunks, _ = H.document()
        self.text = {c["chunk_id"]: c["text"] for c in chunks}
        self.stats = {"batches": 0, "parse_failures": 0, "invalid_items": 0, "repairs": 0, "duplicates": 0,
                      "first_result_seconds": None, "slot_attempts": {}}

    def _accept(self, q, slot, num, accepted, emb_cache):
        from pipeline.quiz import _validate_question, _coerce_ref
        from sentence_transformers import util
        errs = _validate_question(q, require_evidence=True)
        refs = [_coerce_ref(r) for r in (q.get("evidence_refs") or [])]
        if not errs and num[slot.chunk_id] not in refs:
            errs.append("evidence_refs does not point at the assigned passage")
        if errs:
            return errs
        if accepted:
            e = H.embedder().encode([str(q["question"])])
            if float(util.cos_sim(e, emb_cache).max()) >= DUP_COS:
                self.stats["duplicates"] += 1
                return ["semantic duplicate of an accepted question"]
        return []

    def fill(self, slots, batches, t0):
        import numpy as np
        accepted, emb_rows = {}, []
        pending = [list(b) for b in batches]
        attempt = {s.slot: 0 for s in slots}
        while pending:
            group = pending.pop(0)
            for s in group:
                attempt[s.slot] += 1
            avoid = [accepted[k]["question"] for k in sorted(accepted)]
            prompt, num = batch_prompt(group, self.text, avoid)
            kwargs = self.decode(group) if self.decode else {}
            raw = self.lm.generate(prompt, OUT_TOKENS_PER_Q * len(group) + 40, avoid_texts=avoid,
                                   in_answer_guard=self.in_answer_guard if len(group) > 1 else True, **kwargs)
            self.stats["batches"] += 1
            try:
                items = parse_questions(raw)
            except Exception:
                self.stats["parse_failures"] += 1
                items = []
            by_slot = {}
            for i, q in enumerate(items):
                sl = q.get("slot") if isinstance(q, dict) else None
                sl = sl if isinstance(sl, int) and any(s.slot == sl for s in group) else (group[i].slot if i < len(group) else None)
                if sl is not None and sl not in by_slot:
                    by_slot[sl] = q
            failed = []
            for s in group:
                q = by_slot.get(s.slot)
                errs = ["missing from output"] if q is None else self._accept(q, s, num, accepted,
                                                                               np.array(emb_rows) if emb_rows else None)
                if errs:
                    self.stats["invalid_items"] += 1
                    failed.append(s)
                    continue
                accepted[s.slot] = {"slot": s.slot, "question": str(q["question"]).strip(),
                                    "options": [str(o).strip() for o in q["options"]], "correct_index": q["correct_index"],
                                    "explanation": str(q.get("explanation", "")).strip(), "evidence_ids": [s.chunk_id]}
                emb_rows.append(H.embedder().encode([accepted[s.slot]["question"]])[0])
                if self.stats["first_result_seconds"] is None:
                    self.stats["first_result_seconds"] = round(time.perf_counter() - t0, 1)
            for s in failed:                     # targeted repair: one slot per call
                if attempt[s.slot] <= REPAIR_ATTEMPTS:
                    self.stats["repairs"] += 1
                    pending.append([s])
        self.stats["slot_attempts"] = attempt
        return [accepted[k] for k in sorted(accepted)]


def make_batches(slots, arm, ntok, chunk_text):
    if arm == "seq":
        return [[s] for s in slots]
    if arm in ("b5", "b10"):
        k = int(arm[1:])
        return [slots[i:i + k] for i in range(0, len(slots), k)]
    if arm == "adaptive":
        out, cur, cur_ids, tok = [], [], set(), 0
        for s in slots:
            add = 0 if s.chunk_id in cur_ids else ntok(chunk_text[s.chunk_id])
            if cur and (tok + add > ADAPTIVE_EVIDENCE_TOKENS or len(cur) >= ADAPTIVE_MAX_SLOTS):
                out.append(cur); cur, cur_ids, tok = [], set(), 0
                add = ntok(chunk_text[s.chunk_id])
            cur.append(s); cur_ids.add(s.chunk_id); tok += add
        return out + ([cur] if cur else [])
    raise ValueError(arm)


def run_arm(lm, arm, requests, decode=None, label=None):
    from pipeline.quiz_intent import detect_quiz_intent
    g = gold()
    chunks, _ = H.document()
    text = {c["chunk_id"]: c["text"] for c in chunks}
    res, private = {}, {}
    for key in requests:
        req = g["quiz_requests"][key]
        intent = detect_quiz_intent(req["prompt"], has_attached_document=True)
        bp = build(req["scope"] if req["scope"] == "whole" else "topic", intent.topic, intent.count, lm.ntok)
        before = (lm.log.calls, lm.log.input_tokens, lm.log.output_tokens)
        t0 = time.perf_counter()
        filler = SlotFiller(lm, decode)
        qs = filler.fill(bp.slots, make_batches(bp.slots, arm, lm.ntok, text), t0) if bp.supported else []
        secs = time.perf_counter() - t0
        rec = {"requested": intent.count, "planned_slots": len(bp.slots), "supported": bp.supported, "note": bp.note,
               "complete": len(qs) == len(bp.slots), "seconds": round(secs, 1),
               "calls": lm.log.calls - before[0], "input_tokens": lm.log.input_tokens - before[1],
               "output_tokens": lm.log.output_tokens - before[2], **{k: v for k, v in filler.stats.items() if k != "slot_attempts"},
               "blueprint_chapters": sorted({s.chapter for s in bp.slots}),
               **H.score_quiz(qs, req, ["red-black", "red black"] if key == "F" else None)}
        res[key] = rec
        private[key] = {"blueprint": bp.as_dict(), "questions": qs}
        print(f"[{label or arm}] {key}", json.dumps({k: rec.get(k) for k in ("supported", "planned_slots", "n_questions", "complete",
                                                                            "seconds", "first_result_seconds", "calls", "input_tokens",
                                                                            "output_tokens", "invalid_items", "repairs", "parse_failures",
                                                                            "near_duplicate_pairs", "n_gold_concepts", "cited_chapters",
                                                                            "lexically_supported", "copied_embedded_questions")}), flush=True)
    return res, private


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["seq", "b5", "b10", "adaptive"])
    ap.add_argument("--requests", default="A,B,C,D,E,F")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    if a.fake:
        from experiments.quiz_bench.run_gen_baseline import FakeLM
        lm = FakeBatchLM()
    else:
        lm = H.MistralLM()
    out, private = {"load_seconds": getattr(lm, "load_seconds", 0), "arms": {}}, {}
    for arm in a.arms:
        out["arms"][arm], private[arm] = run_arm(lm, arm, a.requests.split(","))
    out["gpu_peak_gib"] = H.gpu_peak_gib()
    json.dump(out, open(a.out, "w"), indent=1)
    if not a.fake:
        print("private outputs ->", H.save_private("batch", private))


class FakeBatchLM:
    """Echoes valid questions for every slot in the prompt (dry run)."""
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
        self.log.input_tokens += self.ntok(prompt)
        qs = []
        for slot, ev in re.findall(r"- slot (\d+): use evidence \[(\d+)\]", prompt):
            self.n += 1
            qs.append({"slot": int(slot), "question": f"Distinct fake question number {self.n} about topic {self.n * 7}?",
                       "options": ["a", "b", "c", "d"], "correct_index": 0, "explanation": "x", "evidence_refs": [int(ev)]})
        return json.dumps({"questions": qs})


if __name__ == "__main__":
    main()
