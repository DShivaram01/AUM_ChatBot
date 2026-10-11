"""Experiment 3: hierarchical parent-child retrieval with token-budgeted packing.

Hierarchy: document -> section (markdown headings) -> paragraph -> child.
Children (CHILD tokens) are what gets embedded and searched; each child
belongs to exactly one parent (PARENT tokens, never crossing a section
boundary). Evidence strategies, all packed under the same token BUDGET:

  flat        production chunks (structured400), whole chunks, in rank order
  flat500     production behaviour: top-6 chunks, first 500 chars each
  parent      child hits -> their parents, deduplicated, in first-hit order
  neighbors   child hit + its previous/next child in the same section
  child       child hits alone

Metric: a gold concept counts as covered when its anchor phrase lies inside
the packed evidence. Evaluated per concept query and for quiz requests D/E.

    python -m experiments.quiz_bench.parent_child --out results/parent_child.json
"""

import argparse
import json
import re
from functools import lru_cache
from statistics import median

import numpy as np

from experiments.quiz_bench.fixture import gold, norm, production_pages

CHILD_SIZES = [128, 256, 384]
PARENT_SIZES = [512, 1024, 1536]
BUDGETS = [1000, 2000]


@lru_cache(maxsize=1)
def _tok():
    import config
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.LLM_ID)


@lru_cache(maxsize=None)
def ntok(t: str) -> int:
    return len(_tok()(t, add_special_tokens=False)["input_ids"])


def sections(pages: list[dict]) -> list[dict]:
    """[{"path": [...], "paras": [(text, page)]}] from production markdown."""
    out, path, levels, cur = [], [], [], {"path": [], "paras": []}
    for p in pages:
        pg = p["metadata"]["page_number"]
        for para in re.split(r"\n\s*\n", p["text"]):
            para = para.strip()
            if not para:
                continue
            m = re.match(r"^(#{1,6})\s+(.+)", para)
            if m:
                out.append(cur)
                lvl, title = len(m.group(1)), re.sub(r"[*#]", "", m.group(2)).strip()
                while levels and levels[-1] >= lvl:
                    levels.pop(); path.pop()
                levels.append(lvl); path.append(title)
                cur = {"path": list(path), "paras": []}
                rest = para[m.end():].strip()
                if rest:
                    cur["paras"].append((rest, pg))
                continue
            cur["paras"].append((" ".join(para.split()) if not para.lstrip().startswith("|") else para, pg))
    out.append(cur)
    return [s for s in out if s["paras"]]


def _units(paras, size):
    """Paragraphs, split at sentences when longer than `size`."""
    for text, pg in paras:
        if ntok(text) <= size or text.lstrip().startswith("|"):
            yield text, pg
            continue
        cur = ""
        for s in re.split(r"(?<=[.!?])\s+", text):
            if cur and ntok(cur + " " + s) > size:
                yield cur, pg
                cur = s
            else:
                cur = (cur + " " + s).strip()
        if cur:
            yield cur, pg


def _pack(units, size):
    groups, cur, n = [], [], 0
    for text, pg in units:
        t = ntok(text)
        if cur and n + t > size:
            groups.append(cur); cur, n = [], 0
        cur.append((text, pg)); n += t
    return groups + ([cur] if cur else [])


def build(pages, child_size, parent_size):
    """children: [{"text","page","parent","section","pos"}]; parents: [{"text","page","page_end","path"}]."""
    children, parents = [], []
    for si, s in enumerate(sections(pages)):
        for group in _pack(list(_units(s["paras"], child_size)), parent_size):
            pid = len(parents)
            parents.append({"text": "\n\n".join(t for t, _ in group), "page": group[0][1], "page_end": group[-1][1], "path": s["path"]})
            for cg in _pack(list(_units(group, child_size)), child_size):
                children.append({"text": "\n\n".join(t for t, _ in cg), "page": cg[0][1], "parent": pid, "section": si,
                                 "pos": len(children)})
    return children, parents


class Index:
    def __init__(self, emb, texts):
        self.emb = emb
        self.m = emb.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

    def rank(self, q, k=50):
        v = self.emb.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)[0]
        s = self.m @ v
        return [int(i) for i in np.argsort(-s)[:k]]


def pack_budget(texts_in_order, budget):
    out, used, seen = [], 0, set()
    for t in texts_in_order:
        if t in seen:
            continue
        n = ntok(t)
        if used + n > budget:
            if not out:                       # always return at least a truncated first passage
                ids = _tok()(t, add_special_tokens=False)["input_ids"][:budget]
                out.append(_tok().decode(ids)); used = budget
            continue
        out.append(t); used += n; seen.add(t)
    return out, used


def covered(evidence, anchors):
    text = norm(" ".join(evidence))
    return all(norm(a) in text for a in anchors)


def evaluate(emb):
    g = gold()
    pages = production_pages()
    from pipeline.quiz_intent import detect_quiz_intent
    from ingestion.document_parser import structured_chunks
    flat = structured_chunks(pages)
    flat_idx = Index(emb, [c["text"] for c in flat])
    queries = [(c["id"], c["query"], [c]) for c in g["concepts"]]
    for key in ("D", "E"):
        req = g["quiz_requests"][key]
        queries.append((key, detect_quiz_intent(req["prompt"], has_attached_document=True).topic,
                        [c for c in g["concepts"] if c["id"] in req["concepts"]]))
    results = {}

    def score(name, strategy, budget):
        per, used_tok = {}, []
        for qid, q, targets in queries:
            ev, used = strategy(q, budget)
            per[qid] = sum(covered(ev, t["anchors"]) for t in targets) / len(targets)
            used_tok.append(used)
        concept = [per[c["id"]] for c in g["concepts"]]
        name = f"{name}@{budget}"
        results[name] = {"budget": budget, "concept_coverage": sum(concept) / len(concept),
                         "D_target_coverage": per["D"], "E_target_coverage": per["E"],
                         "evidence_tokens_median": median(used_tok), "per_query": per}
        print(f"{name:34s} budget={str(budget):5s} concepts={results[name]['concept_coverage']:.3f} "
              f"D={per['D']:.2f} E={per['E']:.2f} medTok={median(used_tok):.0f}", flush=True)

    def flat500(q, budget):
        ev = [flat[i]["text"][:500] for i in flat_idx.rank(q, 6)]
        return ev, sum(ntok(t) for t in ev)

    score("flat500 (production prompt)", flat500, None)
    for budget in BUDGETS:
        score("flat structured400", lambda q, b: pack_budget([flat[i]["text"] for i in flat_idx.rank(q)], b), budget)
        for cs in CHILD_SIZES:
            for ps in PARENT_SIZES:
                if ps <= cs:
                    continue
                children, parents = build(pages, cs, ps)
                idx = Index(emb, [c["text"] for c in children])
                score(f"parent c{cs}/p{ps}", lambda q, b, ch=children, pa=parents, ix=idx:
                      pack_budget([pa[ch[i]["parent"]]["text"] for i in ix.rank(q)], b), budget)
            children, parents = build(pages, cs, 10**6)

            def neighbors(q, b, ch=children, ix=None):
                out = []
                for i in ix.rank(q):
                    sec = ch[i]["section"]
                    win = [ch[j]["text"] for j in (i - 1, i, i + 1) if 0 <= j < len(ch) and ch[j]["section"] == sec]
                    out.append("\n\n".join(win))
                return pack_budget(out, b)
            idx = Index(emb, [c["text"] for c in children])
            score(f"neighbors c{cs}", lambda q, b, ch=children, ix=idx: neighbors(q, b, ch, ix), budget)
            score(f"child-only c{cs}", lambda q, b, ch=children, ix=idx: pack_budget([ch[i]["text"] for i in ix.rank(q)], b), budget)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    res = evaluate(SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu"))
    from experiments.quiz_bench.paired_bootstrap import paired_ci
    ids = [c["id"] for c in gold()["concepts"]]
    pairs = [("flat500 (production prompt)@None", "flat structured400@1000"),
             ("flat structured400@2000", "parent c256/p1024@2000"),
             ("flat structured400@2000", "parent c128/p512@2000"),
             ("flat structured400@1000", "parent c384/p512@1000")]
    res["_bootstrap"] = {}
    for base, cand in pairs:
        d, lo, hi = paired_ci([res[base]["per_query"][i] for i in ids], [res[cand]["per_query"][i] for i in ids])
        res["_bootstrap"][f"{cand} - {base}"] = [round(d, 3), round(lo, 3), round(hi, 3)]
        print(f"{cand} - {base}: coverage diff {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  n={len(ids)}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
