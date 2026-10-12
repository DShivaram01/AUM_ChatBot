"""Boundary analysis (evaluator guide section 3), measured not assumed.

For each evidence unit: Mistral-token start/end offsets in the extracted
text; for s in {256, 400, 512, 768}: distance of each edge to the nearest
multiple of s, flagged vulnerable when < 25 tokens; and whether the unit is
actually split (contained in no single chunk) in the zero-overlap F{s} run
vs the 15 %-overlap run.

    python -m experiments.cs451.boundary --out experiments/cs451/results/boundary.json
"""

import argparse
import bisect
import json

from experiments.cs451 import chunking as C
from experiments.cs451.units import contained_units, unit_spans

SIZES = [256, 400, 512, 768]
VULNERABLE = 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    offs = C.token_offsets()
    starts = [o[0] for o in offs]
    tok_span = {lid: (bisect.bisect_right(starts, s) - 1, bisect.bisect_left(starts, e))
                for lid, (s, e) in unit_spans().items()}
    out = {"units": {lid: {"token_start": a_, "token_end": b_} for lid, (a_, b_) in tok_span.items()}, "by_size": {}}
    for s in SIZES:
        covered = {}
        for ov in (0, 15):
            covered[ov] = {u for c in C.build(f"F{s}_ov{ov}") for u in contained_units(c["text"])}
        vul = {}
        for lid, (a_, b_) in tok_span.items():
            d = min(min(a_ % s, s - a_ % s), min(b_ % s, s - b_ % s))
            crosses = a_ // s != (b_ - 1) // s
            vul[lid] = {"edge_distance": d, "vulnerable_edge": d < VULNERABLE, "crosses_boundary": crosses,
                        "split_ov0": lid not in covered[0], "split_ov15": lid not in covered[15]}
        n = len(vul)
        out["by_size"][s] = {
            "units_crossing_a_boundary": sum(v["crosses_boundary"] for v in vul.values()),
            "units_with_vulnerable_edge": sum(v["vulnerable_edge"] for v in vul.values()),
            "split_zero_overlap": sorted(l for l, v in vul.items() if v["split_ov0"]),
            "split_15pct_overlap": sorted(l for l, v in vul.items() if v["split_ov15"]),
            "n_units": n, "detail": vul,
        }
        b = out["by_size"][s]
        print(f"F{s}: crossing={b['units_crossing_a_boundary']}/{n} vulnerable_edge={b['units_with_vulnerable_edge']} "
              f"split@0%={len(b['split_zero_overlap'])} {b['split_zero_overlap'][:8]} split@15%={len(b['split_15pct_overlap'])} "
              f"{b['split_15pct_overlap'][:6]}")
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
