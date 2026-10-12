"""Dump the parent/child hierarchy (each parent passage with its child
chunks nested) for every child/parent size used in Experiment 3.

    QZ_FIXTURE=dsa_textbook python -m experiments.quiz_bench.dump_parent_child
Contains document text: private repository only.
"""
from pathlib import Path

from experiments.quiz_bench.fixture import FIXTURE, production_pages
from experiments.quiz_bench.parent_child import CHILD_SIZES, PARENT_SIZES, build, ntok

OUT = Path(__file__).resolve().parent / "dumps" / FIXTURE / "parent_child"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pages = production_pages()
    for cs in CHILD_SIZES:
        for ps in PARENT_SIZES:
            if ps <= cs:
                continue
            children, parents = build(pages, cs, ps)
            lines = [f"# {FIXTURE} — parent-child c{cs}/p{ps}", "",
                     f"{len(parents)} parents, {len(children)} children (children are embedded and searched; parents are returned as evidence).", ""]
            for pid, p in enumerate(parents):
                kids = [c for c in children if c["parent"] == pid]
                lines += [f"## parent {pid} · p.{p['page']}-{p['page_end']} · {ntok(p['text'])} tokens · {' > '.join(p['path'])} · {len(kids)} children", ""]
                for c in kids:
                    lines += [f"### child {c['pos']} · p.{c['page']} · {ntok(c['text'])} tokens", "", "```text", c["text"], "```", ""]
            (OUT / f"c{cs}_p{ps}.md").write_text("\n".join(lines))
            print("wrote", f"c{cs}_p{ps}", len(parents), len(children))


if __name__ == "__main__":
    main()
