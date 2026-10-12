"""Write human-readable artifacts for inspection: extracted pages, every
chunk of every registered chunker (text, pages, heading path, Mistral
tokens, whether the 500-char prompt view cuts it), the gold set, and any
private generation outputs.

    QZ_FIXTURE=dsa_textbook python -m experiments.quiz_bench.dump_artifacts --chunkers structured400 recursive

Writes under experiments/quiz_bench/dumps/<fixture>/. These files contain
document text: only commit them to a PRIVATE repository.
"""

import argparse
import importlib
import json
import shutil
from pathlib import Path

from experiments.quiz_bench import chunkers as reg
from experiments.quiz_bench.fixture import FIXTURE, bench_dir, gold, production_pages

OUT = Path(__file__).resolve().parent / "dumps" / FIXTURE


def chunk_md(name, chunks, ntok):
    sizes = [ntok(c["text"]) for c in chunks]
    lines = [f"# {FIXTURE} — chunker `{name}`", "",
             f"{len(chunks)} chunks; Mistral tokens min/median/max = {min(sizes)}/{sorted(sizes)[len(sizes) // 2]}/{max(sizes)}; "
             f"{sum(len(c['text']) > 500 for c in chunks)} chunks exceed the 500-char production prompt view.", ""]
    for i, (c, t) in enumerate(zip(chunks, sizes)):
        pages = f"p.{c['page']}" if c["page"] == c.get("page_end", c["page"]) else f"p.{c['page']}-{c['page_end']}"
        cut = f" — **prompt sees only the first 500 of {len(c['text'])} chars** (cut marked ⟂)" if len(c["text"]) > 500 else ""
        body = c["text"][:500] + "\n⟂\n" + c["text"][500:] if len(c["text"]) > 500 else c["text"]
        lines += [f"## chunk {i} · {pages} · {t} tokens · {' > '.join(c.get('heading_path', [])) or '(no heading)'}{cut}",
                  "", "```text", body, "```", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunkers", nargs="*", default=["structured400"])
    ap.add_argument("--modules", nargs="*", default=["recursive_chunking", "structure_chunking"])
    a = ap.parse_args()
    for m in a.modules:
        try:
            importlib.import_module(f"experiments.quiz_bench.{m}")
        except ModuleNotFoundError:
            pass
    import config
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.LLM_ID)
    ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])  # noqa: E731
    pages = production_pages()
    (OUT / "chunks").mkdir(parents=True, exist_ok=True)
    (OUT / "extracted_pages.md").write_text(
        f"# {FIXTURE} — production extraction (pymupdf4llm), page by page\n\n" +
        "\n\n".join(f"## page {p['metadata']['page_number']} · {ntok(p['text'])} tokens\n\n```markdown\n{p['text']}\n```" for p in pages))
    names = [n for n in sorted(reg.CHUNKERS) if any(n == p or n.startswith(p) for p in a.chunkers)]
    for n in names:
        chunks = reg.CHUNKERS[n](pages)
        (OUT / "chunks" / f"{n}.md").write_text(chunk_md(n, chunks, ntok))
        (OUT / "chunks" / f"{n}.json").write_text(json.dumps(chunks, indent=1))
        print("wrote", n, len(chunks))
    (OUT / "gold.json").write_text(json.dumps(gold(), indent=1))
    outs = sorted(bench_dir().glob("*outputs*.json"))
    if outs:
        (OUT / "generation_outputs").mkdir(exist_ok=True)
        for f in outs:
            shutil.copy(f, OUT / "generation_outputs" / f.name)
            print("copied", f.name)


if __name__ == "__main__":
    main()
