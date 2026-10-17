"""Build real-content fixture documents sized to ~8k and ~25k tokens
(measured with the actual Mistral-7B-Instruct-v0.3 tokenizer, not a char
estimate), for benchmarking how far past the app's current 4096-token
cap (pipeline/answer.py) this model can actually go.

Assembled from real AUM content already in this repo -- the Housing
policy PDF, the COS research abstracts, and the deep-research
architecture doc -- concatenated in that order and cut at the first
chunk boundary that crosses each target token count. Not synthetic
filler: a summarizer benchmark on repeated/lorem-ipsum text would not
tell us anything about real-document behavior.
"""

import json
from pathlib import Path

from transformers import AutoTokenizer

import config
from ingestion.document_parser import extract_pdf_pages

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
TARGETS = {"8k": 8000, "12k": 12000, "14k": 14000, "16k": 16000, "25k": 25000}


def _housing_text() -> list[str]:
    pages = extract_pdf_pages(REPO_ROOT / "data" / "AUM-Housing-Community-Standards.pdf")
    return [p["text"] for p in pages if p.get("text", "").strip()]


def _cos_text() -> list[str]:
    # Actual schema is {"id": ..., "text": "Title: ...\nMentor: ...\n\nAbstract:\n..."}
    # -- one already-formatted block per record, nothing to reassemble.
    path = REPO_ROOT / "data" / "cos_data.jsonl"
    blocks = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = row.get("text", "").strip()
            if text:
                blocks.append(text)
    return blocks


def _architecture_text() -> list[str]:
    path = REPO_ROOT / "docs" / "project_guides" / "AUM_DEEP_RESEARCH_ARCHITECTURE_2026-08-24.md"
    text = path.read_text(encoding="utf-8")
    # split on blank lines into paragraph-sized units, same granularity
    # as the other two sources, so the cutoff loop below is fine-grained.
    return [block for block in text.split("\n\n") if block.strip()]


def main():
    tok = AutoTokenizer.from_pretrained(config.LLM_ID, trust_remote_code=True)

    units = _housing_text() + _cos_text() + _architecture_text()
    print(f"Assembled {len(units)} source units from Housing PDF + COS abstracts + architecture doc.")

    FIXTURES_DIR.mkdir(exist_ok=True)
    cursor = 0
    for label, target in sorted(TARGETS.items(), key=lambda kv: kv[1]):
        pieces = []
        token_count = 0
        while cursor < len(units) and token_count < target:
            pieces.append(units[cursor])
            token_count = len(tok.encode("\n\n".join(pieces)))
            cursor += 1
        text = "\n\n".join(pieces)
        actual_tokens = len(tok.encode(text))
        out_path = FIXTURES_DIR / f"doc_{label}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"{label}: wrote {out_path} -- {actual_tokens} tokens, {len(text)} chars, {len(pieces)} units")
        # Next target resumes consuming units right where this one left
        # off (fresh content, not an overlapping superset) -- cursor is
        # deliberately NOT reset between iterations.


if __name__ == "__main__":
    main()
