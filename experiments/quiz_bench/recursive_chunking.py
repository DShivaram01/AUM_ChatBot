"""Experiment 1: token-aware recursive text splitting (no new dependencies).

Splits the concatenated document text by a separator hierarchy
(paragraph -> line -> sentence -> word), packing pieces up to `size` Mistral
tokens with `overlap` tokens carried into the next chunk. Page provenance is
kept via character offsets; heading_path is the most recent markdown heading
before the chunk starts (provenance only -- headings do not force splits,
which is the difference from the production structured chunker).
"""

import bisect
import re
from functools import lru_cache

from experiments.quiz_bench.chunkers import register

SEPARATORS = ["\n\n", "\n", ". ", " "]
SIZES = [256, 400, 512, 768, 1024]
OVERLAPS = [0, 0.15]   # fraction of size


@lru_cache(maxsize=1)
def _tokenizer():
    import config
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(config.LLM_ID)


def ntok(text: str) -> int:
    return len(_tokenizer()(text, add_special_tokens=False)["input_ids"])


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    """Pieces no longer than `size` tokens, splitting on the coarsest separator first."""
    if ntok(text) <= size or not seps:
        return [text]
    sep, rest = seps[0], seps[1:]
    parts = [p + sep for p in text.split(sep)]
    parts[-1] = parts[-1][: len(parts[-1]) - len(sep)]
    out = []
    for p in parts:
        out.extend(_split(p, size, rest) if ntok(p) > size else [p])
    return out


def recursive_chunks(pages: list[dict], size: int, overlap_frac: float) -> list[dict]:
    text, starts = "", []
    for p in pages:
        starts.append(len(text))
        text += p["text"].strip() + "\n\n"
    page_nums = [p["metadata"]["page_number"] for p in pages]
    headings = [(m.start(), re.sub(r"[*#]", "", m.group(0)).strip()) for m in re.finditer(r"(?m)^#{1,6} .+$", text)]

    pieces, pos = [], 0
    for piece in _split(text, size, SEPARATORS):
        pieces.append((pos, piece))
        pos += len(piece)
    overlap = int(size * overlap_frac)
    chunks, cur, cur_tok = [], [], 0
    for start, piece in pieces:
        t = ntok(piece)
        if cur and cur_tok + t > size:
            chunks.append(cur)
            carry, carry_tok = [], 0
            for s, pc in reversed(cur):          # overlap: re-use trailing pieces
                pt = ntok(pc)
                if carry_tok + pt > overlap:
                    break
                carry.insert(0, (s, pc)); carry_tok += pt
            cur, cur_tok = carry, carry_tok
        cur.append((start, piece)); cur_tok += t
    if cur:
        chunks.append(cur)

    out = []
    for c in chunks:
        body = "".join(p for _, p in c).strip()
        if not body:
            continue
        s, e = c[0][0], c[-1][0] + len(c[-1][1]) - 1
        hi = bisect.bisect_right([h for h, _ in headings], s) - 1
        out.append({"text": body,
                    "page": page_nums[bisect.bisect_right(starts, s) - 1],
                    "page_end": page_nums[bisect.bisect_right(starts, e) - 1],
                    "heading_path": [headings[hi][1]] if hi >= 0 else []})
    return out


for _size in SIZES:
    for _ov in OVERLAPS:
        register(f"recursive{_size}_ov{int(_ov * 100)}")(
            lambda pages, s=_size, o=_ov: recursive_chunks(pages, s, o))
