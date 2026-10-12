"""Benchmark fixture: the private benchmark paper + its provisional gold set.

Neither file is in git (unpublished manuscript). benchmarks/quiz_paper.json
records their hashes; the files live in QZ_BENCH_DIR (default
~/aum_benchmarks/quiz_paper) and are verified on load.
"""

import hashlib
import json
import os
import re
import tempfile
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# Select the fixture with QZ_FIXTURE (a benchmarks/<name>.json); the
# research paper (quiz_paper) is inactive at the owner's request.
FIXTURE = os.environ.get("QZ_FIXTURE", "dsa_textbook")
CONFIG = ROOT / "benchmarks" / f"{FIXTURE}.json"


def bench_dir() -> Path:
    default = Path.home() / "aum_benchmarks" / json.loads(CONFIG.read_text()).get("bench_subdir", FIXTURE)
    return Path(os.environ.get("QZ_BENCH_DIR", default))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def config() -> dict:
    return json.loads(CONFIG.read_text())


def paper_bytes() -> bytes:
    path = bench_dir() / config()["paper_file"]
    if _sha(path) != config()["paper_sha256"]:
        raise RuntimeError(f"{path} does not match the recorded benchmark hash")
    return path.read_bytes()


@lru_cache(maxsize=1)
def gold() -> dict:
    data = json.loads((bench_dir() / config()["gold_file"]).read_text())
    if data["document_sha256"] != config()["paper_sha256"]:
        raise RuntimeError("gold set was built for a different document")
    return data


@lru_cache(maxsize=1)
def production_pages() -> list[dict]:
    """Pages exactly as production extracts them (pymupdf4llm page chunks)."""
    from ingestion.document_parser import extract_pdf_pages
    with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
        f.write(paper_bytes()); f.flush()
        return [{"metadata": {"page_number": int(p["metadata"]["page_number"])}, "text": p["text"]}
                for p in extract_pdf_pages(Path(f.name))]


def norm(text: str) -> str:
    """Normalisation used for anchor matching: drop markdown emphasis, collapse
    whitespace, lowercase."""
    return re.sub(r"\s+", " ", re.sub(r"[*_]", "", text)).strip().lower()
