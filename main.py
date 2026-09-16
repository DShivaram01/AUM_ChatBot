"""
main.py
========
Orchestrates the aum_chatbot startup sequence in the same relative order
backend.py's flat script did (models, then scratch warmup, then COS data +
BM25, then Housing, then the NAME_INDEX/classifier smoke tests), then
launches the standalone Gradio UI when run directly.

server/api_server.py imports load_everything() from here (lazily, at
FastAPI startup) instead of duplicating this sequence -- see its docstring.
"""

import config
from pipeline.memory import logger, _log_path
from pipeline import retrieval
from pipeline.classifier import run_smoke_tests
from models import loader
import server.gradio_ui as gradio_ui

_loaded = False


def load_everything():
    """Loads models + data, builds indices, and wires them into
    server.gradio_ui. Safe to call more than once -- a no-op after the
    first call, so api_server.py's startup event and this file's own
    __main__ block can both call it without double-loading."""
    global _loaded
    if _loaded:
        return

    embedder           = loader.load_embedder()
    reranker            = loader.load_reranker()
    llm_tok, llm_model  = loader.load_llm()
    loader.warmup(llm_tok, llm_model)

    logger.info("[Scratch] Warming up local cache...")
    local_cos_jsonl   = retrieval.scratch_copy(config.COS_JSONL, f"{config.SCRATCH}/cos_data.jsonl")
    local_housing_pdf = retrieval.scratch_copy(config.HOUSING_PDF, f"{config.SCRATCH}/AUM-Housing-Community-Standards.pdf")
    for fname in ["embeddings.npy", "faiss_ip.index", "ids.json", "metadata.json", "texts.json",
                  "embeddings_manifest.json", "housing_chunks.json", "housing_emb.npy", "housing_faiss.index"]:
        retrieval.scratch_copy(f"{config.EMB_STORE}/{fname}", f"{config.SCRATCH}/{fname}")
    logger.info("[Scratch] Done.")

    logger.info("[COS] Loading JSONL...")
    cos_rows = retrieval.load_jsonl(local_cos_jsonl)
    cos_index, cos_EMB, cos_IDS, cos_META, cos_TEXTS = retrieval.build_cos_store(cos_rows, embedder)

    logger.info("[COS] Building BM25...")
    cos_bm25 = retrieval.build_bm25(cos_META, cos_TEXTS)
    logger.info(f"[COS] Ready - {cos_index.ntotal} vectors")

    logger.info("[Housing] Building/loading...")
    H_index, H_EMB, housing_chunks = retrieval.load_or_build_housing(local_housing_pdf, embedder)
    housing_ok = H_index is not None
    if housing_ok:
        logger.info(f"[Housing] Ready - {len(housing_chunks)} chunks")
    else:
        logger.warning("[Housing] PDF not found - Housing tab will show error")

    # Build the runtime name index during normal ingest/startup. Smoke tests
    # validate it afterwards; they no longer create application state.
    retrieval.build_name_index(cos_META)
    run_smoke_tests(cos_META)

    gradio_ui.init_gradio_ui(
        embedder_=embedder, reranker_=reranker, llm_tok_=llm_tok, llm_model_=llm_model,
        cos_index_=cos_index, cos_EMB_=cos_EMB, cos_META_=cos_META, cos_TEXTS_=cos_TEXTS, cos_bm25_=cos_bm25,
        H_index_=H_index, H_EMB_=H_EMB, housing_chunks_=housing_chunks, housing_ok_=housing_ok,
        log_path_=_log_path,
    )
    _loaded = True


if __name__ == "__main__":
    load_everything()
    gradio_ui.launch()
