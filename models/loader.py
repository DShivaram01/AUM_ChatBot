"""
models/loader.py
=================
Model loading, extracted from backend.py:
  - load_llm()       <- backend.py:159-181 (tokenizer + conditional cuda/cpu load)
  - load_embedder()  <- backend.py:1464-1466
  - load_reranker()  <- backend.py:1468-1470
  - warmup()         <- backend.py:1473-1477

Unlike the flat script, these are functions that RETURN the loaded objects
instead of assigning module-level globals -- main.py is responsible for
holding onto what they return and passing it to the pipeline/server layers.
"""

import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForCausalLM

import config
from pipeline.memory import logger


def load_embedder():
    logger.info("[Models] Loading embedder...")
    embedder = SentenceTransformer(config.EMBED_MODEL_ID, device=config.DEVICE)
    logger.info("[Models] Embedder ready.")
    return embedder


def load_reranker():
    logger.info("[Models] Loading reranker...")
    reranker = CrossEncoder(config.RERANKER_ID, device=config.DEVICE)
    logger.info("[Models] Reranker ready.")
    return reranker


def _cuda_attention_implementation():
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        logger.warning("[Models] flash-attn unavailable; using PyTorch SDPA.")
        return "sdpa"
    return "flash_attention_2"


def load_llm():
    """
    Load strategy depends on which machine this runs on:
      - GPU (workstation): device_map="auto" lets transformers place the
        model on the GPU automatically; float16 is the standard/fast choice
        when CUDA is available.
      - CPU: .to(DEVICE) explicitly. bfloat16 avoids the NaN-on-CPU failure
        mode float16 can hit on CPU, without float32's ~15GB resident
        footprint. low_cpu_mem_usage=True reduces peak RAM *during*
        loading.
    """
    logger.info(f"[Models] Loading LLM: {config.LLM_ID} on {config.DEVICE}...")

    llm_tok = AutoTokenizer.from_pretrained(config.LLM_ID, trust_remote_code=True)
    llm_tok.pad_token = llm_tok.eos_token

    if config.DEVICE == "cuda":
        # attn_implementation="flash_attention_2" (TASK 13): flash-attention
        # is GPU-only, so this only applies to the CUDA branch -- the CPU
        # branch below must NOT get this kwarg. Requires the flash_attn
        # package to actually import cleanly (see workspace.md TASK 13 log
        # entries for the ABI-mismatch chase that preceded this working);
        # if flash_attn is broken or absent, this line raises instead of
        # silently falling back to eager, which is the right failure mode
        # here -- a silent fallback would hide exactly the kind of
        # regression that already broke model loading once this session.
        llm_model = AutoModelForCausalLM.from_pretrained(
            config.LLM_ID,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation=_cuda_attention_implementation(),
            trust_remote_code=True,
        )
    else:
        llm_model = AutoModelForCausalLM.from_pretrained(
            config.LLM_ID,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(config.DEVICE)

    llm_model.eval()
    logger.info(
        f"[Models] LLM loaded successfully "
        f"({'float16/GPU' if config.DEVICE == 'cuda' else 'bfloat16/CPU'})"
    )
    return llm_tok, llm_model


def warmup(llm_tok, llm_model):
    logger.info("[Models] Warming up LLM...")
    _wi = llm_tok("<s>[INST] hi [/INST]", return_tensors="pt").to(llm_model.device)
    with torch.no_grad():
        llm_model.generate(**_wi, max_new_tokens=1)
    logger.info("[Models] All models ready.")
