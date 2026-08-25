"""
config.py
=========
Central configuration for the reorganised aum_chatbot/ project.

Values are carried over from the flat /home/gh204/Desktop/backend.py script
(see workspace.md TASK 11 log entry for the exact source line numbers each
value was taken from). Paths are repointed at the new project root,
/home/gh204/Desktop/aum_chatbot/, instead of the old
/home/gh204/Desktop/COS_dataset/COS_dataset/ location.

Two flagged deviations from a literal source-extraction (see TASK 11 log
entry for full detail, not just the notes below):
  - LOG_DIR points at aum_chatbot/logs/ (a top-level sibling of data/),
    matching the directory STEP 1 of TASK 11 actually created, NOT
    literally under aum_chatbot/data/ as the task text's constant list
    summary said -- that summary line was inconsistent with the actual
    mkdir layout and the original backend.py, where LOG_DIR was also a
    sibling of (not nested under) the data paths.
  - MIN_FREE_RAM_FOR_LLM_GB does not exist anywhere in backend.py (no
    RAM-checking logic of any kind was found there). It is set to None
    here rather than a fabricated number -- there is no "actual value
    from backend.py" to extract because this constant was never defined
    in the source file.
"""

import os
import torch

# ---- Project root ----
# backend.py's BASE_DIR was the COS_dataset/COS_dataset directory (parent
# of its logs/ and scratch/). Here BASE_DIR is the new project root, and
# LOG_DIR / SCRATCH are its logs/ and data/scratch/ subdirectories, same
# relationship backend.py had, just rooted at the new location.
BASE_DIR = os.path.abspath(os.environ.get("AUM_CHATBOT_HOME", os.path.dirname(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---- Data files (paths, not literal values, in backend.py's COS_JSONL /
# HOUSING_PDF / EMB_STORE -- not in the task's explicit constant list, but
# included because pipeline/retrieval.py cannot locate the data without
# them) ----
COS_JSONL   = os.path.join(DATA_DIR, "cos_data.jsonl")
HOUSING_PDF = os.path.join(DATA_DIR, "AUM-Housing-Community-Standards.pdf")
EMB_STORE   = os.path.join(DATA_DIR, "emb_store")

LOG_DIR = os.path.join(BASE_DIR, "logs")
SCRATCH = os.environ.get("SCRATCH", os.path.join(DATA_DIR, "scratch"))

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)
os.makedirs(EMB_STORE, exist_ok=True)

# ---- Models (backend.py:121-131, current values as of TASK 3/4) ----
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_ID    = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_ID         = "mistralai/Mistral-7B-Instruct-v0.3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Retrieval / thresholding (backend.py:137-138, 697-698) ----
RERANK_THRESHOLD = 0.0
RRF_K            = 60    # standard RRF constant
SIGMA            = 0.8   # relative-threshold multiplier (backend.py:697)

# ---- NOT FOUND IN backend.py ----
# No RAM-checking logic exists anywhere in the ~2227-line source file
# (verified by a full read, not just grep). Left as None rather than an
# invented number; set a real value here before adding any RAM-gating
# behavior to models/loader.py.
MIN_FREE_RAM_FOR_LLM_GB = None
