#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export AUM_CHATBOT_HOME="$ROOT"
test -f "$ROOT/data/cos_data.jsonl"
test -f "$ROOT/data/AUM-Housing-Community-Standards.pdf"
conda run --no-capture-output --name aum_chatbot python -m py_compile "$ROOT/config.py" "$ROOT/main.py" "$ROOT/server/api_server.py"
conda run --no-capture-output --name aum_chatbot python -c "import torch,faiss,fastapi,gradio,pdfplumber,rank_bm25,sentence_transformers,transformers; print('dependencies import successfully'); print('CUDA available:',torch.cuda.is_available())"
(cd "$ROOT/desktop/client" && node --check renderer.js && node --check main.js && node --check preload.js)
