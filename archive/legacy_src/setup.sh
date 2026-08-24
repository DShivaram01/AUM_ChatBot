#!/usr/bin/env bash
set -e

echo "🔧 CS-BOT setup starting..."

# Go to repo root (directory where this script lives)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# 1) Install Python dependencies
if [ ! -f "requirements.txt" ]; then
  echo "❌ requirements.txt not found in $ROOT_DIR"
  exit 1
fi

echo "📦 Installing Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# 2) Quick smoke test (no big model downloads)
echo "🧪 Running quick smoke test..."

python - << 'PY'
print("➡ Importing core libraries...")
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np

print("✅ Core libraries imported successfully.")

# Optional: tiny embedding check (no network if model already cached)
try:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(["test sentence"], convert_to_numpy=True)
    print("✅ SentenceTransformer basic encode works. Shape:", emb.shape)
except Exception as e:
    print("⚠️ Could not run embedding test:", e)

print("🎉 Setup smoke test finished.")
PY

echo "✅ CS-BOT setup complete."
