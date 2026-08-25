#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
if ! command -v conda >/dev/null 2>&1; then echo "Install Conda or Miniforge first." >&2; exit 1; fi
if conda env list | awk '{print $1}' | grep -qx aum_chatbot; then
  conda env update --name aum_chatbot --file "$ROOT/environment.yml" --prune
else
  conda env create --name aum_chatbot --file "$ROOT/environment.yml"
fi
(cd "$ROOT/desktop/client" && npm ci)
mkdir -p "$ROOT/data/emb_store" "$ROOT/data/scratch" "$ROOT/logs"
echo "Setup complete. Run scripts/verify_install.sh."
