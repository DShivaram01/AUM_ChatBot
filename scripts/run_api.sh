#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export AUM_CHATBOT_HOME="$ROOT"
export SCRATCH="$ROOT/data/scratch"
cd "$ROOT"
HEALTH_URL="${AUM_API_URL:-http://127.0.0.1:8000}/api/health"
if curl --silent --fail "$HEALTH_URL" | grep -q '"status":"ok"'; then
  echo "AUM API is already healthy at $HEALTH_URL; reusing it."
  exit 0
fi
exec conda run --no-capture-output --name aum_chatbot python -m server.api_server
