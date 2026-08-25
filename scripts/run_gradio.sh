#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export AUM_CHATBOT_HOME="$ROOT"
export SCRATCH="$ROOT/data/scratch"
cd "$ROOT"
exec conda run --no-capture-output --name aum_chatbot python main.py
