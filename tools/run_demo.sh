#!/usr/bin/env bash
# tools/run_demo.sh — launch the browser demo UI.
#
# v1 defaults: no concierge (no vLLM needed), reasoner stub only, bound to
# localhost on an unusual port so it doesn't collide with anything else on a
# shared lab machine. The SoulX-Duplug turn-taking server (SOULX_URL, default
# ws://localhost:8000/turn) is still required -- that piece is out of scope
# for this script, same as tools/run_all.sh.
#
# Usage:
#   tools/run_demo.sh
#   PORT=9001 USE_CONCIERGE=1 tools/run_demo.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$HOME/.cache/modelscope}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="src:${PYTHONPATH:-}"

# Off by default: v1 needs only the turn-taking server + reasoner stub, no
# vLLM. Set USE_CONCIERGE=1 once a concierge vLLM server is up.
export USE_CONCIERGE="${USE_CONCIERGE:-0}"

HOST=127.0.0.1
PORT="${PORT:-8977}"

echo "rtvoice demo: http://${HOST}:${PORT}  (USE_CONCIERGE=${USE_CONCIERGE})"
echo "Needs the SoulX-Duplug turn-taking server reachable at SOULX_URL (default ws://localhost:8000/turn)."
exec uv run uvicorn rtvoice.orchestrator:create_default_app --factory --host "$HOST" --port "$PORT"
