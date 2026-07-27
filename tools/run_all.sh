#!/usr/bin/env bash
# tools/run_all.sh — run ON the 3090
set -euo pipefail
eval "$(conda shell.bash hook)"

conda activate soulx-duplug
( cd ~/SoulX-Duplug && bash run.sh ) &          # SoulX-Duplug on :8000

vllm serve Qwen/Qwen3-4B  --port 8001 --gpu-memory-utilization 0.25 \
     --max-model-len 4096 &                     # concierge
# The reasoner STUB runs in-process inside the orchestrator (see _default_app).
# This vLLM instance is only needed for the optional LLM-roleplay reasoner path;
# omit it to save ~8GB of VRAM while developing.
# vllm serve Qwen/Qwen3-8B --port 8002 --gpu-memory-utilization 0.40 \
#      --max-model-len 8192 &

cd ~/rtvoice
uv run uvicorn rtvoice.orchestrator:app --port 8003 --host 127.0.0.1
