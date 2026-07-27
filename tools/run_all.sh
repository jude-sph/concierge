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
# --factory: the app is built by create_default_app() rather than at import
# time, so importing the module has no filesystem or network side effects.
uv run uvicorn rtvoice.orchestrator:create_default_app --factory --port 8003 --host 127.0.0.1

# Drive the loop from a WAV instead of a browser:
#   uv run python tools/audio_client.py <16khz-mono.wav> --mute --no-concierge
