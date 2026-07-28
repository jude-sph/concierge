#!/usr/bin/env bash
# Speech synthesis, on whichever machine can host Chatterbox.
#
# Not the demo box: chatterbox-tts requires numpy<2, and downgrading numpy
# there would break the turn-taking model that shares that environment. It is
# also at 97% disk with ~3.4GB of VRAM left. A laptop is fine -- ~1-2GB, and
# MPS or CPU both work.
#
# Then point the orchestrator at it:  export TTS_URL=http://127.0.0.1:8020
# (over a reverse tunnel if it is on a different machine:
#      ssh -R 8020:127.0.0.1:8020 <demo-host>)
# Unset TTS_URL and everything falls straight back to Kokoro, in process.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# Chatterbox has no voice list -- a voice IS a reference recording, cloned
# zero-shot, and turbo's own default speaker differs from the standard model's.
# This clip is the standard model's male voice, captured so turbo can wear it.
export TTS_VOICE_REF="${TTS_VOICE_REF:-$HERE/scripts/voices/chatterbox_male.wav}"
export PYTHONPATH="$HERE"

exec "${TTS_PYTHON:-python}" -m uvicorn scripts.tts_server:app \
     --host 127.0.0.1 --port "${TTS_PORT:-8020}"
