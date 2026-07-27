# Real-Time Voice Interface — Core Loop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a working headless voice loop where a human speaks a compound command, a stubbed reasoner mutates a fake device state, and the result is spoken back — with a complete event log for every session.

**Architecture:** A cascade. SoulX-Duplug provides semantic turn-taking over the user's audio; an orchestrator runs a turn policy, a task registry and a small "concierge" LLM in parallel with a larger "reasoner" LLM that owns device truth. All models run on a remote 3090 reached through an SSH tunnel. Everything renders from an append-only JSONL event log.

**Tech Stack:** Python 3.10, uv, FastAPI, pytest, vLLM (OpenAI-compatible), SoulX-Duplug 0.6B, SenseVoice Small (ASR, via SoulX), Kokoro (TTS), pydantic v2.

**Scope note:** This plan delivers the headless loop only — driveable via manual text injection and a CLI audio client, verifiable by asserting on `device_state.json`. The instrument UI, browser audio client, and replay controls are a **separate follow-up plan** that renders from the event log produced here. The event log is the interface between the two.

## Global Constraints

- **Python 3.10 exactly.** SoulX-Duplug pins it; newer versions lack wheels for its dependencies.
- **Never install SoulX-Duplug's `requirements.txt`.** It is a full `pip freeze` dump (~400 pinned packages including Aliyun SDKs and `bitsandbytes`). Install the minimal set explicitly.
- **All audio is 16 kHz, mono, float32.** SoulX-Duplug's wire format is JSON text frames carrying base64-encoded float32 PCM — not binary frames, not int16.
- **Chunk size is 2560 samples (160 ms).** Matches `config.yaml`'s `chunk_size`.
- **The reasoner is the gatekeeper.** Every finalised utterance goes to it. The concierge never decides what is actionable.
- **No pattern matching on generated prose anywhere.** Concierge output is validated as a typed schema only.
- **Cancellation never depends on the reasoner receiving a message.** The orchestrator fires a token; notifying the reasoner is secondary.
- **SSH host alias `gpu3090` must resolve** with `LocalForward` for ports 8000–8003. Task 1 establishes this.
- Spec: `docs/superpowers/specs/2026-07-27-realtime-voice-interface-design.md`

## File Structure

```
src/rtvoice/
  states.py           # domain states + SoulX wire adapter (pure)
  events.py           # event types + JSONL event log
  registry.py         # task registry state machine (pure)
  cancellation.py     # cancellation tokens
  policy.py           # turn policy (pure)
  protocol.py         # orchestrator <-> reasoner message types
  concierge.py        # speech acts, schema validation, vLLM client
  device.py           # journaled device state (reasoner-stub side)
  reasoner_stub.py    # roleplay agent + protocol server
  soulx_client.py     # WebSocket client to SoulX-Duplug
  tts.py              # Kokoro streaming wrapper
  recorder.py         # dual-channel WAV session recorder
  voice_service.py    # FastAPI: audio in, domain events out
  orchestrator.py     # FastAPI: wiring, event WS, manual injection
tools/
  measure_tunnel.py   # RTT/jitter measurement
  make_corpus.py      # disfluent compound-command corpus
  measure_turntaking.py
  sync.sh             # rsync to the 3090
tests/
  test_states.py  test_events.py  test_registry.py
  test_cancellation.py  test_policy.py  test_concierge.py
  test_device.py  test_end_to_end.py
```

Pure-logic modules (`states`, `registry`, `cancellation`, `policy`, `concierge` validation, `device`) have no network or model dependencies and carry the bulk of the test suite.

---

## Phase 0 — De-risk

These three tasks come first because either of their measurements can invalidate design assumptions before any code is built on them.

### Task 1: Repo scaffold, remote sync, and tunnel measurement

**Files:**
- Create: `pyproject.toml`, `tools/sync.sh`, `tools/measure_tunnel.py`
- Test: `tests/test_scaffold.py`

**Interfaces:**
- Consumes: nothing
- Produces: `uv` project with `pytest` runnable; `tools/sync.sh` pushing the repo to `gpu3090:~/rtvoice`; `measure_tunnel.py` printing RTT percentiles.

- [ ] **Step 1: Create the uv project**

```bash
uv init --python 3.10 --no-workspace
uv add --dev pytest pytest-asyncio
uv add fastapi uvicorn websockets numpy soundfile pydantic httpx
```

- [ ] **Step 2: Write a scaffold test**

```python
# tests/test_scaffold.py
import sys

def test_python_version_is_310():
    assert sys.version_info[:2] == (3, 10), (
        "SoulX-Duplug pins Python 3.10; newer versions lack wheels for its deps"
    )

def test_package_imports():
    import rtvoice  # noqa: F401
```

- [ ] **Step 3: Run it and watch it fail**

Run: `uv run pytest tests/test_scaffold.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice'`

- [ ] **Step 4: Create the package**

```bash
mkdir -p src/rtvoice && touch src/rtvoice/__init__.py
```

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["src"]
asyncio_mode = "auto"
```

- [ ] **Step 5: Run it and watch it pass**

Run: `uv run pytest tests/test_scaffold.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Write the sync script**

```bash
# tools/sync.sh
#!/usr/bin/env bash
set -euo pipefail
REMOTE="${REMOTE:-gpu3090}"
DEST="${DEST:-~/rtvoice}"
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude 'sessions' --exclude 'pretrained_models' \
  ./ "${REMOTE}:${DEST}/"
echo "synced to ${REMOTE}:${DEST}"
```

Then `chmod +x tools/sync.sh`.

- [ ] **Step 7: Write the tunnel measurement tool**

```python
# tools/measure_tunnel.py
"""Measure RTT and jitter to the 3090 through the SSH tunnel.

Run the remote echo server first:
    ssh gpu3090 'python3 -m http.server 8003'
Then locally:
    uv run python tools/measure_tunnel.py --n 200
"""
import argparse, socket, statistics, time


def probe(host: str, port: int, timeout: float = 2.0) -> float | None:
    start = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return (time.perf_counter() - start) * 1000.0
    except OSError:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8003)
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()

    samples = []
    failures = 0
    for _ in range(args.n):
        rtt = probe(args.host, args.port)
        if rtt is None:
            failures += 1
        else:
            samples.append(rtt)
        time.sleep(0.05)

    if not samples:
        print("all probes failed - is the tunnel up?")
        return

    samples.sort()
    p = lambda q: samples[min(int(len(samples) * q), len(samples) - 1)]
    print(f"n={len(samples)} failures={failures}")
    print(f"mean   {statistics.mean(samples):7.1f} ms")
    print(f"p50    {p(0.50):7.1f} ms")
    print(f"p95    {p(0.95):7.1f} ms")
    print(f"p99    {p(0.99):7.1f} ms")
    print(f"max    {samples[-1]:7.1f} ms")
    print(f"jitter {statistics.pstdev(samples):7.1f} ms (stdev)")
    print()
    print("DECISION: p95 > 150ms or jitter > 50ms means move voice-service")
    print("to the Mac (spec: 'Escape hatch'). Otherwise keep all on the 3090.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 8: Set up the SSH alias and run the measurement**

Add to `~/.ssh/config` (substituting real hostnames):

```
Host gpu3090
    HostName <box-hostname>
    User <user>
    ProxyJump <user>@<uni-gateway>
    LocalForward 8000 localhost:8000
    LocalForward 8001 localhost:8001
    LocalForward 8002 localhost:8002
    LocalForward 8003 localhost:8003
    ServerAliveInterval 30
```

Then in one terminal `ssh gpu3090 'python3 -m http.server 8003'`, and locally:

Run: `uv run python tools/measure_tunnel.py --n 200`
Expected: printed percentiles. **Record the numbers in the commit message** — they decide the deployment shape.

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml tools/ tests/ src/
git commit -m "feat: scaffold, remote sync, tunnel measurement

Tunnel: p50=<X>ms p95=<Y>ms jitter=<Z>ms"
```

---

### Task 2: Deploy SoulX-Duplug and prove the wire protocol

**Files:**
- Create: `tools/soulx_setup.sh`, `src/rtvoice/soulx_client.py`
- Test: `tests/test_soulx_client.py`

**Interfaces:**
- Consumes: `gpu3090` SSH alias from Task 1
- Produces: `SoulXClient.feed(chunk: np.ndarray) -> dict` returning the raw wire dict with keys `state` (`"idle"|"nonidle"|"speak"|"blank"`), and situationally `asr_buffer`, `asr_segment`, `text`.

- [ ] **Step 1: Write the remote setup script**

```bash
# tools/soulx_setup.sh — run ON the 3090, not locally
#!/usr/bin/env bash
set -euo pipefail
cd ~
git clone https://github.com/Soul-AILab/SoulX-Duplug.git || true
cd SoulX-Duplug

conda create -n soulx-duplug -y python=3.10 || true
eval "$(conda shell.bash hook)" && conda activate soulx-duplug

# Minimal deps ONLY. Do not use requirements.txt (400+ pins, bitsandbytes, Aliyun SDKs).
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft accelerate funasr modelscope \
            fastapi uvicorn numpy soundfile librosa websocket-client pyyaml

mkdir -p pretrained_models
huggingface-cli download Soul-AILab/SoulX-Duplug --local-dir pretrained_models/SoulX-Duplug
huggingface-cli download THUDM/glm-4-voice-tokenizer --local-dir pretrained_models/glm-4-voice-tokenizer
huggingface-cli download Qwen/Qwen3-0.6B --local-dir pretrained_models/Qwen3-0.6B-expand_vocab_v2

# English config
python - <<'PY'
import yaml, pathlib
p = pathlib.Path("config/config.yaml")
c = yaml.safe_load(p.read_text())
c["infer_config"]["asr"] = {"model_name": "sensevoice", "language": "en"}
p.write_text(yaml.safe_dump(c, sort_keys=False))
print("config set to sensevoice/en")
PY

echo "now run: bash run.sh"
```

- [ ] **Step 2: Write the failing client test**

```python
# tests/test_soulx_client.py
import numpy as np
from rtvoice.soulx_client import encode_chunk, decode_response, CHUNK_SAMPLES


def test_chunk_size_is_160ms_at_16khz():
    assert CHUNK_SAMPLES == 2560


def test_encode_chunk_is_base64_float32():
    import base64
    chunk = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    payload = encode_chunk("sess-1", chunk)
    assert payload["type"] == "audio"
    assert payload["session_id"] == "sess-1"
    raw = base64.b64decode(payload["audio"])
    assert np.allclose(np.frombuffer(raw, dtype=np.float32), chunk)


def test_decode_response_extracts_inner_state():
    wire = {"type": "turn_state", "session_id": "s",
            "state": {"state": "speak", "text": "book me a table"}, "ts": 1.0}
    assert decode_response(wire) == {"state": "speak", "text": "book me a table"}
```

- [ ] **Step 3: Run it and watch it fail**

Run: `uv run pytest tests/test_soulx_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.soulx_client'`

- [ ] **Step 4: Implement the client**

```python
# src/rtvoice/soulx_client.py
"""WebSocket client for the SoulX-Duplug turn-taking server.

Wire protocol (from Soul-AILab/SoulX-Duplug server.py):
  send: {"type":"audio","session_id":str,"audio":base64(float32 PCM)}
  recv: {"type":"turn_state","session_id":str,"state":{...},"ts":float}

Inner state dict:
  {"state": "idle"|"nonidle"|"speak"|"blank",
   "asr_buffer": str,   # last ~3.2s, present when nonidle
   "asr_segment": str,  # current chunk, present when nonidle
   "text": str}         # full utterance, present when speak
"""
from __future__ import annotations

import base64
import json
import uuid

import numpy as np
import websockets

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 2560  # 160 ms, matches config.yaml chunk_size


def encode_chunk(session_id: str, chunk: np.ndarray) -> dict:
    audio = np.asarray(chunk, dtype=np.float32)
    return {
        "type": "audio",
        "session_id": session_id,
        "audio": base64.b64encode(audio.tobytes()).decode(),
    }


def decode_response(wire: dict) -> dict:
    """Pull the inner state dict out of the envelope."""
    return wire.get("state", {})


class SoulXClient:
    def __init__(self, url: str = "ws://localhost:8000/turn", session_id: str | None = None):
        self.url = url
        self.session_id = session_id or uuid.uuid4().hex
        self._ws = None

    async def connect(self) -> None:
        self._ws = await websockets.connect(self.url, max_size=None)

    async def feed(self, chunk: np.ndarray) -> dict:
        """Send one chunk, return the inner state dict."""
        if self._ws is None:
            await self.connect()
        await self._ws.send(json.dumps(encode_chunk(self.session_id, chunk)))
        raw = await self._ws.recv()
        return decode_response(json.loads(raw))

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
```

- [ ] **Step 5: Run it and watch it pass**

Run: `uv run pytest tests/test_soulx_client.py -v`
Expected: PASS (3 passed)

- [ ] **Step 6: Smoke-test against the real server**

Sync and start the server:

```bash
./tools/sync.sh
ssh gpu3090 'cd ~/SoulX-Duplug && bash tools/soulx_setup.sh'
ssh gpu3090 'cd ~/SoulX-Duplug && conda run -n soulx-duplug bash run.sh'
```

With the tunnel up, run this smoke script locally:

```python
# tools/smoke_soulx.py
import asyncio, numpy as np, soundfile as sf, sys
from rtvoice.soulx_client import SoulXClient, CHUNK_SAMPLES

async def main(path):
    audio, sr = sf.read(path, dtype="float32")
    assert sr == 16000, f"expected 16kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    c = SoulXClient()
    await c.connect()
    for i in range(0, len(audio) - CHUNK_SAMPLES, CHUNK_SAMPLES):
        st = await c.feed(audio[i:i + CHUNK_SAMPLES])
        if st.get("state") != "blank":
            print(f"{i/16000:6.2f}s  {st.get('state'):8s}  {st.get('text') or st.get('asr_buffer','')}")
    await c.close()

asyncio.run(main(sys.argv[1]))
```

Run: `uv run python tools/smoke_soulx.py <a-16khz-wav-of-you-speaking>`
Expected: a stream of `idle`/`nonidle` lines ending in a `speak` with your transcript.

- [ ] **Step 7: Commit**

```bash
git add tools/soulx_setup.sh tools/smoke_soulx.py src/rtvoice/soulx_client.py tests/test_soulx_client.py
git commit -m "feat: SoulX-Duplug deployment script and wire client"
```

---

### Task 3: Measure turn-taking on disfluent compound commands

This is the measurement most likely to invalidate the design. The paper reports 0.352 pause-takeover on their English set; the open question is what it is on *our* command style.

**Files:**
- Create: `tools/make_corpus.py`, `tools/measure_turntaking.py`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `SoulXClient` from Task 2
- Produces: `corpus/*.wav` plus `corpus/manifest.json` with per-clip `pause_ms_marks`; a printed false-takeover rate.

- [ ] **Step 1: Write the corpus manifest test**

```python
# tests/test_corpus.py
from tools.make_corpus import build_manifest, COMMANDS


def test_commands_contain_midsentence_pauses():
    assert all("|" in c for c in COMMANDS), "each command needs a | pause marker"


def test_manifest_records_pause_offsets():
    m = build_manifest("find chinese food | and rename my contacts", words_per_sec=3.0)
    assert m["text"] == "find chinese food and rename my contacts"
    assert m["pause_marks_sec"] == [1.0]  # 3 words at 3 w/s
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'tools.make_corpus'`

- [ ] **Step 3: Implement corpus generation**

```python
# tools/make_corpus.py
"""Generate disfluent compound commands in *our* style and TTS them.

'|' marks a mid-sentence pause: the point where SoulX-Duplug must NOT take
the turn. A takeover at a '|' is a false positive.
"""
from __future__ import annotations

import json
import pathlib

COMMANDS = [
    "find all the chinese restaurants in soho | and set all my contacts to Hans",
    "book me a table for four | uh | and text sarah about it",
    "change my work contacts | the ones in the london group | to Hans",
    "get me an uber to the station | actually | make that the airport",
    "delete the messages from yesterday | and | add a reminder for tuesday",
    "what's in my calendar tomorrow | and can you move the 3pm | to thursday",
    "search for indian food nearby | then | call the first one",
    "rename everyone in my favourites | um | to Hans please",
]


def build_manifest(command: str, words_per_sec: float = 3.0) -> dict:
    """Compute where pause marks fall in seconds, assuming a constant speech rate."""
    marks = []
    spoken = []
    for part in command.split("|"):
        words = part.split()
        spoken.extend(words)
        marks.append(len(spoken) / words_per_sec)
    return {
        "text": " ".join(spoken),
        "pause_marks_sec": marks[:-1],  # drop the trailing end-of-utterance
    }


def main() -> None:
    out = pathlib.Path("corpus")
    out.mkdir(exist_ok=True)
    manifest = []
    for i, cmd in enumerate(COMMANDS):
        m = build_manifest(cmd)
        m["id"] = f"cmd_{i:02d}"
        m["wav"] = f"{m['id']}.wav"
        m["source"] = cmd
        manifest.append(m)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(manifest)} entries to {out/'manifest.json'}")
    print("Now synthesise each 'text' with pauses at '|' into corpus/<id>.wav")
    print("at 16kHz mono. Use Kokoro (Task 11) or record yourself reading them.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run it and watch it pass**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Implement the measurement harness**

```python
# tools/measure_turntaking.py
"""Feed the corpus through SoulX-Duplug and count false takeovers.

A false takeover = a 'speak' state emitted within TOLERANCE_SEC of a marked
mid-sentence pause. That is the number that decides whether the design holds.
"""
from __future__ import annotations

import asyncio, json, pathlib
import numpy as np, soundfile as sf

from rtvoice.soulx_client import SoulXClient, CHUNK_SAMPLES

TOLERANCE_SEC = 1.0


async def run_clip(entry: dict, root: pathlib.Path) -> dict:
    audio, sr = sf.read(root / entry["wav"], dtype="float32")
    assert sr == 16000
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    client = SoulXClient()
    await client.connect()
    speaks = []
    for i in range(0, len(audio) - CHUNK_SAMPLES, CHUNK_SAMPLES):
        st = await client.feed(audio[i:i + CHUNK_SAMPLES])
        if st.get("state") == "speak":
            speaks.append(i / 16000)
    await client.close()

    false_takeovers = [
        t for t in speaks
        if any(abs(t - m) < TOLERANCE_SEC for m in entry["pause_marks_sec"])
    ]
    return {
        "id": entry["id"],
        "speaks": speaks,
        "pauses": entry["pause_marks_sec"],
        "false_takeovers": len(false_takeovers),
        "n_pauses": len(entry["pause_marks_sec"]),
        "took_final_turn": bool(speaks) and not false_takeovers,
    }


async def main() -> None:
    root = pathlib.Path("corpus")
    manifest = json.loads((root / "manifest.json").read_text())
    results = [await run_clip(e, root) for e in manifest if (root / e["wav"]).exists()]

    total_pauses = sum(r["n_pauses"] for r in results)
    total_false = sum(r["false_takeovers"] for r in results)
    rate = total_false / total_pauses if total_pauses else 0.0

    for r in results:
        print(f"{r['id']}  false={r['false_takeovers']}/{r['n_pauses']}  speaks={[round(s,2) for s in r['speaks']]}")
    print()
    print(f"FALSE TAKEOVER RATE: {rate:.3f}  ({total_false}/{total_pauses})")
    print(f"paper baseline (their EN set): 0.352")
    print()
    if rate > 0.5:
        print("DESIGN RISK: over half of mid-sentence pauses trigger a takeover.")
        print("Mitigations: raise max_wait_num in config.yaml, or add a hold-off")
        print("timer in the turn policy before dispatching on user_complete.")


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: Generate the corpus and measure**

```bash
uv run python tools/make_corpus.py
# synthesise or record corpus/*.wav at 16kHz mono, then:
uv run python tools/measure_turntaking.py
```

Expected: a printed false-takeover rate. **Record it in the commit message.** If it exceeds 0.5, stop and revisit the turn policy before continuing — Task 7 will need a hold-off timer.

- [ ] **Step 7: Commit**

```bash
git add tools/make_corpus.py tools/measure_turntaking.py tests/test_corpus.py corpus/manifest.json
git commit -m "feat: disfluent compound-command corpus and turn-taking measurement

False takeover rate: <X> (paper baseline 0.352)"
```

---

## Phase 1 — Pure logic

No models, no network. These tests run in milliseconds and carry most of the build's correctness.

### Task 4: Domain states and the SoulX wire adapter

**Files:**
- Create: `src/rtvoice/states.py`
- Test: `tests/test_states.py`

**Interfaces:**
- Consumes: nothing
- Produces: `UserState` enum; `TurnEvent(state: UserState, transcript: str, t_ms: int)`; `StateAdapter.feed(wire: dict, t_ms: int) -> list[TurnEvent]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_states.py
from rtvoice.states import StateAdapter, UserState


def feed_all(adapter, wires):
    out = []
    for i, w in enumerate(wires):
        out.extend(adapter.feed(w, t_ms=i * 160))
    return [e.state for e in out]


def test_blank_produces_no_event():
    a = StateAdapter()
    assert a.feed({"state": "blank"}, 0) == []


def test_idle_and_nonidle_map_directly():
    a = StateAdapter()
    assert feed_all(a, [{"state": "idle"}]) == [UserState.IDLE]
    assert feed_all(StateAdapter(), [{"state": "nonidle", "asr_buffer": "hi"}]) == [
        UserState.NONIDLE
    ]


def test_speak_becomes_complete_with_transcript():
    a = StateAdapter()
    events = a.feed({"state": "speak", "text": "book me a table"}, 0)
    assert len(events) == 1
    assert events[0].state == UserState.COMPLETE
    assert events[0].transcript == "book me a table"


def test_speak_with_backchannel_text_becomes_backchannel():
    a = StateAdapter()
    events = a.feed({"state": "speak", "text": "mm hm"}, 0)
    assert events[0].state == UserState.BACKCHANNEL


def test_incomplete_is_inferred_from_nonidle_to_idle_without_speak():
    """The model declining to take the turn IS the incompleteness signal."""
    a = StateAdapter()
    states = feed_all(a, [
        {"state": "nonidle", "asr_buffer": "find chinese food and"},
        {"state": "idle"},
    ])
    assert states == [UserState.NONIDLE, UserState.INCOMPLETE, UserState.IDLE]


def test_no_incomplete_after_a_completed_turn():
    a = StateAdapter()
    states = feed_all(a, [
        {"state": "nonidle", "asr_buffer": "book a table"},
        {"state": "speak", "text": "book a table"},
        {"state": "idle"},
    ])
    assert UserState.INCOMPLETE not in states


def test_incomplete_carries_the_partial_transcript():
    a = StateAdapter()
    a.feed({"state": "nonidle", "asr_buffer": "find chinese food and"}, 0)
    events = a.feed({"state": "idle"}, 160)
    assert events[0].state == UserState.INCOMPLETE
    assert events[0].transcript == "find chinese food and"
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_states.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.states'`

- [ ] **Step 3: Implement**

```python
# src/rtvoice/states.py
"""Domain turn states, derived from SoulX-Duplug's four wire states.

SoulX-Duplug's shipped server emits: idle | nonidle | speak | blank.
The paper's five semantic states are recovered as follows:

  blank            -> (no event; insufficient audio buffered)
  idle             -> user_idle
  nonidle          -> user_nonidle
  speak            -> user_complete, or user_backchannel if the text is a backchannel
  nonidle -> idle  -> user_incomplete (inferred: the model declined the turn)

The last is the important one: there is no wire state for "paused but not
finished". The absence of a `speak` between speech and silence IS the signal.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Mirrors utils/backchannel_utils.py in Soul-AILab/SoulX-Duplug, English subset.
_BACKCHANNELS = {
    "", "mm", "mhm", "mm hm", "mmhm", "uh huh", "uhhuh", "ah", "oh", "ok",
    "okay", "yeah", "yep", "yes", "right", "sure", "hmm", "huh", "i see",
}


def is_backchannel(text: str) -> bool:
    cleaned = text.strip().lower().rstrip(".,!?").strip()
    return cleaned in _BACKCHANNELS


class UserState(str, Enum):
    IDLE = "user_idle"
    NONIDLE = "user_nonidle"
    BACKCHANNEL = "user_backchannel"
    COMPLETE = "user_complete"
    INCOMPLETE = "user_incomplete"


@dataclass(frozen=True)
class TurnEvent:
    state: UserState
    transcript: str
    t_ms: int


class StateAdapter:
    """Stateful mapper from wire dicts to domain events. Pure; no I/O."""

    def __init__(self) -> None:
        self._prev_wire: str | None = None
        self._partial: str = ""

    def feed(self, wire: dict, t_ms: int) -> list[TurnEvent]:
        ws = wire.get("state")
        if ws == "blank":
            return []

        events: list[TurnEvent] = []

        if ws == "idle":
            if self._prev_wire == "nonidle":
                events.append(TurnEvent(UserState.INCOMPLETE, self._partial, t_ms))
            events.append(TurnEvent(UserState.IDLE, "", t_ms))
            self._partial = ""

        elif ws == "nonidle":
            self._partial = wire.get("asr_buffer", "")
            events.append(TurnEvent(UserState.NONIDLE, self._partial, t_ms))

        elif ws == "speak":
            text = wire.get("text", "")
            state = UserState.BACKCHANNEL if is_backchannel(text) else UserState.COMPLETE
            events.append(TurnEvent(state, text, t_ms))
            self._partial = ""

        self._prev_wire = ws
        return events
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest tests/test_states.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rtvoice/states.py tests/test_states.py
git commit -m "feat: domain states and SoulX wire adapter

Derives user_incomplete and user_backchannel, which the shipped API
does not serve directly."
```

---

### Task 5: Event log

**Files:**
- Create: `src/rtvoice/events.py`
- Test: `tests/test_events.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Event(kind: str, t_ms: int, data: dict)`; `EventLog(path)` with `.append(kind, **data) -> Event`, `.subscribe() -> AsyncIterator[Event]`, `.read(path) -> list[Event]` (classmethod).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_events.py
import json
import pytest
from rtvoice.events import Event, EventLog


def test_append_writes_jsonl(tmp_path):
    log = EventLog(tmp_path / "events.jsonl")
    log.append("turn_state", state="user_complete", transcript="hi")
    lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "turn_state"
    assert rec["data"]["state"] == "user_complete"


def test_timestamps_are_monotonic_offsets_from_session_start(tmp_path):
    log = EventLog(tmp_path / "e.jsonl")
    a = log.append("a")
    b = log.append("b")
    assert a.t_ms >= 0
    assert b.t_ms >= a.t_ms


def test_read_roundtrips(tmp_path):
    p = tmp_path / "e.jsonl"
    log = EventLog(p)
    log.append("x", v=1)
    log.append("y", v=2)
    events = EventLog.read(p)
    assert [e.kind for e in events] == ["x", "y"]
    assert events[1].data["v"] == 2


@pytest.mark.asyncio
async def test_subscribers_receive_appended_events(tmp_path):
    log = EventLog(tmp_path / "e.jsonl")
    received = []

    async def consume():
        async for ev in log.subscribe():
            received.append(ev.kind)
            if len(received) == 2:
                return

    import asyncio
    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    log.append("one")
    log.append("two")
    await asyncio.wait_for(task, timeout=1.0)
    assert received == ["one", "two"]
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_events.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.events'`

- [ ] **Step 3: Implement**

```python
# src/rtvoice/events.py
"""Append-only JSONL event log.

Everything downstream renders from this: the live UI subscribes, replay reads
a file, and tests assert on it. One artifact, three consumers.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Event:
    kind: str
    t_ms: int
    data: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"kind": self.kind, "t_ms": self.t_ms, "data": self.data})

    @staticmethod
    def from_dict(d: dict) -> "Event":
        return Event(kind=d["kind"], t_ms=d["t_ms"], data=d.get("data", {}))


class EventLog:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")
        self._t0 = time.monotonic()
        self._queues: list[asyncio.Queue] = []

    def append(self, kind: str, **data) -> Event:
        ev = Event(kind=kind, t_ms=int((time.monotonic() - self._t0) * 1000), data=data)
        self._fh.write(ev.to_json() + "\n")
        self._fh.flush()
        for q in self._queues:
            q.put_nowait(ev)
        return ev

    async def subscribe(self):
        q: asyncio.Queue = asyncio.Queue()
        self._queues.append(q)
        try:
            while True:
                yield await q.get()
        finally:
            self._queues.remove(q)

    def close(self) -> None:
        self._fh.close()

    @classmethod
    def read(cls, path: str | Path) -> list[Event]:
        return [
            Event.from_dict(json.loads(line))
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest tests/test_events.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rtvoice/events.py tests/test_events.py
git commit -m "feat: append-only JSONL event log with live subscription"
```

---

### Task 6: Task registry and cancellation tokens

**Files:**
- Create: `src/rtvoice/registry.py`, `src/rtvoice/cancellation.py`, `src/rtvoice/protocol.py`
- Test: `tests/test_registry.py`, `tests/test_cancellation.py`

**Interfaces:**
- Consumes: nothing
- Produces: `TaskStatus` enum; `Task` dataclass; `TaskRegistry` with `.apply(msg: ReasonerMessage)`, `.get(task_id)`, `.live_ids()`, `.fact_block()`; `CancellationToken` with `.cancel()`, `.cancelled`, `.check()`; `Cancelled` exception; `ReasonerMessage` / `OrchestratorMessage` pydantic models.

- [ ] **Step 1: Write the failing protocol and registry tests**

```python
# tests/test_registry.py
from rtvoice.protocol import ReasonerMessage
from rtvoice.registry import TaskRegistry, TaskStatus


def msg(kind, **kw):
    return ReasonerMessage(kind=kind, **kw)


def test_ack_creates_a_pending_task():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename contacts to Hans"))
    t = r.get("t1")
    assert t.status == TaskStatus.PENDING
    assert t.understood_as == "rename contacts to Hans"


def test_confirm_required_moves_to_awaiting_confirm_and_stores_verbatim():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("confirm_required", task_id="t1",
                verbatim_text="This will rename 47 contacts. Confirm?"))
    assert r.get("t1").status == TaskStatus.AWAITING_CONFIRM
    assert r.verbatim_span("t1") == "This will rename 47 contacts. Confirm?"


def test_done_stores_result_verbatim():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename"))
    r.apply(msg("done", task_id="t1", result="renamed 47 contacts"))
    assert r.get("t1").status == TaskStatus.DONE
    assert r.verbatim_span("t1") == "renamed 47 contacts"


def test_live_ids_excludes_terminal_tasks():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="a"))
    r.apply(msg("ack", task_id="t2", understood_as="b"))
    r.apply(msg("done", task_id="t2", result="ok"))
    assert r.live_ids() == ["t1"]


def test_fact_block_contains_only_current_state():
    r = TaskRegistry()
    r.apply(msg("ack", task_id="t1", understood_as="rename contacts"))
    r.apply(msg("progress", task_id="t1", status="scanning"))
    block = r.fact_block()
    assert "t1" in block
    assert "rename contacts" in block
    assert "scanning" in block


def test_noop_is_ignored():
    r = TaskRegistry()
    r.apply(msg("noop"))
    assert r.live_ids() == []
```

```python
# tests/test_cancellation.py
import pytest
from rtvoice.cancellation import Cancelled, CancellationToken


def test_token_starts_uncancelled():
    assert not CancellationToken().cancelled


def test_check_raises_after_cancel():
    tok = CancellationToken()
    tok.check()  # no raise
    tok.cancel()
    assert tok.cancelled
    with pytest.raises(Cancelled):
        tok.check()


def test_cancel_is_idempotent():
    tok = CancellationToken()
    tok.cancel()
    tok.cancel()
    assert tok.cancelled
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_registry.py tests/test_cancellation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.protocol'`

- [ ] **Step 3: Implement the protocol types**

```python
# src/rtvoice/protocol.py
"""Messages between orchestrator and reasoner. Async, unordered, both directions."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

ReasonerKind = Literal[
    "ack", "need_clarification", "progress", "done", "failed",
    "confirm_required", "noop",
]

OrchestratorKind = Literal[
    "utterance", "clarification_answer", "cancel", "nudge",
]


class ReasonerMessage(BaseModel):
    kind: ReasonerKind
    seq: int = 0
    task_id: Optional[str] = None
    understood_as: str = ""
    question: str = ""
    missing: str = ""
    options: list[str] = []
    status: str = ""
    result: str = ""
    reason: str = ""
    verbatim_text: str = ""


class OrchestratorMessage(BaseModel):
    kind: OrchestratorKind
    seq: int = 0
    task_id: Optional[str] = None
    text: str = ""
    raw_transcript: str = ""
    t_ms: int = 0
```

- [ ] **Step 4: Implement cancellation**

```python
# src/rtvoice/cancellation.py
"""Cancellation as a control-plane operation.

Firing a token does not require the reasoner's cooperation, any more than
SIGINT requires asking a process nicely. The executor polls between operations.
"""
from __future__ import annotations


class Cancelled(Exception):
    """Raised by CancellationToken.check() once the token has been fired."""


class CancellationToken:
    def __init__(self) -> None:
        self._cancelled = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def cancel(self) -> None:
        self._cancelled = True

    def check(self) -> None:
        if self._cancelled:
            raise Cancelled()
```

- [ ] **Step 5: Implement the registry**

```python
# src/rtvoice/registry.py
"""Single source of truth for task state, mirrored from reasoner messages.

The concierge never infers status; it reads fact_block(). Verbatim spans are
stored separately so the concierge's relays can be checked against them.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

from .protocol import ReasonerMessage


class TaskStatus(str, Enum):
    PENDING = "pending"
    AWAITING_CONFIRM = "awaiting_confirm"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL = {TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED}


@dataclass
class Task:
    task_id: str
    understood_as: str
    status: TaskStatus = TaskStatus.PENDING
    detail: str = ""
    created_ms: int = field(default_factory=lambda: int(time.monotonic() * 1000))
    updated_ms: int = field(default_factory=lambda: int(time.monotonic() * 1000))


class TaskRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._verbatim: dict[str, str] = {}

    def apply(self, msg: ReasonerMessage) -> None:
        if msg.kind == "noop" or msg.task_id is None:
            return

        tid = msg.task_id
        if tid not in self._tasks:
            self._tasks[tid] = Task(task_id=tid, understood_as=msg.understood_as)

        task = self._tasks[tid]
        task.updated_ms = int(time.monotonic() * 1000)

        if msg.kind == "ack":
            task.understood_as = msg.understood_as or task.understood_as
            task.status = TaskStatus.PENDING
        elif msg.kind == "confirm_required":
            task.status = TaskStatus.AWAITING_CONFIRM
            task.detail = msg.verbatim_text
            self._verbatim[tid] = msg.verbatim_text
        elif msg.kind == "need_clarification":
            task.status = TaskStatus.AWAITING_CONFIRM
            task.detail = msg.missing
        elif msg.kind == "progress":
            task.status = TaskStatus.RUNNING
            task.detail = msg.status
        elif msg.kind == "done":
            task.status = TaskStatus.DONE
            task.detail = msg.result
            self._verbatim[tid] = msg.result
        elif msg.kind == "failed":
            task.status = TaskStatus.FAILED
            task.detail = msg.reason
            self._verbatim[tid] = msg.reason

    def mark_cancelled(self, task_id: str) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].status = TaskStatus.CANCELLED

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def all(self) -> list[Task]:
        return list(self._tasks.values())

    def live_ids(self) -> list[str]:
        return [t.task_id for t in self._tasks.values() if t.status not in TERMINAL]

    def verbatim_span(self, task_id: str) -> str | None:
        return self._verbatim.get(task_id)

    def fact_block(self) -> str:
        """The complete, current, authoritative fact set given to the concierge.

        Deliberately contains no history — gaps in stale state are what
        invented task status is made of.
        """
        if not self._tasks:
            return "No tasks are in progress."
        lines = ["Current tasks (these are the ONLY task facts you may state):"]
        for t in self._tasks.values():
            line = f"- [{t.task_id}] {t.understood_as} — status: {t.status.value}"
            if t.detail:
                line += f' — exact wording to use: "{t.detail}"'
            lines.append(line)
        return "\n".join(lines)
```

- [ ] **Step 6: Run them and watch them pass**

Run: `uv run pytest tests/test_registry.py tests/test_cancellation.py -v`
Expected: PASS (9 passed)

- [ ] **Step 7: Commit**

```bash
git add src/rtvoice/protocol.py src/rtvoice/registry.py src/rtvoice/cancellation.py tests/test_registry.py tests/test_cancellation.py
git commit -m "feat: protocol types, task registry, cancellation tokens"
```

---

### Task 7: Turn policy

**Files:**
- Create: `src/rtvoice/policy.py`
- Test: `tests/test_policy.py`

**Interfaces:**
- Consumes: `TurnEvent`, `UserState` from Task 4
- Produces: `PolicyState(speaking: bool, pending_question: str | None)`; action dataclasses `Stop`, `SendUtterance`, `AskConcierge`; `decide(event, state) -> list[Action]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_policy.py
from rtvoice.policy import AskConcierge, PolicyState, SendUtterance, Stop, decide
from rtvoice.states import TurnEvent, UserState


def ev(state, transcript=""):
    return TurnEvent(state=state, transcript=transcript, t_ms=0)


def test_incomplete_does_nothing():
    """The whole reason SoulX-Duplug is here: hold through mid-sentence pauses."""
    assert decide(ev(UserState.INCOMPLETE, "find food and"), PolicyState()) == []


def test_backchannel_does_not_stop_speech():
    assert decide(ev(UserState.BACKCHANNEL, "mm hm"), PolicyState(speaking=True)) == []


def test_nonidle_while_speaking_stops_reflexively():
    actions = decide(ev(UserState.NONIDLE, "wait"), PolicyState(speaking=True))
    assert actions == [Stop()]


def test_nonidle_while_silent_does_nothing():
    assert decide(ev(UserState.NONIDLE, "hi"), PolicyState(speaking=False)) == []


def test_nonidle_during_pending_question_is_an_answer_not_a_bargein():
    state = PolicyState(speaking=True, pending_question="which contacts?")
    assert decide(ev(UserState.NONIDLE, "the work ones"), state) == []


def test_complete_dispatches_and_asks_concierge():
    actions = decide(ev(UserState.COMPLETE, "rename my contacts"), PolicyState())
    assert actions == [
        SendUtterance(text="rename my contacts"),
        AskConcierge(trigger="user_turn"),
    ]


def test_complete_while_speaking_stops_first():
    actions = decide(ev(UserState.COMPLETE, "stop"), PolicyState(speaking=True))
    assert actions[0] == Stop()
    assert SendUtterance(text="stop") in actions


def test_idle_does_nothing():
    assert decide(ev(UserState.IDLE), PolicyState()) == []
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_policy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.policy'`

- [ ] **Step 3: Implement**

```python
# src/rtvoice/policy.py
"""Turn policy: a pure function from (domain event, system state) to actions.

No models, no I/O. This is the spec's turn-policy table made executable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .states import TurnEvent, UserState


@dataclass(frozen=True)
class Stop:
    """Halt TTS immediately. Reflexive — no inference, harmless, reversible."""


@dataclass(frozen=True)
class SendUtterance:
    text: str


@dataclass(frozen=True)
class AskConcierge:
    trigger: str


Action = Stop | SendUtterance | AskConcierge


@dataclass
class PolicyState:
    speaking: bool = False
    pending_question: str | None = None


def decide(event: TurnEvent, state: PolicyState) -> list[Action]:
    if event.state in (UserState.IDLE, UserState.INCOMPLETE, UserState.BACKCHANNEL):
        # INCOMPLETE: the user paused mid-sentence. Wait. This is the point.
        # BACKCHANNEL: "mm hm" is not a turn grab. Keep speaking.
        return []

    if event.state == UserState.NONIDLE:
        # Barge-in is reflexive, unless we are holding a question, in which
        # case speech is the answer to it rather than an interruption.
        if state.speaking and state.pending_question is None:
            return [Stop()]
        return []

    if event.state == UserState.COMPLETE:
        actions: list[Action] = []
        if state.speaking:
            actions.append(Stop())
        # Every finalised utterance goes to the reasoner — it is the gatekeeper.
        # The concierge responds in parallel, never waiting for it.
        actions.append(SendUtterance(text=event.transcript))
        actions.append(AskConcierge(trigger="user_turn"))
        return actions

    return []
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest tests/test_policy.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rtvoice/policy.py tests/test_policy.py
git commit -m "feat: turn policy as a pure decision function"
```

---

## Phase 2 — Services

### Task 8: Journaled device state

**Files:**
- Create: `src/rtvoice/device.py`, `fixtures/device_state.json`
- Test: `tests/test_device.py`

**Interfaces:**
- Consumes: `CancellationToken`, `Cancelled` from Task 6
- Produces: `DeviceState(state_path, journal_path)` with `.query(table, where=None)`, `.update(table, set_fields, where=None, token=None) -> int`, `.commit()`, `.rollback()`, `.snapshot() -> dict`.

- [ ] **Step 1: Create the fixture**

```json
{
  "contacts": [
    {"id": 1, "first_name": "Sarah",  "last_name": "Chen",    "group": "work"},
    {"id": 2, "first_name": "Marcus", "last_name": "Webb",    "group": "work"},
    {"id": 3, "first_name": "Priya",  "last_name": "Nair",    "group": "family"},
    {"id": 4, "first_name": "Tom",    "last_name": "Okafor",  "group": "friends"}
  ],
  "messages": [
    {"id": 1, "to": "Sarah Chen", "body": "running late", "sent": "2026-07-26"}
  ],
  "calendar": [
    {"id": 1, "title": "standup", "when": "2026-07-28T09:00"}
  ],
  "places": []
}
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_device.py
import json
import pytest
from rtvoice.cancellation import Cancelled, CancellationToken
from rtvoice.device import DeviceState


@pytest.fixture
def dev(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [
            {"id": 1, "first_name": "Sarah", "group": "work"},
            {"id": 2, "first_name": "Marcus", "group": "work"},
            {"id": 3, "first_name": "Priya", "group": "family"},
        ]
    }))
    return DeviceState(state, tmp_path / "journal.jsonl")


def test_query_all(dev):
    assert len(dev.query("contacts")) == 3


def test_query_with_where(dev):
    assert len(dev.query("contacts", {"group": "work"})) == 2


def test_update_is_not_visible_until_commit(dev):
    n = dev.update("contacts", {"first_name": "Hans"})
    assert n == 3
    on_disk = json.loads(dev.state_path.read_text())
    assert on_disk["contacts"][0]["first_name"] == "Sarah"
    dev.commit()
    on_disk = json.loads(dev.state_path.read_text())
    assert all(c["first_name"] == "Hans" for c in on_disk["contacts"])


def test_rollback_discards_uncommitted_writes(dev):
    dev.update("contacts", {"first_name": "Hans"})
    dev.rollback()
    dev.commit()
    assert [c["first_name"] for c in dev.query("contacts")] == ["Sarah", "Marcus", "Priya"]


def test_update_respects_where(dev):
    dev.update("contacts", {"first_name": "Hans"}, {"group": "work"})
    dev.commit()
    names = {c["id"]: c["first_name"] for c in dev.query("contacts")}
    assert names == {1: "Hans", 2: "Hans", 3: "Priya"}


def test_cancellation_stops_mid_update_and_rolls_back(dev):
    token = CancellationToken()

    class CancelAfterOne(CancellationToken):
        def __init__(self):
            super().__init__()
            self.n = 0

        def check(self):
            self.n += 1
            if self.n > 1:
                raise Cancelled()

    with pytest.raises(Cancelled):
        dev.update("contacts", {"first_name": "Hans"}, token=CancelAfterOne())
    dev.rollback()
    dev.commit()
    assert [c["first_name"] for c in dev.query("contacts")] == ["Sarah", "Marcus", "Priya"]


def test_journal_records_every_write(dev):
    dev.update("contacts", {"first_name": "Hans"}, {"group": "work"})
    dev.commit()
    entries = [json.loads(l) for l in dev.journal_path.read_text().splitlines() if l.strip()]
    assert any(e["op"] == "update" and e["table"] == "contacts" for e in entries)
    assert any(e["op"] == "commit" for e in entries)
```

- [ ] **Step 3: Run them and watch them fail**

Run: `uv run pytest tests/test_device.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.device'`

- [ ] **Step 4: Implement**

```python
# src/rtvoice/device.py
"""Journaled fake device memory with two-phase commit.

Destructive writes stage into a working copy and are only persisted on
commit(), so an abort before commit leaves nothing behind. Every operation is
journaled, which is what makes "want me to undo the 12 I already changed?"
answerable.
"""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

from .cancellation import CancellationToken


class DeviceState:
    def __init__(self, state_path: str | Path, journal_path: str | Path) -> None:
        self.state_path = Path(state_path)
        self.journal_path = Path(journal_path)
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self._committed = json.loads(self.state_path.read_text(encoding="utf-8"))
        self._working = copy.deepcopy(self._committed)

    def _journal(self, op: str, **kw) -> None:
        rec = {"op": op, "ts": time.time(), **kw}
        with self.journal_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    @staticmethod
    def _matches(row: dict, where: dict | None) -> bool:
        return where is None or all(row.get(k) == v for k, v in where.items())

    def query(self, table: str, where: dict | None = None) -> list[dict]:
        return [r for r in self._working.get(table, []) if self._matches(r, where)]

    def update(
        self,
        table: str,
        set_fields: dict,
        where: dict | None = None,
        token: CancellationToken | None = None,
    ) -> int:
        """Stage an update. Raises Cancelled if the token fires mid-loop."""
        n = 0
        for row in self._working.get(table, []):
            if token is not None:
                token.check()
            if self._matches(row, where):
                row.update(set_fields)
                n += 1
        self._journal("update", table=table, set=set_fields, where=where, rows=n)
        return n

    def commit(self) -> None:
        self._committed = copy.deepcopy(self._working)
        self.state_path.write_text(
            json.dumps(self._committed, indent=2), encoding="utf-8"
        )
        self._journal("commit")

    def rollback(self) -> None:
        self._working = copy.deepcopy(self._committed)
        self._journal("rollback")

    def snapshot(self) -> dict:
        return copy.deepcopy(self._working)
```

- [ ] **Step 5: Run them and watch them pass**

Run: `uv run pytest tests/test_device.py -v`
Expected: PASS (7 passed)

- [ ] **Step 6: Commit**

```bash
git add src/rtvoice/device.py fixtures/device_state.json tests/test_device.py
git commit -m "feat: journaled device state with two-phase commit and rollback"
```

---

### Task 9: Concierge speech acts and validation

**Files:**
- Create: `src/rtvoice/concierge.py`
- Test: `tests/test_concierge.py`

**Interfaces:**
- Consumes: `TaskRegistry` from Task 6
- Produces: `SpeechAct` pydantic model; `validate_act(act, registry) -> tuple[bool, str]`; `Concierge(base_url, model)` with `async .respond(fact_block, history, trigger) -> SpeechAct`; `SPEECH_ACT_SCHEMA` dict for guided decoding.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_concierge.py
from rtvoice.concierge import SpeechAct, validate_act
from rtvoice.protocol import ReasonerMessage
from rtvoice.registry import TaskRegistry


def registry_with_done_task():
    r = TaskRegistry()
    r.apply(ReasonerMessage(kind="ack", task_id="t1", understood_as="rename contacts"))
    r.apply(ReasonerMessage(kind="done", task_id="t1", result="renamed 47 contacts"))
    return r


def test_acknowledge_needs_no_citation():
    ok, _ = validate_act(SpeechAct(act="acknowledge", text="on it"), TaskRegistry())
    assert ok


def test_chat_needs_no_citation():
    ok, _ = validate_act(SpeechAct(act="chat", text="sure, what else?"), TaskRegistry())
    assert ok


def test_relay_without_citation_is_rejected():
    ok, err = validate_act(
        SpeechAct(act="relay", text="I renamed all your contacts"), registry_with_done_task()
    )
    assert not ok
    assert "cites" in err


def test_relay_citing_unknown_task_is_rejected():
    ok, err = validate_act(
        SpeechAct(act="relay", text="done", cites="nope"), registry_with_done_task()
    )
    assert not ok
    assert "unknown" in err


def test_relay_missing_the_verbatim_span_is_rejected():
    """The whole point: the concierge may not restate a fact in its own words."""
    ok, err = validate_act(
        SpeechAct(act="relay", text="all done with your contacts!", cites="t1"),
        registry_with_done_task(),
    )
    assert not ok
    assert "verbatim" in err


def test_relay_containing_the_verbatim_span_is_accepted():
    ok, _ = validate_act(
        SpeechAct(act="relay", text="Okay — renamed 47 contacts. Anything else?", cites="t1"),
        registry_with_done_task(),
    )
    assert ok


def test_abort_requires_a_live_task():
    r = TaskRegistry()
    r.apply(ReasonerMessage(kind="ack", task_id="t1", understood_as="rename"))
    ok, _ = validate_act(SpeechAct(act="abort", cites="t1"), r)
    assert ok
    ok, err = validate_act(SpeechAct(act="abort", cites="ghost"), r)
    assert not ok
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_concierge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.concierge'`

- [ ] **Step 3: Implement**

```python
# src/rtvoice/concierge.py
"""The concierge: owns the conversation, never asserts task facts.

Three structural measures replace any inspection of generated prose:
  1. Narrow context  - it sees only the current fact block, no history of state
  2. Split authorship - the reasoner writes facts; the concierge frames them
  3. Typed speech acts - validation is a schema check on typed fields
"""
from __future__ import annotations

import json
from typing import Literal, Optional

import httpx
from pydantic import BaseModel

from .registry import TaskRegistry

SYSTEM_PROMPT = """You are the voice of an assistant that controls a phone.

A separate reasoning system does all the actual work and owns all facts about
the phone. You own the conversation.

RULES:
- You may ONLY state task facts that appear in the CURRENT TASKS block below.
- When reporting a result, use act="relay", cite the task id, and include the
  exact wording given for that task verbatim. You may add conversational
  framing around it, but never reword the fact itself.
- If you do not know something, say so. Never guess at task status.
- Keep replies short and natural - they will be spoken aloud.

Reply with a single JSON object and nothing else:
  {"act": "acknowledge", "text": "..."}   brief filler while work happens
  {"act": "relay", "cites": "<task_id>", "text": "..."}  report a result
  {"act": "ask", "text": "..."}           ask the user something
  {"act": "abort", "cites": "<task_id>"}  user clearly wants a task stopped now
  {"act": "chat", "text": "..."}          ordinary conversation
"""

SPEECH_ACT_SCHEMA = {
    "type": "object",
    "properties": {
        "act": {"type": "string", "enum": ["relay", "ask", "acknowledge", "abort", "chat"]},
        "text": {"type": "string"},
        "cites": {"type": ["string", "null"]},
    },
    "required": ["act"],
}


class SpeechAct(BaseModel):
    act: Literal["relay", "ask", "acknowledge", "abort", "chat"]
    text: str = ""
    cites: Optional[str] = None


def validate_act(act: SpeechAct, registry: TaskRegistry) -> tuple[bool, str]:
    """Schema-level check on typed fields. Never inspects natural language."""
    if act.act == "relay":
        if not act.cites:
            return False, "relay requires cites"
        if registry.get(act.cites) is None:
            return False, f"unknown task id: {act.cites}"
        span = registry.verbatim_span(act.cites)
        if span and span not in act.text:
            return False, f"relay must contain the verbatim span: {span!r}"
    if act.act == "abort":
        if not act.cites or act.cites not in registry.live_ids():
            return False, "abort requires a live task id"
    return True, ""


class Concierge:
    def __init__(
        self,
        base_url: str = "http://localhost:8001/v1",
        model: str = "Qwen/Qwen3-4B",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)
        self.violations = 0

    async def _complete(self, messages: list[dict]) -> SpeechAct:
        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "max_tokens": 200,
                "temperature": 0.6,
                "guided_json": SPEECH_ACT_SCHEMA,
            },
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return SpeechAct(**json.loads(content))

    async def respond(
        self, registry: TaskRegistry, history: list[dict], trigger: str
    ) -> SpeechAct:
        """Generate one speech act. Re-prompts once on schema violation, then
        falls back to acknowledge. Never rewrites the model's output."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": f"CURRENT TASKS:\n{registry.fact_block()}"},
            *history,
            {"role": "user", "content": f"[trigger: {trigger}]"},
        ]

        act = await self._complete(messages)
        ok, err = validate_act(act, registry)
        if ok:
            return act

        self.violations += 1
        messages.append({"role": "system", "content": f"Rejected: {err}. Try again."})
        act = await self._complete(messages)
        ok, _ = validate_act(act, registry)
        if ok:
            return act

        self.violations += 1
        return SpeechAct(act="acknowledge", text="Let me check on that.")

    async def aclose(self) -> None:
        await self._client.aclose()
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest tests/test_concierge.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rtvoice/concierge.py tests/test_concierge.py
git commit -m "feat: concierge speech acts with schema validation

Citation discipline enforced structurally; violations counted as a metric."
```

---

### Task 10: Reasoner stub

**Files:**
- Create: `src/rtvoice/reasoner_stub.py`
- Test: `tests/test_reasoner_stub.py`

**Interfaces:**
- Consumes: `DeviceState` (Task 8), `ReasonerMessage`/`OrchestratorMessage` (Task 6), `CancellationToken` (Task 6)
- Produces: FastAPI app on port 8002 with `POST /message` accepting `OrchestratorMessage` and returning `list[ReasonerMessage]`; `ReasonerStub.handle(msg) -> list[ReasonerMessage]`; a `latency_ms` setting for the artificial-slowness control.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_reasoner_stub.py
import json
import pytest
from rtvoice.device import DeviceState
from rtvoice.protocol import OrchestratorMessage
from rtvoice.reasoner_stub import ReasonerStub


@pytest.fixture
def stub(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [{"id": 1, "first_name": "Sarah", "group": "work"},
                     {"id": 2, "first_name": "Marcus", "group": "work"}]
    }))
    return ReasonerStub(DeviceState(state, tmp_path / "j.jsonl"), latency_ms=0)


@pytest.mark.asyncio
async def test_chat_returns_noop(stub):
    out = await stub.handle(OrchestratorMessage(kind="utterance", text="hello there"))
    assert [m.kind for m in out] == ["noop"]


@pytest.mark.asyncio
async def test_destructive_command_requires_confirmation_before_writing(stub):
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    kinds = [m.kind for m in out]
    assert "ack" in kinds
    assert "confirm_required" in kinds
    # nothing written yet
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_confirmation_commits_the_write(stub):
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    done = await stub.handle(
        OrchestratorMessage(kind="clarification_answer", task_id=tid, text="yes do it")
    )
    assert any(m.kind == "done" for m in done)
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Hans", "Hans"]


@pytest.mark.asyncio
async def test_cancel_before_confirm_leaves_state_untouched(stub):
    out = await stub.handle(
        OrchestratorMessage(kind="utterance", text="set all my contacts to Hans")
    )
    tid = next(m.task_id for m in out if m.kind == "confirm_required")
    await stub.handle(OrchestratorMessage(kind="cancel", task_id=tid))
    assert [c["first_name"] for c in stub.device.query("contacts")] == ["Sarah", "Marcus"]


@pytest.mark.asyncio
async def test_compound_command_creates_multiple_tasks(stub):
    out = await stub.handle(OrchestratorMessage(
        kind="utterance",
        text="find chinese restaurants in soho and set all my contacts to Hans",
    ))
    acks = [m for m in out if m.kind == "ack"]
    assert len(acks) == 2
    assert len({m.task_id for m in acks}) == 2
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_reasoner_stub.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.reasoner_stub'`

- [ ] **Step 3: Implement**

```python
# src/rtvoice/reasoner_stub.py
"""Stand-in for the memory-enabled reasoner.

Rule-based intent extraction keeps the tests deterministic; an LLM roleplay
path is available behind `use_llm` for demos. Either way it mutates a real
journaled DeviceState, so writes are verifiable by diffing a file.

`latency_ms` exists so long-wait behaviour can be exercised on demand - the
stub's latency is chosen rather than discovered, and that is the point.
"""
from __future__ import annotations

import asyncio
import itertools
import re

from fastapi import FastAPI

from .cancellation import CancellationToken
from .device import DeviceState
from .protocol import OrchestratorMessage, ReasonerMessage

_ids = itertools.count(1)

# Splitting on " and " is enough for the compound commands we demo; the real
# reasoner does this properly.
_SPLIT = re.compile(r"\s+and\s+(?=(?:find|get|set|change|rename|book|text|call|delete|add|search)\b)")
_RENAME = re.compile(r"(?:set|change|rename)\s+(?:all\s+)?(?:my\s+)?contacts?.*?to\s+(\w+)", re.I)
_SEARCH = re.compile(r"(?:find|search for)\s+(.+)", re.I)


class ReasonerStub:
    def __init__(self, device: DeviceState, latency_ms: int = 0) -> None:
        self.device = device
        self.latency_ms = latency_ms
        self._pending: dict[str, dict] = {}
        self._tokens: dict[str, CancellationToken] = {}

    async def handle(self, msg: OrchestratorMessage) -> list[ReasonerMessage]:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)

        if msg.kind == "cancel":
            return self._cancel(msg.task_id)
        if msg.kind == "clarification_answer":
            return await self._confirm(msg.task_id, msg.text)
        if msg.kind == "nudge":
            task = self._pending.get(msg.task_id or "")
            return [ReasonerMessage(kind="progress", task_id=msg.task_id,
                                    status="still working" if task else "no such task")]
        if msg.kind != "utterance":
            return [ReasonerMessage(kind="noop")]

        out: list[ReasonerMessage] = []
        for clause in _SPLIT.split(msg.text):
            out.extend(self._plan(clause.strip()))
        return out or [ReasonerMessage(kind="noop")]

    def _plan(self, clause: str) -> list[ReasonerMessage]:
        rename = _RENAME.search(clause)
        if rename:
            name = rename.group(1)
            tid = f"t{next(_ids)}"
            n = len(self.device.query("contacts"))
            self._pending[tid] = {"op": "rename", "name": name}
            return [
                ReasonerMessage(kind="ack", task_id=tid,
                                understood_as=f"rename all contacts to {name}"),
                ReasonerMessage(
                    kind="confirm_required", task_id=tid,
                    verbatim_text=f"This will rename {n} contacts to {name}. Confirm?",
                ),
            ]

        search = _SEARCH.search(clause)
        if search:
            tid = f"t{next(_ids)}"
            query = search.group(1)
            hits = 12  # the stub does not really search
            return [
                ReasonerMessage(kind="ack", task_id=tid, understood_as=f"search: {query}"),
                ReasonerMessage(kind="done", task_id=tid, result=f"found {hits} results for {query}"),
            ]

        return []

    async def _confirm(self, task_id: str | None, answer: str) -> list[ReasonerMessage]:
        plan = self._pending.pop(task_id or "", None)
        if plan is None:
            return [ReasonerMessage(kind="noop")]
        if not re.search(r"\b(yes|yeah|yep|do it|go ahead|confirm|ok|okay)\b", answer, re.I):
            self.device.rollback()
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="cancelled by user")]

        token = CancellationToken()
        self._tokens[task_id] = token
        try:
            n = self.device.update("contacts", {"first_name": plan["name"]}, token=token)
            self.device.commit()
        except Exception:
            self.device.rollback()
            return [ReasonerMessage(kind="failed", task_id=task_id, reason="stopped partway")]
        return [ReasonerMessage(kind="done", task_id=task_id,
                                result=f"renamed {n} contacts to {plan['name']}")]

    def _cancel(self, task_id: str | None) -> list[ReasonerMessage]:
        if task_id in self._tokens:
            self._tokens[task_id].cancel()
        self._pending.pop(task_id or "", None)
        self.device.rollback()
        return [ReasonerMessage(kind="failed", task_id=task_id, reason="cancelled")]


def create_app(device: DeviceState, latency_ms: int = 0) -> FastAPI:
    app = FastAPI()
    stub = ReasonerStub(device, latency_ms=latency_ms)
    app.state.stub = stub

    @app.post("/message")
    async def message(msg: OrchestratorMessage) -> list[ReasonerMessage]:
        return await app.state.stub.handle(msg)

    @app.post("/latency/{ms}")
    async def set_latency(ms: int) -> dict:
        app.state.stub.latency_ms = ms
        return {"latency_ms": ms}

    return app
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest tests/test_reasoner_stub.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rtvoice/reasoner_stub.py tests/test_reasoner_stub.py
git commit -m "feat: reasoner stub with two-phase commit and latency control"
```

---

### Task 11: TTS and dual-channel session recorder

**Files:**
- Create: `src/rtvoice/tts.py`, `src/rtvoice/recorder.py`
- Test: `tests/test_recorder.py`

**Interfaces:**
- Consumes: nothing
- Produces: `KokoroTTS(voice)` with `async .stream(text) -> AsyncIterator[np.ndarray]` yielding float32 16 kHz chunks and `.stop()`; `SessionRecorder(session_dir)` with `.write_user(chunk)`, `.write_model(chunk)`, `.close()` producing `user.wav`, `model.wav`, `mix.wav`.

- [ ] **Step 1: Write the failing recorder tests**

```python
# tests/test_recorder.py
import numpy as np
import soundfile as sf
from rtvoice.recorder import SessionRecorder


def test_writes_three_wavs(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.write_user(np.zeros(1600, dtype=np.float32))
    rec.write_model(np.ones(1600, dtype=np.float32) * 0.1)
    rec.close()
    for name in ("user.wav", "model.wav", "mix.wav"):
        assert (tmp_path / name).exists(), name


def test_channels_are_kept_separate(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.write_user(np.ones(1600, dtype=np.float32) * 0.5)
    rec.write_model(np.zeros(1600, dtype=np.float32))
    rec.close()
    user, sr = sf.read(tmp_path / "user.wav", dtype="float32")
    assert sr == 16000
    assert np.allclose(user, 0.5, atol=1e-3)


def test_mix_is_stereo_user_left_model_right(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.write_user(np.ones(1600, dtype=np.float32) * 0.5)
    rec.write_model(np.ones(1600, dtype=np.float32) * 0.25)
    rec.close()
    mix, sr = sf.read(tmp_path / "mix.wav", dtype="float32")
    assert mix.ndim == 2 and mix.shape[1] == 2
    assert np.allclose(mix[:, 0], 0.5, atol=1e-3)
    assert np.allclose(mix[:, 1], 0.25, atol=1e-3)


def test_unequal_lengths_are_zero_padded(tmp_path):
    rec = SessionRecorder(tmp_path)
    rec.write_user(np.ones(3200, dtype=np.float32) * 0.5)
    rec.write_model(np.ones(1600, dtype=np.float32) * 0.25)
    rec.close()
    mix, _ = sf.read(tmp_path / "mix.wav", dtype="float32")
    assert mix.shape[0] == 3200
    assert np.allclose(mix[1600:, 1], 0.0, atol=1e-6)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.recorder'`

- [ ] **Step 3: Implement the recorder**

```python
# src/rtvoice/recorder.py
"""Dual-channel session recorder.

Channels are kept separate as well as mixed so user and model audio can be
analysed independently - which is how acoustic echo gets diagnosed rather
than guessed at. Recording is always on; sessions are curated afterwards.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16000


class SessionRecorder:
    def __init__(self, session_dir: str | Path) -> None:
        self.dir = Path(session_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._user: list[np.ndarray] = []
        self._model: list[np.ndarray] = []

    def write_user(self, chunk: np.ndarray) -> None:
        self._user.append(np.asarray(chunk, dtype=np.float32).copy())

    def write_model(self, chunk: np.ndarray) -> None:
        self._model.append(np.asarray(chunk, dtype=np.float32).copy())

    def close(self) -> None:
        user = np.concatenate(self._user) if self._user else np.zeros(0, dtype=np.float32)
        model = np.concatenate(self._model) if self._model else np.zeros(0, dtype=np.float32)

        sf.write(self.dir / "user.wav", user, SAMPLE_RATE)
        sf.write(self.dir / "model.wav", model, SAMPLE_RATE)

        n = max(len(user), len(model))
        mix = np.zeros((n, 2), dtype=np.float32)
        mix[: len(user), 0] = user
        mix[: len(model), 1] = model
        sf.write(self.dir / "mix.wav", mix, SAMPLE_RATE)
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Implement the TTS wrapper**

```python
# src/rtvoice/tts.py
"""Kokoro TTS, streamed and interruptible.

Kokoro emits 24 kHz; everything else in this system is 16 kHz, so we resample
on the way out. stop() sets a flag checked between chunks, so barge-in halts
speech within one chunk rather than at the end of the utterance.
"""
from __future__ import annotations

from typing import AsyncIterator

import numpy as np

SAMPLE_RATE = 16000
KOKORO_RATE = 24000


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst:
        return audio.astype(np.float32)
    n = int(round(len(audio) * dst / src))
    idx = np.linspace(0, len(audio) - 1, n)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


class KokoroTTS:
    def __init__(self, voice: str = "af_heart", lang_code: str = "a") -> None:
        from kokoro import KPipeline  # imported lazily; needs GPU deps

        self._pipeline = KPipeline(lang_code=lang_code)
        self.voice = voice
        self._stopped = False

    def stop(self) -> None:
        self._stopped = True

    async def stream(self, text: str) -> AsyncIterator[np.ndarray]:
        self._stopped = False
        for _, _, audio in self._pipeline(text, voice=self.voice):
            if self._stopped:
                return
            yield _resample(np.asarray(audio, dtype=np.float32), KOKORO_RATE, SAMPLE_RATE)
```

- [ ] **Step 6: Verify TTS on the 3090**

```bash
./tools/sync.sh
ssh gpu3090 'cd ~/rtvoice && uv run python -c "
import asyncio, numpy as np, soundfile as sf
from rtvoice.tts import KokoroTTS
async def main():
    tts = KokoroTTS()
    chunks = [c async for c in tts.stream(\"Renamed forty seven contacts to Hans.\")]
    sf.write(\"/tmp/tts_check.wav\", np.concatenate(chunks), 16000)
    print(\"wrote /tmp/tts_check.wav\", sum(len(c) for c in chunks)/16000, \"sec\")
asyncio.run(main())
"'
```

Expected: prints a duration around 2–3 seconds. Copy the file back and listen to it.

- [ ] **Step 7: Commit**

```bash
git add src/rtvoice/tts.py src/rtvoice/recorder.py tests/test_recorder.py
git commit -m "feat: interruptible Kokoro TTS and dual-channel session recorder"
```

---

### Task 12: Orchestrator wiring and end-to-end verification

**Files:**
- Create: `src/rtvoice/orchestrator.py`, `src/rtvoice/voice_service.py`, `tools/run_all.sh`
- Test: `tests/test_end_to_end.py`

**Interfaces:**
- Consumes: everything above
- Produces: `Orchestrator` with `async .on_turn_event(ev)`, `async .on_reasoner_messages(msgs)`, `.policy_state`; FastAPI app on 8000 exposing `POST /inject` (manual text), `GET /events` (WS), `GET /state`.

- [ ] **Step 1: Write the failing end-to-end tests**

First create the shared test doubles — Task 13 reuses them, so they live in
their own module rather than being duplicated:

```python
# tests/fakes.py
"""Shared test doubles. Imported as `from fakes import ...` — pytest's default
prepend import mode puts tests/ on sys.path, and there is no tests/__init__.py.
"""
from rtvoice.concierge import SpeechAct


class FakeConcierge:
    """Records what it was asked; returns a fixed acknowledge."""

    def __init__(self):
        self.calls = []
        self.violations = 0

    async def respond(self, registry, history, trigger):
        self.calls.append((registry.fact_block(), trigger))
        return SpeechAct(act="acknowledge", text="on it")


class FakeVoice:
    def __init__(self):
        self.spoken = []
        self.stops = 0

    async def speak(self, text, utterance_id):
        self.spoken.append(text)

    async def stop(self):
        self.stops += 1
```

```python
# tests/test_end_to_end.py
import json

import pytest
from fakes import FakeConcierge, FakeVoice

from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.registry import TaskStatus
from rtvoice.states import TurnEvent, UserState


@pytest.fixture
def orch(tmp_path):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({
        "contacts": [{"id": 1, "first_name": "Sarah", "group": "work"},
                     {"id": 2, "first_name": "Marcus", "group": "work"}]
    }))
    device = DeviceState(state, tmp_path / "journal.jsonl")
    return Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(),
        voice=FakeVoice(),
        log=EventLog(tmp_path / "events.jsonl"),
        device=device,
    )


@pytest.mark.asyncio
async def test_incomplete_does_not_dispatch(orch):
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "find food and", 0))
    assert orch.registry.live_ids() == []


@pytest.mark.asyncio
async def test_compound_command_creates_two_tasks(orch):
    await orch.on_turn_event(TurnEvent(
        UserState.COMPLETE,
        "find chinese restaurants in soho and set all my contacts to Hans", 0))
    assert len(orch.registry.all()) == 2


@pytest.mark.asyncio
async def test_confirmed_destructive_command_mutates_device_state(orch):
    await orch.on_turn_event(
        TurnEvent(UserState.COMPLETE, "set all my contacts to Hans", 0))
    tid = next(t.task_id for t in orch.registry.all()
               if t.status == TaskStatus.AWAITING_CONFIRM)
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "yes do it", 1000))
    assert orch.registry.get(tid).status == TaskStatus.DONE
    assert [c["first_name"] for c in orch.device.query("contacts")] == ["Hans", "Hans"]


@pytest.mark.asyncio
async def test_barge_in_stops_speech(orch):
    orch.policy_state.speaking = True
    await orch.on_turn_event(TurnEvent(UserState.NONIDLE, "wait", 0))
    assert orch.voice.stops == 1


@pytest.mark.asyncio
async def test_backchannel_does_not_stop_speech(orch):
    orch.policy_state.speaking = True
    await orch.on_turn_event(TurnEvent(UserState.BACKCHANNEL, "mm hm", 0))
    assert orch.voice.stops == 0


@pytest.mark.asyncio
async def test_every_turn_is_logged(orch):
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "hello", 0))
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "turn_event" in kinds
    assert "concierge_act" in kinds
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_end_to_end.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.orchestrator'`

- [ ] **Step 3: Implement the orchestrator**

```python
# src/rtvoice/orchestrator.py
"""Wires turn events, the reasoner and the concierge together.

The concierge and reasoner run in PARALLEL, not as a pipeline: the concierge
answers every turn at conversational latency while the reasoner acts only on
what is actionable. Neither waits for the other.
"""
from __future__ import annotations

import asyncio
import uuid

from .cancellation import CancellationToken
from .events import EventLog
from .policy import AskConcierge, PolicyState, SendUtterance, Stop, decide
from .protocol import OrchestratorMessage, ReasonerMessage
from .registry import TaskRegistry, TaskStatus
from .states import TurnEvent

SPEAK_ON = {"done", "failed", "need_clarification", "confirm_required"}


class Orchestrator:
    def __init__(self, reasoner, concierge, voice, log: EventLog, device) -> None:
        self.reasoner = reasoner
        self.concierge = concierge
        self.voice = voice
        self.log = log
        self.device = device
        self.registry = TaskRegistry()
        self.policy_state = PolicyState()
        self.history: list[dict] = []
        self.tokens: dict[str, CancellationToken] = {}
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    async def on_turn_event(self, ev: TurnEvent) -> None:
        self.log.append("turn_event", state=ev.state.value,
                        transcript=ev.transcript, t_ms=ev.t_ms)

        for action in decide(ev, self.policy_state):
            if isinstance(action, Stop):
                await self.voice.stop()
                self.policy_state.speaking = False
                self.log.append("tts_stopped")

            elif isinstance(action, SendUtterance):
                self.history.append({"role": "user", "content": action.text})
                await self._dispatch(action.text)

            elif isinstance(action, AskConcierge):
                await self._ask_concierge(action.trigger)

    async def _dispatch(self, text: str) -> None:
        awaiting = [t for t in self.registry.all()
                    if t.status == TaskStatus.AWAITING_CONFIRM]
        if awaiting:
            msg = OrchestratorMessage(kind="clarification_answer",
                                      task_id=awaiting[0].task_id,
                                      text=text, seq=self._next_seq())
        else:
            msg = OrchestratorMessage(kind="utterance", text=text,
                                      raw_transcript=text, seq=self._next_seq())

        self.log.append("to_reasoner", **msg.model_dump())
        await self.on_reasoner_messages(await self.reasoner.handle(msg))

    async def on_reasoner_messages(self, msgs: list[ReasonerMessage]) -> None:
        should_speak = False
        for m in msgs:
            self.log.append("from_reasoner", **m.model_dump())
            self.registry.apply(m)
            if m.kind in SPEAK_ON:
                should_speak = True
            if m.kind == "confirm_required":
                self.policy_state.pending_question = m.verbatim_text
            if m.kind in ("done", "failed"):
                self.policy_state.pending_question = None
        if should_speak:
            await self._ask_concierge("reasoner_update")

    async def _ask_concierge(self, trigger: str) -> None:
        act = await self.concierge.respond(self.registry, self.history, trigger)
        self.log.append("concierge_act", act=act.act, cites=act.cites, text=act.text,
                        violations=getattr(self.concierge, "violations", 0))

        if act.act == "abort" and act.cites:
            await self._abort(act.cites)
            return

        if act.text:
            self.history.append({"role": "assistant", "content": act.text})
            self.policy_state.speaking = True
            await self.voice.speak(act.text, uuid.uuid4().hex)
            self.policy_state.speaking = False

    async def _abort(self, task_id: str) -> None:
        """Fire the token FIRST; notifying the reasoner is secondary and
        correctness never depends on it arriving."""
        token = self.tokens.get(task_id)
        if token is not None:
            token.cancel()
        self.registry.mark_cancelled(task_id)
        self.log.append("aborted", task_id=task_id)
        msg = OrchestratorMessage(kind="cancel", task_id=task_id, seq=self._next_seq())
        asyncio.create_task(self.reasoner.handle(msg))
```

- [ ] **Step 4: Add the FastAPI surface**

Append to `src/rtvoice/orchestrator.py`:

```python
def create_app(orch: Orchestrator) -> "FastAPI":
    """HTTP/WS surface. /inject is the development affordance that lets the
    whole loop be exercised without a microphone."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from pydantic import BaseModel

    from .states import TurnEvent, UserState

    class Inject(BaseModel):
        text: str

    app = FastAPI()
    app.state.orch = orch

    @app.post("/inject")
    async def inject(body: Inject) -> dict:
        o: Orchestrator = app.state.orch
        await o.on_turn_event(TurnEvent(UserState.COMPLETE, body.text, 0))
        return {
            "tasks": [
                {"task_id": t.task_id, "understood_as": t.understood_as,
                 "status": t.status.value, "detail": t.detail}
                for t in o.registry.all()
            ]
        }

    @app.get("/state")
    async def state() -> dict:
        o: Orchestrator = app.state.orch
        return {
            "tasks": [
                {"task_id": t.task_id, "understood_as": t.understood_as,
                 "status": t.status.value, "detail": t.detail}
                for t in o.registry.all()
            ],
            "device": o.device.snapshot(),
            "speaking": o.policy_state.speaking,
            "pending_question": o.policy_state.pending_question,
        }

    @app.websocket("/events")
    async def events(ws: WebSocket) -> None:
        await ws.accept()
        o: Orchestrator = app.state.orch
        try:
            async for ev in o.log.subscribe():
                await ws.send_text(ev.to_json())
        except WebSocketDisconnect:
            pass

    return app
```

Then create the module-level `app` that `run_all.sh` launches, at the end of the same file:

```python
def _default_app():
    import os
    from pathlib import Path

    from .device import DeviceState
    from .concierge import Concierge
    from .reasoner_stub import ReasonerStub
    from .voice_service import VoiceService

    session = Path("sessions") / os.environ.get("SESSION_ID", "dev")
    device = DeviceState("fixtures/device_state.json", session / "device_journal.jsonl")
    log = EventLog(session / "events.jsonl")
    voice = VoiceService(session)
    concierge = Concierge(base_url=os.environ.get("CONCIERGE_URL", "http://localhost:8001/v1"))
    orch = Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=int(os.environ.get("REASONER_LATENCY_MS", "0"))),
        concierge=concierge, voice=voice, log=log, device=device,
    )
    return create_app(orch)


app = _default_app()
```

- [ ] **Step 5: Run them and watch them pass**

Run: `uv run pytest tests/test_end_to_end.py -v`
Expected: PASS (6 passed)

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -v`
Expected: PASS — all tests from Tasks 1–12 green.

- [ ] **Step 7: Implement voice-service and the launcher**

```python
# src/rtvoice/voice_service.py
"""Owns all audio: SoulX-Duplug in, Kokoro out, both channels recorded.

This is the relocatable boundary. If the tunnel measures badly (Task 1), this
service moves to the Mac and only text crosses the wire.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .recorder import SessionRecorder
from .soulx_client import CHUNK_SAMPLES, SoulXClient
from .states import StateAdapter, TurnEvent


class VoiceService:
    def __init__(self, session_dir: str | Path, soulx_url: str = "ws://localhost:8000/turn"):
        self.client = SoulXClient(soulx_url)
        self.adapter = StateAdapter()
        self.recorder = SessionRecorder(session_dir)
        self.tts = None  # set by the caller; lazily imported to avoid GPU deps in tests
        self._t_ms = 0

    async def feed_audio(self, chunk: np.ndarray) -> list[TurnEvent]:
        self.recorder.write_user(chunk)
        wire = await self.client.feed(chunk)
        events = self.adapter.feed(wire, self._t_ms)
        self._t_ms += int(CHUNK_SAMPLES / 16000 * 1000)
        return events

    async def speak(self, text: str, utterance_id: str) -> None:
        async for chunk in self.tts.stream(text):
            self.recorder.write_model(chunk)

    async def stop(self) -> None:
        if self.tts is not None:
            self.tts.stop()

    def close(self) -> None:
        self.recorder.close()
```

```bash
# tools/run_all.sh — run ON the 3090
#!/usr/bin/env bash
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
```

- [ ] **Step 8: Verify end to end against the real stack**

```bash
./tools/sync.sh
ssh gpu3090 'cd ~/rtvoice && bash tools/run_all.sh' &
# with the tunnel up:
curl -s -X POST localhost:8003/inject \
     -H 'content-type: application/json' \
     -d '{"text":"set all my contacts to Hans"}' | python3 -m json.tool
curl -s -X POST localhost:8003/inject \
     -H 'content-type: application/json' -d '{"text":"yes do it"}'
ssh gpu3090 'cat ~/rtvoice/fixtures/device_state.json' | grep first_name
```

Expected: every contact's `first_name` is `Hans`, and `sessions/<ts>/events.jsonl` contains the full turn/reasoner/concierge trace.

- [ ] **Step 9: Commit**

```bash
git add src/rtvoice/orchestrator.py src/rtvoice/voice_service.py tools/run_all.sh tests/test_end_to_end.py
git commit -m "feat: orchestrator wiring and end-to-end verification

Concierge and reasoner run in parallel; abort fires the cancellation token
before notifying the reasoner."
```

---

### Task 13: Resilience — timeouts and the concierge bypass flag

Closes three spec requirements: the "turn-taking never fires" and "reasoner never replies"
rows of the error-handling table, and the concierge removability flag.

**Files:**
- Modify: `src/rtvoice/orchestrator.py`
- Test: `tests/test_resilience.py`

**Interfaces:**
- Consumes: `Orchestrator` from Task 12
- Produces: `Orchestrator(..., silence_timeout_ms=2000, reasoner_timeout_s=20.0, use_concierge=True)`; `async .on_tick(now_ms)` which forces a `user_complete` after a silence timeout.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_resilience.py
import json
import pytest
from rtvoice.device import DeviceState
from rtvoice.events import EventLog
from rtvoice.orchestrator import Orchestrator
from rtvoice.protocol import ReasonerMessage
from rtvoice.reasoner_stub import ReasonerStub
from rtvoice.states import TurnEvent, UserState
from fakes import FakeConcierge, FakeVoice


def make(tmp_path, **kw):
    state = tmp_path / "device_state.json"
    state.write_text(json.dumps({"contacts": [{"id": 1, "first_name": "Sarah"}]}))
    device = DeviceState(state, tmp_path / "j.jsonl")
    return Orchestrator(
        reasoner=ReasonerStub(device, latency_ms=0),
        concierge=FakeConcierge(), voice=FakeVoice(),
        log=EventLog(tmp_path / "e.jsonl"), device=device, **kw,
    )


@pytest.mark.asyncio
async def test_silence_timeout_forces_a_turn(tmp_path):
    """If SoulX-Duplug never emits 'speak', the system must not hang forever."""
    orch = make(tmp_path, silence_timeout_ms=2000)
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "rename my contacts", 0))
    await orch.on_tick(now_ms=1000)
    assert orch.registry.all() == []       # not yet
    await orch.on_tick(now_ms=2500)
    assert len(orch.registry.all()) >= 1   # forced dispatch


@pytest.mark.asyncio
async def test_completed_turn_clears_the_silence_timer(tmp_path):
    orch = make(tmp_path, silence_timeout_ms=2000)
    await orch.on_turn_event(TurnEvent(UserState.INCOMPLETE, "partial", 0))
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "rename my contacts to Hans", 500))
    n = len(orch.registry.all())
    await orch.on_tick(now_ms=5000)
    assert len(orch.registry.all()) == n    # no duplicate dispatch


@pytest.mark.asyncio
async def test_hung_reasoner_produces_a_failed_task(tmp_path):
    class HungReasoner:
        async def handle(self, msg):
            import asyncio
            await asyncio.sleep(10)
            return []

    orch = make(tmp_path, reasoner_timeout_s=0.05)
    orch.reasoner = HungReasoner()
    await orch.on_turn_event(TurnEvent(UserState.COMPLETE, "do a thing", 0))
    kinds = [e.kind for e in EventLog.read(orch.log.path)]
    assert "reasoner_timeout" in kinds


@pytest.mark.asyncio
async def test_concierge_bypass_speaks_reasoner_text_directly(tmp_path):
    """With use_concierge=False, reasoner verbatim text goes straight to TTS."""
    orch = make(tmp_path, use_concierge=False)
    await orch.on_reasoner_messages([
        ReasonerMessage(kind="ack", task_id="t1", understood_as="rename"),
        ReasonerMessage(kind="done", task_id="t1", result="renamed 47 contacts"),
    ])
    assert orch.voice.spoken == ["renamed 47 contacts"]
    assert orch.concierge.calls == []
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_resilience.py -v`
Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'silence_timeout_ms'`

- [ ] **Step 3: Extend the orchestrator**

Replace `Orchestrator.__init__` and add the new methods:

```python
    def __init__(
        self, reasoner, concierge, voice, log: EventLog, device,
        silence_timeout_ms: int = 2000,
        reasoner_timeout_s: float = 20.0,
        use_concierge: bool = True,
    ) -> None:
        self.reasoner = reasoner
        self.concierge = concierge
        self.voice = voice
        self.log = log
        self.device = device
        self.registry = TaskRegistry()
        self.policy_state = PolicyState()
        self.history: list[dict] = []
        self.tokens: dict[str, CancellationToken] = {}
        self._seq = 0
        self.silence_timeout_ms = silence_timeout_ms
        self.reasoner_timeout_s = reasoner_timeout_s
        self.use_concierge = use_concierge
        self._pending_partial: str | None = None
        self._pending_since_ms: int | None = None

    async def on_tick(self, now_ms: int) -> None:
        """Drive the silence timeout. Called ~every 160ms by voice-service.

        SoulX-Duplug declining to take the turn is normally correct, but if it
        never fires the system would hang. After silence_timeout_ms of holding
        an incomplete utterance, dispatch it anyway.
        """
        if self._pending_since_ms is None or self._pending_partial is None:
            return
        if now_ms - self._pending_since_ms < self.silence_timeout_ms:
            return
        text = self._pending_partial
        self._pending_partial = None
        self._pending_since_ms = None
        self.log.append("silence_timeout", transcript=text)
        self.history.append({"role": "user", "content": text})
        await self._dispatch(text)
```

In `on_turn_event`, record and clear the pending partial before running the policy:

```python
        if ev.state is UserState.INCOMPLETE and ev.transcript.strip():
            self._pending_partial = ev.transcript
            self._pending_since_ms = ev.t_ms
        elif ev.state is UserState.COMPLETE:
            self._pending_partial = None
            self._pending_since_ms = None
```

(`UserState` is already imported via `.states`; add it to that import.)

Wrap the reasoner call in `_dispatch` with a timeout:

```python
        self.log.append("to_reasoner", **msg.model_dump())
        try:
            replies = await asyncio.wait_for(
                self.reasoner.handle(msg), timeout=self.reasoner_timeout_s
            )
        except asyncio.TimeoutError:
            self.log.append("reasoner_timeout", seq=msg.seq)
            replies = [ReasonerMessage(
                kind="failed", task_id=msg.task_id or "unknown",
                reason="timed out waiting for the reasoner",
            )]
        await self.on_reasoner_messages(replies)
```

And make speaking respect the bypass flag — replace the tail of `on_reasoner_messages`:

```python
        if not should_speak:
            return
        if self.use_concierge:
            await self._ask_concierge("reasoner_update")
            return
        # Bypass: speak the reasoner's verbatim text directly. Nothing can be
        # invented because nothing is generated.
        for m in msgs:
            if m.kind in SPEAK_ON:
                text = m.result or m.reason or m.verbatim_text or m.question
                if text:
                    self.policy_state.speaking = True
                    await self.voice.speak(text, uuid.uuid4().hex)
                    self.policy_state.speaking = False
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest tests/test_resilience.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/rtvoice/orchestrator.py tests/test_resilience.py
git commit -m "feat: silence and reasoner timeouts, concierge bypass flag

Bypass makes 'does the concierge earn its keep' a measurement rather than
an architecture commitment."
```

---

### Task 14: Instrumentation and replay regression

Closes the spec's "standing instruments" and testing Layer 2.

**Files:**
- Create: `src/rtvoice/instruments.py`, `tools/replay_session.py`
- Test: `tests/test_instruments.py`

**Interfaces:**
- Consumes: `EventLog` (Task 5), `StateAdapter` (Task 4), `SoulXClient` (Task 2)
- Produces: `latency_report(events) -> dict` with keys `n`, `mean_ms`, `p50_ms`, `p95_ms`, `max_ms`; `violation_rate(events) -> float`; `tools/replay_session.py` asserting a session reproduces its recorded state sequence.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_instruments.py
import pytest

from rtvoice.events import Event
from rtvoice.instruments import latency_report, violation_rate


def ev(kind, t_ms, **data):
    return Event(kind=kind, t_ms=t_ms, data=data)


def test_latency_measures_turn_to_first_speech():
    events = [
        ev("turn_event", 1000, state="user_complete", transcript="hi"),
        ev("concierge_act", 1400, act="acknowledge", text="on it"),
        ev("turn_event", 5000, state="user_complete", transcript="again"),
        ev("concierge_act", 5600, act="acknowledge", text="sure"),
    ]
    r = latency_report(events)
    assert r["n"] == 2
    assert r["mean_ms"] == 500
    assert r["max_ms"] == 600


def test_latency_ignores_non_dispatching_turns():
    events = [
        ev("turn_event", 0, state="user_incomplete", transcript="and"),
        ev("turn_event", 1000, state="user_complete", transcript="hi"),
        ev("concierge_act", 1200, act="acknowledge"),
    ]
    assert latency_report(events)["n"] == 1


def test_empty_log_reports_zero_not_a_crash():
    assert latency_report([])["n"] == 0


def test_violation_rate_is_violations_over_acts():
    events = [
        ev("concierge_act", 0, act="acknowledge", violations=0),
        ev("concierge_act", 1, act="relay", violations=1),
        ev("concierge_act", 2, act="chat", violations=1),
    ]
    assert violation_rate(events) == pytest.approx(1 / 3)
```

- [ ] **Step 2: Run them and watch them fail**

Run: `uv run pytest tests/test_instruments.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rtvoice.instruments'`

- [ ] **Step 3: Implement**

```python
# src/rtvoice/instruments.py
"""Metrics derived from an event log. Pure functions over Event lists, so they
work identically on a live session and a replayed one.
"""
from __future__ import annotations

import statistics

from .events import Event


def latency_report(events: list[Event]) -> dict:
    """Time from a dispatching turn (user_complete) to the concierge's reply."""
    gaps: list[int] = []
    awaiting: int | None = None
    for e in events:
        if e.kind == "turn_event" and e.data.get("state") == "user_complete":
            awaiting = e.t_ms
        elif e.kind == "concierge_act" and awaiting is not None:
            gaps.append(e.t_ms - awaiting)
            awaiting = None

    if not gaps:
        return {"n": 0, "mean_ms": 0, "p50_ms": 0, "p95_ms": 0, "max_ms": 0}

    gaps.sort()
    pick = lambda q: gaps[min(int(len(gaps) * q), len(gaps) - 1)]
    return {
        "n": len(gaps),
        "mean_ms": int(statistics.mean(gaps)),
        "p50_ms": pick(0.50),
        "p95_ms": pick(0.95),
        "max_ms": gaps[-1],
    }


def violation_rate(events: list[Event]) -> float:
    """Fraction of concierge acts that required a schema re-prompt.

    The spec's key instrument: hallucination risk becomes a number rather
    than something merely suppressed.
    """
    acts = [e for e in events if e.kind == "concierge_act"]
    if not acts:
        return 0.0
    return max(e.data.get("violations", 0) for e in acts) / len(acts)
```

- [ ] **Step 4: Run them and watch them pass**

Run: `uv run pytest tests/test_instruments.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Implement the replay regression tool**

```python
# tools/replay_session.py
"""Testing Layer 2: replay a recorded session's audio and assert the turn
states reproduce. Since recording is always on, the corpus accumulates for
free as the system gets used.

    uv run python tools/replay_session.py sessions/2026-07-27T14-32-05
"""
from __future__ import annotations

import asyncio, pathlib, sys
import numpy as np, soundfile as sf

from rtvoice.events import EventLog
from rtvoice.soulx_client import CHUNK_SAMPLES, SoulXClient
from rtvoice.states import StateAdapter


async def main(session_dir: str) -> int:
    root = pathlib.Path(session_dir)
    audio, sr = sf.read(root / "user.wav", dtype="float32")
    assert sr == 16000, f"expected 16kHz, got {sr}"
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    expected = [
        e.data["state"] for e in EventLog.read(root / "events.jsonl")
        if e.kind == "turn_event"
    ]

    client = SoulXClient()
    await client.connect()
    adapter = StateAdapter()
    actual: list[str] = []
    t_ms = 0
    for i in range(0, len(audio) - CHUNK_SAMPLES, CHUNK_SAMPLES):
        wire = await client.feed(audio[i:i + CHUNK_SAMPLES])
        for ev in adapter.feed(wire, t_ms):
            actual.append(ev.state.value)
        t_ms += 160
    await client.close()

    if actual == expected:
        print(f"MATCH  {len(actual)} states reproduced")
        return 0

    print(f"MISMATCH  expected {len(expected)} states, got {len(actual)}")
    for i, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            print(f"  first divergence at {i}: expected {a}, got {b}")
            break
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(sys.argv[1])))
```

- [ ] **Step 6: Verify against a real session**

```bash
# after running a live session via /inject or audio:
uv run python tools/replay_session.py sessions/dev
uv run python -c "
from rtvoice.events import EventLog
from rtvoice.instruments import latency_report, violation_rate
evs = EventLog.read('sessions/dev/events.jsonl')
print('latency:', latency_report(evs))
print('violation rate:', violation_rate(evs))
"
```

Expected: `MATCH` from the replay, and a latency report with `p95_ms` in the 360–550 ms band the spec predicts. A p95 far above that means the tunnel is the problem — revisit the Task 1 decision.

- [ ] **Step 7: Run the whole suite and commit**

Run: `uv run pytest -v`
Expected: PASS — all tests from Tasks 1–14 green.

```bash
git add src/rtvoice/instruments.py tools/replay_session.py tests/test_instruments.py
git commit -m "feat: latency and violation instruments, replay regression tool"
```

---

## What this plan does NOT cover

Deferred to a follow-up plan, which renders entirely from `events.jsonl`:

- Browser client (WebAudio capture/playback, `echoCancellation: true`, headphones)
- Instrument UI panels: state timeline, transcript, task registry cards, reasoner terminal, device-state diffs, concierge acts, latency waterfall
- Replay controls and the session browser
- Screen capture export and session zip bundling

The event log written by Task 5 and populated throughout Task 12 is the complete interface between the two plans. Nothing in the UI plan requires changes here.
