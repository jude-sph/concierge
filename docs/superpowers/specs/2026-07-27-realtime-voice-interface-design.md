# Real-Time Voice Interface for a Memory-Enabled Reasoning Agent

**Date:** 2026-07-27
**Status:** Approved design, ready for implementation planning

## Context

A separate project provides a *memory-enabled reasoner*: a model that reads and writes a
mobile device's memory directly, bypassing the device UI. This project builds the
conversational interface to it — a real-time, two-way voice layer through which a human
issues high-level, abstract commands ("find Chinese restaurants in Soho and change all my
contacts to Hans and get an Uber"), discusses and refines them, and hears what happened.

We do not have access to the real reasoner yet. This build defines the protocol and
substitutes a local model role-playing the reasoner against a fake device state.

### Why not an end-to-end speech model

The obvious approach — Moshi or a similar full-duplex speech-to-speech model — was
evaluated and rejected on evidence:

- Moshi ranked last in the ICASSP 2026 HumDial full-duplex track (34.5, vs Freeze-Omni
  43.8, Gemini-2.5 62.3, best entry 76.6). Cascaded systems swept the top three places.
- Its pause-handling takeover rate is **0.983** — it interrupts on 98.3% of mid-sentence
  pauses. Our users issue long compound commands with hesitations, so this is
  disqualifying on its own.
- Audio training costs it most of its factual knowledge (TriviaQA 22.8 vs 56.4 for its own
  text backbone), so it confabulates. Every open full-duplex model is 7–9B and shares this.
- Since the reasoner authors all facts, an end-to-end model's generative ability is wasted
  and its hallucination is pure liability.

A cascade with a dedicated turn-taking module gets better turn management, faithful
speech by construction, and a fraction of the footprint.

## Goals

1. **Prove the interaction pattern.** A working loop that demonstrates the concierge /
   reasoner split is the right decomposition.
2. **De-risk the unknowns.** Specifically: turn-taking on disfluent compound commands,
   concierge hallucination, async result delivery, and cancellation safety.
3. **Demoable end-to-end.** Speak a compound command, watch fake device memory actually
   change, hear it reported back.

### Non-goals

- **On-device / phone deployment.** Explicitly out of scope. Component boundaries are drawn
  so the phone story stays open, but no footprint optimisation happens now.
- Diffusion-rendered UI, click interpretation, or any non-voice input *as a product surface*.
  (The UI's manual text injection is a development affordance, not an input modality.)
- Production hardening, multi-user, authentication.
- Beating any benchmark. This is an architecture probe.

## Architecture

Five services across three processes.

```
┌─ Mac: browser ───────────┐   ┌─ 3090 (via SSH tunnel) ──────────────────┐
│                          │   │                                           │
│  audio I/O               │◄WS┼─►┌─────────────────────────────────────┐ │
│   · mic capture          │PCM│  │ voice-service                       │ │
│   · playback (headphones)│16k│  │  · SenseVoice Small (STT)           │ │
│                          │   │  │  · SoulX-Duplug (state @ 160ms)     │ │
│  instrument UI           │   │  │  · Kokoro (TTS)                     │ │
│   · state timeline       │   │  │  · dual-channel recorder            │ │
│   · transcripts          │   │  └───────────────┬─────────────────────┘ │
│   · task registry        │   │                  │ (state, transcript)    │
│   · device state + diffs │   │                  ▼                        │
│   · replay controls      │◄WS┼──┌─────────────────────────────────────┐ │
│                          │evt│  │ orchestrator       ← the real build  │ │
└──────────────────────────┘   │  │  · turn policy                      │ │
                               │  │  · task registry (source of truth)  │ │
                               │  │  · cancellation tokens              │ │
                               │  │  · concierge (3–8B)                 │ │
                               │  │  · event log (JSONL)                │ │
                               │  │  · static UI + event WS             │ │
                               │  └───────────────┬─────────────────────┘ │
                               │                  │ protocol messages      │
                               │                  ▼                        │
                               │  ┌─────────────────────────────────────┐ │
                               │  │ reasoner-stub                       │ │
                               │  │  · 8–14B roleplaying memory agent   │ │
                               │  │  · journaled device_state.json      │ │
                               │  └─────────────────────────────────────┘ │
                               └───────────────────────────────────────────┘
```

### Deployment rationale

All models run on the 3090. SoulX-Duplug is CUDA-native and its shipped config works
unmodified, avoiding an MPS port whose main unknown (the GLM-4-Voice tokenizer's operator
coverage) has no bearing on the goals above.

The 3090 sits behind two SSH hops on a university network. `ProxyJump` collapses this:

```
Host gpu3090
    HostName <box>
    ProxyJump user@<uni-gateway>
    LocalForward 8000 localhost:8000
```

Audio is 256 kbps each way at 16 kHz — bandwidth is a non-issue. Latency is not:

| Stage | Cost |
|---|---|
| Audio up the tunnel | RTT/2 |
| SoulX-Duplug turn detection | 240 ms |
| Concierge first token | 50–150 ms |
| TTS first chunk | 50–100 ms |
| Audio back down | RTT/2 |
| **User stops → first sound** | **~360–550 ms** |

Total added network cost is 1× RTT, not 2×. For reference: Moshi 200 ms, human
conversational gaps ~230 ms, production voice agents on FDB-v3 4.25 s+.

**The risk is jitter, not mean latency.** A steady 500 ms is fine; 300 ms spiking to 1.5 s
is not, and TCP over two SSH hops is where that lives. Barge-in stays crisp regardless at
~RTT + 160 ms.

**Escape hatch.** The voice-service boundary is drawn so it can move to the Mac (~2.5 GB
total) if the tunnel measures badly, leaving only text on the wire. This is a deployment
change, not a rewrite. It is also the configuration that generalises to a phone.

**First task in implementation is to measure the tunnel** — sustained RTT and jitter — since
that decides whether the split is ever needed.

## Components

| Component | Responsibility | Notes |
|---|---|---|
| **browser client** | Mic capture, audio playback, instrument UI, replay | Two sockets: audio to voice-service, events to orchestrator |
| **voice-service** | Owns all audio. Emits `(state, transcript_delta)` every 160 ms; accepts `speak()` / `stop()`; records both channels | The relocatable boundary |
| **orchestrator** | Turn policy, task registry, cancellation, concierge, event log, UI hosting | Everything novel lives here |
| **reasoner-stub** | Roleplays the memory agent against journaled `device_state.json` | Swapped for the real reasoner via the same protocol |

`device_state.json` holds contacts, messages, calendar entries and places. The stub genuinely
reads and writes it, so "set all contacts to Hans" is verifiable by diffing a file.

## Protocols

### voice-service → orchestrator

```json
{"state": "user_complete", "transcript_delta": "...", "chunk": 1247}
```

States are SoulX-Duplug's five: `user_idle`, `user_nonidle`, `user_backchannel`,
`user_complete`, `user_incomplete`. Plus lifecycle events `speaking_started` and
`speaking_finished(utterance_id)` so the orchestrator knows whether a barge-in actually cut
something off.

### orchestrator → voice-service

`speak(text, utterance_id)`, `stop()`.

### orchestrator ↔ reasoner

Async, unordered, both directions. Messages carry monotonic sequence numbers for resync.

| → reasoner | ← reasoner |
|---|---|
| `utterance(text, raw_transcript, t)` | `ack(task_id, understood_as)` |
| `clarification_answer(task_id, text)` | `need_clarification(task_id, missing, options)` |
| `cancel(task_id)` | `progress(task_id, status)` |
| `nudge(task_id)` | `done(task_id, result)` / `failed(task_id, reason)` |
| | `confirm_required(task_id, verbatim_text)` |
| | `noop` |

`nudge` is sent when the user asks after a running task ("is that done yet?") and the
registry has nothing newer than the last `progress` — it prompts the reasoner for a fresh
status without implying cancellation.

`cancel` is sent when a task is aborted, so the reasoner can update its own state. It is
**not** the mechanism that stops execution — see *Cancellation is a control-plane operation*
below. The orchestrator fires the cancellation token first and notifies the reasoner second;
correctness never depends on the reasoner receiving or acting on the message.

## Turn policy

| Trigger | Action |
|---|---|
| `user_complete` | Send `utterance` to reasoner **and** have concierge respond immediately — the two run in parallel |
| `user_incomplete` | Do nothing; keep listening. This is why SoulX-Duplug is here |
| `user_nonidle` while speaking | `stop()` reflexively — unless a question is pending, in which case treat as its answer |
| `user_backchannel` | Do nothing; keep speaking |
| Reasoner message arrives | Concierge speaks on `done` / `failed` / `need_clarification` / `confirm_required`; registry updates silently on `progress` |
| Silence timeout during `user_incomplete` | Force `user_complete` |

The concierge never speaks on a timer. Acknowledge on dispatch, report on completion, stay
responsive in between — that is the long-wait behaviour.

### The reasoner is the gatekeeper

**Every finalised utterance goes to the reasoner**, not only ones judged actionable. The
concierge is too small to safely decide what constitutes a device write; the reasoner needs
full conversation context to resolve anaphora ("do that one too"); and it costs nothing when
both live on the same box. The reasoner replies `noop` for conversation.

This was motivated by safety but also turns out to be the primary defence against
turn-taking errors. SoulX-Duplug still takes the floor on ~35% of mid-sentence pauses, so
truncated commands *will* reach the reasoner — which rejects them as unactionable. If the
concierge were dispatching, a mid-sentence cut would become a device write. One decision,
two risks covered.

## The concierge

### Parallel, not serial

The concierge and reasoner are **two heads on the same stream**, not a pipeline. Both see
every utterance; neither waits for the other. The concierge responds to every turn at
conversational latency; the reasoner acts only when there is something to act on. They meet
at the fact block.

This matters because most of what a user says is not for the reasoner: *"sorry, say that
again"*, *"what was the second thing?"*, *"hold on"*, *"no, I meant the work ones"*. In a
pipeline each costs a reasoner round-trip while it is mid-operation. In parallel the
concierge simply answers.

| | Concierge | Reasoner |
|---|---|---|
| Owns | the conversation | device + task truth |
| Responds to | every turn, immediately | only actionable ones |
| Dialogue repair | ✅ from its own history | never sees it as actionable |
| Deciding what is actionable | ❌ | ✅ |
| Factual values | ❌ | ✅ verbatim |
| Clarification | phrases and times the question | supplies the information *need* |
| Destructive confirmations | relays verbatim, no rephrasing | authors exactly |

### Preventing invented task status

No pattern matching on generated prose anywhere. Three structural measures:

**1. Narrow context.** The concierge never sees the task registry's internals or history.
Each turn it receives a rendered **fact block** — the complete, current, authoritative set of
task facts — and nothing else. Invented status mostly comes from a model filling gaps in
stale or partial state; removing the gaps removes the failure at source.

**2. Split authorship.** The concierge writes conversational framing. The reasoner writes
factual claims, which pass through verbatim. The concierge composes; it does not assert.

**3. Typed speech acts.**

```json
{"act": "relay",       "cites": "task_2", "text": "..."}
{"act": "ask",         "text": "..."}
{"act": "acknowledge", "text": "..."}
{"act": "abort",       "cites": "task_2"}
{"act": "chat",        "text": "..."}
```

Validation is a schema check on typed fields: a `relay` must cite a live task id and must
contain the reasoner's verbatim span. This is the same class of check as validating any
tool call's arguments — no inspection of natural language. On violation: re-prompt once with
the schema error, then fall back to `acknowledge`. Never rewrite.

This turns the mechanism from a filter into an **instrument**: "how often does the concierge
attempt an uncited relay?" becomes a measurable number, which serves the de-risking goal
better than silent suppression would.

Consequence for model selection: the concierge needs reliable structured output, so choose
an instruction-tuned model in the Qwen3-4B class rather than the smallest thing that chats.

### Removability

Whether the concierge earns its place depends on one empirical question: **how slow is the
real reasoner while working?** If it can turn around a conversational reply in ~200 ms, the
concierge is dead weight. If memory operations block it for seconds, conversation dies
without one. We cannot know until the real reasoner exists, and the stub's latency is chosen
rather than discovered.

Therefore the concierge is **removable by config flag**, routing conversational turns either
to it or straight to the reasoner. The answer becomes a measurement rather than an
architecture commitment.

## Safety and cancellation

### Hierarchy of defences

| Defence | Protects against | Speed |
|---|---|---|
| **Two-phase commit** — `confirm_required` before any destructive write | The main case: user objects before anything happens | Instant, no inference |
| **Journaled writes + rollback** | Partial execution already committed | Post-hoc |
| **Concierge `abort`** | Long operations after confirmation | ~150 ms inference |

**Honest limitation:** for fast destructive operations no interrupt is fast enough — renaming
47 contacts takes ~80 ms, less time than the user needs to object. Confirm-before-execute is
the real protection; `abort` is the second line and must not substitute for the gate.

### Cancellation is a control-plane operation

A busy reasoner cannot hear "stop" because it is mid-script. The fix is not a faster way to
tell it — it is that **cancellation does not require the reasoner's cooperation**. Its
executor runs against a cancellation token checked between operations
(`for c in contacts: check_cancel(); rename(c)`). The orchestrator holds that token. Firing
it involves the reasoner model no more than `SIGINT` involves asking a process nicely.

### Layering "stop" by cost

"Stop!" is ambiguous between *stop talking* and *stop doing the thing*. Resolve by cost:

- **Barge-in is reflexive** — fires on `user_nonidle` with zero inference, because halting
  TTS is harmless and instantly reversible.
- **Abort is deliberate** — requires the concierge to judge intent via an `abort` speech act,
  because killing a task mid-write is consequential.

So "stop!" always halts speech immediately, and the concierge then decides within ~150 ms
whether it also meant the task. After an abort, partial state may exist, so the natural
follow-up is *"I stopped partway — want me to undo the 12 I'd already changed?"* The journal
makes that answerable.

## Error handling

| Failure | Handling |
|---|---|
| Turn-taking fires mid-sentence | Partial reaches the reasoner, which replies `need_clarification` or `noop` |
| Turn-taking never fires | Silence timeout forces `user_complete` |
| STT mishears a value ("Hans" → "Hants") | Raw transcript accompanies the paraphrase; destructive ops echo the *actual value* verbatim for confirmation, so the user hears the error |
| Concierge schema violation | Re-prompt once, then fall back to `acknowledge`. Logged as a metric |
| Reasoner never replies | Orchestrator timeout synthesises `failed(task_id, "timed out")` |
| Tunnel drops mid-task | Client buffers and reconnects; tasks continue server-side; registry replays via sequence numbers |
| Barge-in race (interrupt as TTS starts) | `utterance_id` tracking so a stale `speaking_finished` cannot corrupt state |

## Testing

Three layers, arranged so most tests need neither models nor a microphone.

**Layer 1 — orchestrator logic (fast, pure).** Mock voice-service and reasoner; drive state
sequences as fixtures. Covers turn policy, task registry transitions, barge-in, cancel and
rollback, schema validation, timeouts. Most of the build's logic lives here and runs in
milliseconds.

**Layer 2 — voice-service replay.** Replay recorded sessions (`user.wav` from any session
directory) through the real voice-service. Same audio in → same `(state, transcript)`
sequence out, asserted against that session's `events.jsonl`. Regression-tests turn-taking
without speaking into a mic. Since recording is always on, the corpus accumulates for free
as the system gets used.

**Layer 3 — end-to-end.** Assertions on `device_state.json`. "Set all contacts to Hans" →
diff the file. Writes are verifiable, not vibes.

### The measurement that matters most

Synthesise a corpus in *our* command style — compound, multi-intent, hesitations mid-sentence
— and measure how often `user_complete` fires early. SoulX-Duplug reports 0.352 pause-takeover
on their English set; the open question is what it is on *"find chinese restaurants in soho
and… uh… set all my contacts to Hans"*. If that number is bad it is the single thing most
likely to sink the design, and it should be known in week one.

### Standing instruments

Logged every session:

- Concierge citation-violation rate
- Dispatch-to-first-audio latency, mean **and** tail
- Tunnel RTT and jitter
- Turn-taking false-positive / false-negative counts

## Web UI and observability

The browser replaces a separate audio client: it captures mic and plays audio via WebAudio
over the same tunnel, and hosts the instrument panel. Note this is **not** the WebRTC option
rejected earlier — raw PCM over a WebSocket needs no STUN/TURN/UDP, so the university
firewall is irrelevant.

Two sockets preserve the existing boundary: browser → voice-service (binary audio),
browser → orchestrator (JSON events). If voice-service later moves to the Mac, the browser
points at localhost and the escape hatch gets *simpler*, not harder.

**Acoustic echo must be handled from day one.** If the mic hears the TTS, SoulX-Duplug emits
`user_nonidle` and the system barges in on itself continuously. Use
`getUserMedia({echoCancellation: true})` and headphones for demos. Gating the mic while
speaking would also fix it but destroys barge-in, so it is not an option.

### Everything renders from one event stream

The orchestrator emits an append-only JSONL event log — every state token, protocol message,
task transition, device write and metric. The UI is a pure function of that stream.

This is the design point that pays for itself: live view is a subscription; session replay
feeds a recorded log to the same UI with no separate code path; Layer 2 tests assert on the
same log; and demos become reproducible, since a good run can be replayed if the live one
misbehaves. The event log is a first-class artifact, not a debug afterthought.

All timestamps are monotonic offsets from a recorded session start, so audio and events
align frame-accurately on replay.

### Panels

| Panel | Shows | Why |
|---|---|---|
| **State timeline** | Scrolling colour-coded strip of the 160 ms state stream | The money shot — watch it hold through a hesitation on `user_incomplete` instead of barging in. Makes the invisible thing the architecture rests on visible |
| **Transcript** | Both sides, state annotations inline, per-turn timing | |
| **Task registry** | Live cards: id, `understood_as`, status, elapsed | Three cards from one utterance is how compound dispatch demonstrates itself |
| **Reasoner terminal** | Streamed reasoning tokens + operations log, phone-terminal styled | `> UPDATE contacts SET first_name='Hans'  47 rows ✓ committed` |
| **Device state** | Live `device_state.json` with changed fields highlighted on write | The terminal shows the operation; this shows the consequence |
| **Concierge acts** | Raw speech acts as JSON (`act`, `cites`, `text`) + violation counter | Makes the citation discipline visible rather than theoretical |
| **Instruments** | Per-turn latency waterfall (turn-detect / concierge / TTS / network), citation-violation rate, turn-taking FP/FN, tunnel RTT and jitter | Shows *where* the ~450 ms went |

### Development affordances

- **Manual text injection** — type as if spoken, bypassing audio. Needed constantly during
  development and makes the orchestrator exercisable without a mic.
- **Reasoner latency slider** — artificially slow the stub to 5 s / 30 s. The stub's latency
  is chosen rather than discovered, so this is how long-wait behaviour gets tested on demand,
  and how the "does the concierge earn its keep" question gets answered.

### Session recording

**Recording is always on.** Storage is trivial (16 kHz mono WAV is ~32 kB/s; a ten-minute
session is ~20 MB for both channels), and always-on means a good accidental take is never
lost. Sessions are curated after the fact with a "mark" button and a session browser, rather
than by remembering to press record beforehand.

Each session produces a self-contained directory:

```
sessions/2026-07-27T14-32-05/
  events.jsonl        # complete event log — the replayable artifact
  user.wav            # mic input, 16 kHz mono
  model.wav           # TTS output, 16 kHz mono
  mix.wav             # stereo: user left, model right
  device_state.json   # final state
  device_journal.jsonl# every write, for diffing and rollback
  meta.json           # duration, marks, config, model versions
  screen.webm         # optional, if screen capture was used
```

**Audio is recorded server-side in voice-service**, not in the browser — it already holds
both PCM streams, avoids `MediaRecorder` frame loss, and is already aligned to the event
clock. Channels are kept **separate** as well as mixed, so user and model audio can be
analysed independently (and so echo can be diagnosed rather than guessed at).

**Video is an export, not the recording.** The event log plus audio *is* the session and
replays perfectly in the UI. `getDisplayMedia` screen capture is a one-click convenience for
when an actual video file needs to leave the machine.

One **Export** button bundles the session directory as a zip.

### Explicitly out of scope for the UI

No auth, no multi-session, no persistence beyond the session directories, no responsive or
mobile layout. Single user, desktop, local — an instrument panel, not a product. Plain
HTML/JS with **no build step**, served by the orchestrator's existing FastAPI process; adding
a bundler to a demo UI costs a day and returns nothing.

## Open questions

Deliberately left unresolved, to be answered by measurement rather than argument:

1. **Tunnel viability.** Does jitter over two SSH hops force the voice-service onto the Mac?
2. **Turn-taking on our command style.** Is the pause-takeover rate tolerable for compound
   commands with hesitations?
3. **Does the concierge earn its keep?** Answerable only once the real reasoner's working
   latency is known. Hence the config flag.
4. **Concierge size.** Start at 3–8B; shrink only once behaviour is correct.
