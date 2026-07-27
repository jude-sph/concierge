# Known issues and follow-ups

State as of the core-loop branch (`feat/voice-interface-core-loop`, 185 tests passing).
Everything here was found by review and consciously carried rather than missed.

## Blocked on hardware

The 3090 was never reachable during this build, so three things are unverified:

- **Tunnel latency and jitter** (plan Task 1). The whole deployment shape depends on it:
  p95 > 150 ms or jitter > 50 ms means moving `voice-service` to the Mac. Run
  `tools/measure_tunnel.py` first.
- **SoulX-Duplug wire behaviour** (plan Task 2 step 6). `soulx_client.py` is unit-tested
  against the documented protocol but has never spoken to the real server.
- **Turn-taking on disfluent compound commands** (plan Task 3). This is the measurement most
  likely to invalidate the design — the paper reports 0.352 pause-takeover on their English
  set, and our command style ("find chinese restaurants in soho and… uh… set all my contacts
  to Hans") may be worse. `tools/make_corpus.py` and `tools/measure_turntaking.py` were never
  written because they cannot be exercised. Do this early.

Also unverified for the same reason: real Kokoro streaming, real audio through
`tools/audio_client.py`, and uvicorn `--factory` startup. The CLI *was* smoke-run end to end
with only the websocket faked — audio in, contacts renamed on disk, committing only on an
explicit "yes".

## Should fix before a demo that relies on barge-in

**A stale stacked confirmation permanently disables reflexive barge-in.** When the silence
timer force-dispatches a truncated destructive command and the user then completes that
sentence, the completion is correctly treated as new content — which makes the reasoner plan
the same rename a second time. Two identical `awaiting_confirm` tasks now exist from one
spoken sentence. Answering "yes" resolves only the newer one; the older lingers forever,
`pending_question` stays set, and `policy.decide()` reads every subsequent NONIDLE as an
answer rather than an interruption. Measured: barge-in dead across all following commands.

Not a data-safety bug — nothing mutates without an explicit answer-shaped affirmative — but
"Stop! always halts speech immediately" is a headline guarantee of the design.

Likely fix: make the silence-timeout reconciliation prefix-aware rather than exact-match, so
the superset completion resolves the existing task instead of creating a duplicate.

## Cheap usability fixes

- **The answer-shape whitelist rejects natural confirmations.** `"yes go ahead and do it"`
  fails only because `and` is not in `ANSWER_VOCABULARY`. Also rejected: `"sure thing"`,
  `"yeah that works"`, `"do it now"`, `"go ahead then"`. Fail-safe (the question stays open;
  a bare "yes" works) but a bad demo moment. Worth one vocabulary pass.
- **A wordy refusal no longer cancels.** `"no, sure thing"` is not answer-shaped, so it
  returns `noop` and leaves the destructive task armed rather than cancelling it. Contained,
  since stray affirmatives can no longer commit, but the user said no.
- **Open clarifications still swallow new commands.** The fix stopped yes/no confirmations
  from swallowing plainly-new utterances, but `need_clarification` still routes any non-empty
  utterance to itself. With one pending, `"delete my photos"` never produces a task.

## Instrument accuracy

- **`latency_report` has no turn identity.** It pairs "the next `concierge_act`" with "the
  last `user_complete`". Under concurrency, two completed turns before one acknowledgment
  silently drops the first and mis-attributes the second. Inherent to reconstructing pairing
  from an unkeyed event stream — fixing it means putting a turn id in the events.
- **Bypass mode reports no latency at all.** With `use_concierge=False` there is no
  `concierge_act`, so `latency_report` returns `n=0` for sessions that really did speak.
- **`violation_rate` is not bounded to [0,1].** It is mean invalid generations per act, and a
  single turn can contribute 2. The docstring says so; the name is still misleading.

## Deliberately carried

Reviewed, ruled acceptable, and re-validated at the final review:

- Unbounded subscriber queues on the event log — one local browser tab, YAGNI.
- `confirm_required` and `need_clarification` sharing `AWAITING_CONFIRM` — the orchestrator
  branches on message kind, and `clarification_answer` is right for both.
- A reconciled turn produces one duplicate acknowledgment ("on it" twice) — redundant audio
  only, no wrong facts and no double work.
- `DeviceState.commit()` sets `_committed` before the disk write; module-global task id
  counter; `requires-python` not pinned to exactly 3.10.
- Stacked confirmations are answered most-recent-first. A choice, not a necessity, but it
  matches conversational expectation and nothing commits without an explicit affirmative.

## Not started

The instrument UI, browser audio client, replay controls and session bundling are a separate
follow-up plan. `events.jsonl` is the complete interface between the two — nothing in the UI
plan requires changes to this one.
