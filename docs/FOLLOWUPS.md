# Known issues and follow-ups

State as of the core-loop branch (`feat/voice-interface-core-loop`, 353 tests passing).
Everything here was found by review and consciously carried rather than missed.

## The LLM reasoner (`REASONER=llm`)

`LlmReasoner` has never spoken to a real model — every test drives a faked HTTP POST, which
is what keeps the suite offline. Everything downstream of the model's JSON is proven; the
quality of the JSON itself is not. What to watch for on the 3090:

- **Filters are equality-only.** `DeviceState._matches` compares with `==`, so there is no
  range, prefix or substring matching. "Messages from last week" is not expressible; the
  fixture works around it by carrying a date-only `sent` on messages and a date-only `day`
  beside the ISO `when` on calendar entries. A `find contacts called Nair` works only if the
  model filters on `last_name` exactly. Any real corpus will want operators.
- **Relative dates are pre-resolved in the prompt** (`_date_block`) rather than trusted to
  the model. If a session runs past midnight the block is only recomputed per utterance, so
  a task planned before midnight and confirmed after it still writes the date it was planned
  with — which is the desired behaviour, but only by accident.
- **A wrong-but-valid filter is undetectable.** The count in `confirm_required` is always
  true for the filter that will run, but if the model plans `group = "work"` when the user
  meant "the London group", the user hears an accurate count of the wrong rows. The
  confirmation makes it audible; nothing makes it impossible.
- **`misreads` is the metric to watch.** It counts transport failures, unparseable output and
  plans naming things the device does not have. None of them mutate anything, so a high rate
  is a quality signal, not a safety one.
- **No undo.** The journal records every staged operation including the record contents of a
  delete, so "undo the 12 I already deleted" is *answerable*, but nothing implements it.

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

## Carried by the fragment merge window

The turn-taking measurement above came back badly: one 21s compound instruction arrived as
**six** `user_complete` events, because the speaker paused at clause boundaries and each
fragment genuinely is a complete sentence. `Orchestrator.merge_window_ms` (default 1200 ms)
now holds a finalised utterance and concatenates whatever follows before the reasoner sees
it. Three things about it are worth knowing:

- **Merging is gated on the orchestrator being clocked** (`Orchestrator._clocked`). An
  utterance is only held if `on_tick` has already reached that event's timestamp, which is
  always true on the audio path (`AudioDriver` ticks after every 160 ms chunk) and never
  true for `POST /inject`, which synthesises a `COMPLETE` at t=0 and reads the resulting
  tasks out of its own response. Without the gate, injected text would be held for a window
  nothing was going to close. The consequence is that the merge window is inert for any
  caller driving `on_turn_event` without `on_tick` — including most of the existing test
  suite, which is why those tests still assert synchronous dispatch.
- **The window alone is far too short for the measured gaps** (2.4–3.7 s between fragments).
  What actually holds the buffer open is `user_nonidle` refreshing it while the user is
  audibly still speaking. If echo cancellation fails and the mic hears the TTS, NONIDLE
  becomes continuous and a merge could be held open indefinitely.
- **`on_tick` now awaits a reasoner round trip on the common path**, not just on the rare
  silence timeout. `on_turn_event` already did this, so it is not new, but it means a slow
  reasoner stalls the audio feed for longer stretches than before.

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
