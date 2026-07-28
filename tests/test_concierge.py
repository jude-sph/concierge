"""The concierge returns plain spoken text.

It used to emit a typed JSON speech act with a `relay` variant that cited a
task and repeated its wording verbatim. That machinery guarded against a small
model inventing facts while reporting results -- but results no longer pass
through the concierge at all (Orchestrator._speak_facts speaks them directly),
so the guard was protecting a job that had already moved. Its replacement is
structural: the concierge is never handed the phone's data, so it has nothing
to leak.

What is left to test is what actually reaches the speaker, and that the
conversational layer can never take down a turn.
"""
import httpx
import pytest

from rtvoice.concierge import (MAX_REPLY_CHARS, Concierge, _phrase_key,
                               clean_reply)
from rtvoice.protocol import ReasonerMessage
from rtvoice.registry import TaskRegistry


# --- clean_reply: what actually reaches the speaker -----------------------

def test_a_plain_sentence_passes_through():
    assert clean_reply("Let me check your calendar.") == "Let me check your calendar."


def test_surrounding_quotes_are_stripped():
    # Small models routinely quote the sentence they were asked to produce.
    assert clean_reply('"Sure, one sec."') == "Sure, one sec."
    assert clean_reply("'On it.'") == "On it."


def test_a_speaker_label_is_stripped():
    assert clean_reply("Assistant: Hi there!") == "Hi there!"
    assert clean_reply("concierge: Hi there!") == "Hi there!"


def test_markdown_fences_are_stripped():
    assert clean_reply("```\nOn it.\n```") == "On it."


@pytest.mark.parametrize("placeholder", ["...", "…", "  ", "", "-", "!!!"])
def test_placeholders_are_dropped_rather_than_spoken(placeholder):
    """A reply with nothing in it must produce silence.

    "..." was previously read out literally as "dot dot dot" -- a model
    copying the placeholder straight out of its own prompt template.
    """
    assert clean_reply(placeholder) == ""


def test_a_real_sentence_ending_in_an_ellipsis_survives():
    assert clean_reply("Hold on, let me check...") == "Hold on, let me check..."


def test_a_rambling_reply_is_cut_at_a_sentence_boundary():
    long_reply = ("Sure thing, I can help with that. " * 20).strip()
    out = clean_reply(long_reply)
    assert len(out) <= MAX_REPLY_CHARS
    assert out.endswith(".")


# --- respond(): the conversational call -----------------------------------

class _FakePost:
    """Stands in for httpx.AsyncClient.post; records what was sent."""

    def __init__(self, content=None, exc=None):
        self.content = content
        self.exc = exc
        self.payloads = []

    async def __call__(self, url, json=None, **kw):
        self.payloads.append(json)
        if self.exc is not None:
            raise self.exc
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": self.content}}]},
            request=httpx.Request("POST", url),
        )

    def system_text(self):
        return " ".join(m["content"] for m in self.payloads[0]["messages"]
                        if m["role"] == "system")


@pytest.mark.asyncio
async def test_respond_returns_the_spoken_sentence(monkeypatch):
    c = Concierge()
    monkeypatch.setattr(c._client, "post", _FakePost(content="Let me check that."))
    assert await c.respond(TaskRegistry(), [], "user_turn") == "Let me check that."


@pytest.mark.asyncio
async def test_the_in_flight_utterance_is_given_to_the_model(monkeypatch):
    """Without this the concierge speaks blind.

    It is asked to reply at the moment the reasoner has not answered yet.
    Told nothing about what is running, it denied a capability it has --
    "I don't have information about your calendar" while a calendar lookup
    was already in progress. Knowing what is in flight is what makes
    "let me check that" a true statement.
    """
    c = Concierge()
    post = _FakePost(content="Let me check your calendar.")
    monkeypatch.setattr(c._client, "post", post)

    await c.respond(TaskRegistry(), [], "user_turn",
                    in_flight="tell me about my calendar")

    assert "tell me about my calendar" in post.system_text()


@pytest.mark.asyncio
async def test_known_tasks_are_given_to_the_model(monkeypatch):
    r = TaskRegistry()
    r.apply(ReasonerMessage(kind="ack", task_id="t1", understood_as="rename contacts"))
    c = Concierge()
    post = _FakePost(content="On it.")
    monkeypatch.setattr(c._client, "post", post)

    await c.respond(r, [], "user_turn")

    assert "rename contacts" in post.system_text()


@pytest.mark.asyncio
async def test_an_unreachable_model_is_silent_not_fatal(monkeypatch):
    """The conversational layer must never take down a turn.

    The reasoner's work proceeds regardless, and its result is spoken by a
    path that does not involve this component at all.
    """
    c = Concierge()
    monkeypatch.setattr(c._client, "post",
                        _FakePost(exc=httpx.ConnectError("refused")))
    assert await c.respond(TaskRegistry(), [], "user_turn") == ""
    assert c.violations == 1


@pytest.mark.asyncio
async def test_an_unspeakable_reply_is_counted_and_silenced(monkeypatch):
    c = Concierge()
    monkeypatch.setattr(c._client, "post", _FakePost(content="..."))
    assert await c.respond(TaskRegistry(), [], "user_turn") == ""
    assert c.violations == 1


@pytest.mark.asyncio
async def test_history_is_bounded(monkeypatch):
    """A long session must not grow the prompt without limit -- every token
    is latency in the component whose whole job is to answer quickly."""
    c = Concierge()
    post = _FakePost(content="Sure.")
    monkeypatch.setattr(c._client, "post", post)

    history = [{"role": "user", "content": f"line {i}"} for i in range(50)]
    await c.respond(TaskRegistry(), history, "user_turn")

    spoken = [m for m in post.payloads[0]["messages"] if m["role"] != "system"]
    assert len(spoken) <= 8
    assert spoken[-1]["content"] == "line 49"


# --- speaking freely, not reciting ------------------------------------------
#
# "the model says I'm onto it for everything which is getting a bit annoying".
# A small model asked to acknowledge a request converges hard on one phrase and
# then says it every single turn, which is the most robot-like thing this
# component does -- in a component that exists to make the exchange feel like a
# conversation.

def test_the_same_opening_counts_as_a_repeat():
    """Exact-match would miss it: "I'm onto it." and "I'm onto it right now!"
    are the same tic, and treating them as different replies is how the tic
    survives the check."""
    assert _phrase_key("I'm onto it.") == _phrase_key("I'm onto it right now!")
    assert _phrase_key("Sure thing.") != _phrase_key("I'm onto it.")


class _ScriptedPost(_FakePost):
    """Returns a different reply per call, so a re-roll can be observed."""

    def __init__(self, *contents):
        super().__init__(content=None)
        self.contents = list(contents)

    async def __call__(self, url, json=None, **kw):
        self.content = self.contents[min(len(self.payloads), len(self.contents) - 1)]
        return await super().__call__(url, json=json, **kw)


@pytest.mark.asyncio
async def test_a_repeated_reply_is_re_rolled(monkeypatch):
    """The model, not the fake, has to be the one repeating itself: the
    re-roll only fires when a generation comes back matching a recent one."""
    c = Concierge()
    post = _ScriptedPost("I'm onto it.", "I'm onto it.", "Sure, give me a second.")
    monkeypatch.setattr(c._client, "post", post)

    assert await c.respond(TaskRegistry(), [], "user_turn") == "I'm onto it."
    second = await c.respond(TaskRegistry(), [], "user_turn")

    assert second == "Sure, give me a second."
    assert c.repeats == 1
    assert len(post.payloads) == 3, "the repeat cost one extra call, no more"


@pytest.mark.asyncio
async def test_a_fresh_reply_costs_no_extra_call(monkeypatch):
    """Re-rolling unconditionally would double every turn's latency in the one
    component whose whole job is to answer fast."""
    c = Concierge()
    post = _FakePost(content="Sure thing.")
    monkeypatch.setattr(c._client, "post", post)

    await c.respond(TaskRegistry(), [], "user_turn")
    post.content = "No problem."
    await c.respond(TaskRegistry(), [], "user_turn")

    assert len(post.payloads) == 2


@pytest.mark.asyncio
async def test_what_was_already_said_is_shown_to_the_model(monkeypatch):
    c = Concierge()
    post = _FakePost(content="I'm onto it.")
    monkeypatch.setattr(c._client, "post", post)
    await c.respond(TaskRegistry(), [], "user_turn")

    post.content = "Sure."
    await c.respond(TaskRegistry(), [], "user_turn")

    assert "I'm onto it." in post.payloads[1]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_a_failed_re_roll_keeps_the_repeated_reply(monkeypatch):
    """Saying the same thing twice is worse than nothing, but not by enough to
    be worth staying silent over."""
    c = Concierge()
    post = _ScriptedPost("On it.", "On it.", "")
    monkeypatch.setattr(c._client, "post", post)

    await c.respond(TaskRegistry(), [], "user_turn")
    assert await c.respond(TaskRegistry(), [], "user_turn") == "On it."


# --- never reading the briefing aloud ---------------------------------------

@pytest.mark.asyncio
async def test_state_is_folded_into_a_single_system_message(monkeypatch):
    """Position was the cause, not wording.

    A separate context message sits immediately before the model's turn, and a
    small model reproduces the most recent instruction-shaped text it can see:
    the person heard "Say you are onto it. Do not answer." spoken aloud, and
    then, after that was reworded to terse state, heard "LOOKING UP: Please
    change the contact." read out as the reply. So the state now goes at the
    very start of the one system message, with the whole conversation between
    it and the point of generation.
    """
    c = Concierge()
    post = _FakePost(content="Let me look.")
    monkeypatch.setattr(c._client, "post", post)

    await c.respond(TaskRegistry(), [{"role": "user", "content": "hi"}],
                    "user_turn", in_flight="change the contact")

    messages = post.payloads[0]["messages"]
    assert sum(m["role"] == "system" for m in messages) == 1
    assert messages[0]["role"] == "system"
    assert "change the contact" in messages[0]["content"]
    assert messages[-1]["role"] == "user"


@pytest.mark.parametrize("echo", [
    "LOOKING UP: Please change the contact.",
    "TASKS: t1 rename contacts",
    "CURRENT STATE (never read aloud):",
    "already said (do not reuse): On it.",
])
def test_a_reply_echoing_our_own_labels_is_never_spoken(echo):
    """Last line of defence. The real fix is positional, but the failure mode
    is speaking internal bookkeeping AT a person -- observed twice live -- and
    silence is always a safe reply where this never was."""
    assert clean_reply(echo) == ""


def test_a_sentence_that_merely_mentions_a_task_still_speaks():
    assert clean_reply("Looking up your calendar now.") == "Looking up your calendar now."
