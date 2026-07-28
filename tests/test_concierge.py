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

from rtvoice.concierge import MAX_REPLY_CHARS, Concierge, clean_reply
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
