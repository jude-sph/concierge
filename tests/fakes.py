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
