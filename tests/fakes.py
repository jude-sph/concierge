"""Shared test doubles. Imported as `from fakes import ...` — pytest's default
prepend import mode puts tests/ on sys.path, and there is no tests/__init__.py.
"""
class FakeConcierge:
    """Records what it was asked; returns a fixed spoken line.

    The concierge returns plain text now, not a typed speech act -- it no
    longer reports results, so the JSON envelope and citation validation it
    used to need went away with that job.
    """

    def __init__(self):
        self.calls = []
        self.in_flight = []
        self.violations = 0

    async def respond(self, registry, history, trigger, in_flight=None):
        self.calls.append((registry.fact_block(), trigger))
        self.in_flight.append(in_flight)
        return "on it"


class FakeVoice:
    def __init__(self):
        self.spoken = []
        self.stops = 0

    async def speak(self, text, utterance_id):
        self.spoken.append(text)

    async def stop(self):
        self.stops += 1
