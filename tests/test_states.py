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
