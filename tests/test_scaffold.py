import sys

def test_python_version_is_310():
    assert sys.version_info[:2] == (3, 10), (
        "SoulX-Duplug pins Python 3.10; newer versions lack wheels for its deps"
    )

def test_package_imports():
    import rtvoice  # noqa: F401


# --- the concierge and the reasoner are configured separately ---------------
#
# They ran against one endpoint, one model, one mutex, which made them strictly
# serial. Measured: the concierge answered in 0.26s alone and 2.11s fired
# alongside the reasoner, on every turn -- so the layer whose whole purpose is
# to answer fast while slower work proceeds was the slowest thing in the
# system. Being separately addressable is what lets them actually run at once.

def test_the_concierge_endpoint_and_model_are_both_configurable(monkeypatch, tmp_path):
    """CONCIERGE_MODEL was set by the launch script and read by nothing, so
    the name silently stayed at the default. Harmless against the local shim,
    which ignores the field -- a hard error against Ollama or vLLM."""
    from rtvoice.orchestrator import build_default_orchestrator

    monkeypatch.setenv("CONCIERGE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("CONCIERGE_MODEL", "qwen2.5:3b-instruct-q4_K_M")
    monkeypatch.setenv("DEVICE_STATE", str(_fixture(tmp_path)))
    monkeypatch.setenv("ASR_MODEL", "")

    orch = build_default_orchestrator(session_dir=tmp_path / "s")

    assert orch.concierge.base_url == "http://127.0.0.1:11434/v1"
    assert orch.concierge.model == "qwen2.5:3b-instruct-q4_K_M"


def test_the_concierge_can_point_somewhere_the_reasoner_does_not(monkeypatch, tmp_path):
    from rtvoice.orchestrator import build_default_orchestrator

    monkeypatch.setenv("CONCIERGE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.setenv("REASONER", "llm")
    monkeypatch.setenv("REASONER_URL", "http://127.0.0.1:8001/v1")
    monkeypatch.setenv("DEVICE_STATE", str(_fixture(tmp_path)))
    monkeypatch.setenv("ASR_MODEL", "")

    orch = build_default_orchestrator(session_dir=tmp_path / "s")

    assert orch.concierge.base_url != orch.reasoner.base_url


def _fixture(tmp_path):
    import json
    p = tmp_path / "device_state.json"
    p.write_text(json.dumps({"contacts": [], "messages": [], "calendar": [], "places": []}))
    return p
