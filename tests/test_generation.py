from rag_service.generation import openai_timeout_seconds


def test_openai_timeout_seconds_defaults_to_30(monkeypatch):
    monkeypatch.delenv("OPENAI_TIMEOUT_SECONDS", raising=False)

    assert openai_timeout_seconds() == 30.0


def test_openai_timeout_seconds_uses_valid_env(monkeypatch):
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")

    assert openai_timeout_seconds() == 12.5


def test_openai_timeout_seconds_falls_back_for_invalid_env(monkeypatch):
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "slow")

    assert openai_timeout_seconds() == 30.0


def test_openai_timeout_seconds_has_minimum(monkeypatch):
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "0")

    assert openai_timeout_seconds() == 1.0
