import pytest


@pytest.fixture(autouse=True)
def isolated_audio_archives(monkeypatch, tmp_path):
    """Offline provider tests must never write input audio into the user's recordings."""
    monkeypatch.setenv("LLM_TOOLKIT_RECORDING_DIR", str(tmp_path / "recordings"))
