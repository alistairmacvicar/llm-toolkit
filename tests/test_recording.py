import asyncio
import base64
import io
import logging
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import cv2
import imageio_ffmpeg
import numpy as np
import pytest
import soundfile as sf

from llm_toolkit import Agent, RealtimeSession
from llm_toolkit.media.recording import mux_recordings, toggle_recording_sources


def test_every_agent_audio_input_is_archived_before_it_is_queued(tmp_path):
    samples = np.linspace(-0.5, 0.5, 240, dtype=np.float32)
    with patch.object(Agent, "_update_model_support"), patch.object(Agent, "_supports_audio_input", return_value=True):
        agent = Agent(recording_dir=tmp_path)
        agent.send_audio(samples)
        path = Path(agent.last_input_audio_path)
        assert path.is_file()
        queued = agent.messages[0]["content"][0]["input_audio"]
        assert path.read_bytes() == base64.b64decode(queued["data"])
        actual, rate = sf.read(path)
        assert rate == 24000
        np.testing.assert_allclose(actual, samples, atol=1 / 32768)


@pytest.mark.parametrize("input_kind", ["bytes", "path", "file", "base64"])
def test_supplied_audio_is_archived_without_changing_its_encoding(tmp_path, input_kind):
    data = b"encoded audio for a provider"
    path = tmp_path / "source.mp3"
    path.write_bytes(data)
    value = {"bytes": data, "path": path, "file": io.BytesIO(data),
             "base64": base64.b64encode(data).decode()}[input_kind]
    with patch.object(Agent, "_update_model_support"), patch.object(Agent, "_supports_audio_input", return_value=True):
        agent = Agent(recording_dir=tmp_path / "sent")
        agent.send_audio(value, format="mp3")
    saved = Path(agent.last_input_audio_path)
    assert saved.suffix == ".mp3"
    assert saved.read_bytes() == data
    assert saved != path


def test_archive_failure_prevents_unrecorded_audio_from_being_sent():
    with patch.object(Agent, "_update_model_support"), patch(
        "llm_toolkit.media.recording.save_input_audio", side_effect=OSError("Disk full")
    ), patch("llm_toolkit.agents.agent.transcription") as transcribe:
        agent = Agent()
        with pytest.raises(OSError, match="Disk full"):
            agent.send_audio(b"audio")
        assert agent.messages == []
        transcribe.assert_not_called()


def test_raw_messages_and_restored_audio_history_are_archived_before_completion(tmp_path):
    content = [{"type": "input_audio", "input_audio": {
        "data": base64.b64encode(b"record this raw input").decode(), "format": "mp3",
    }}]
    with patch.object(Agent, "_update_model_support"), patch.object(Agent, "_supports_audio_input", return_value=True):
        agent = Agent(recording_dir=tmp_path / "raw")
        agent.send_message(content)
        assert Path(agent.last_input_audio_path).read_bytes() == b"record this raw input"
        restored = Agent(messages=agent.get_messages(), recording_dir=tmp_path / "restored")

        def complete(**kwargs):
            assert Path(restored.last_input_audio_path).read_bytes() == b"record this raw input"
            return SimpleNamespace(choices=[SimpleNamespace(message={"role": "assistant", "content": "Hello"})])

        with patch("llm_toolkit.agents.agent.completion", side_effect=complete), patch.object(restored, "_track_token_usage"):
            restored.get_response()
            restored.get_response()
        assert len(list((tmp_path / "restored").iterdir())) == 1


def test_each_new_send_gets_a_recording_without_duplicating_its_history(tmp_path):
    with patch.object(Agent, "_update_model_support"), patch.object(Agent, "_supports_audio_input", return_value=True):
        agent = Agent(recording_dir=tmp_path)
        agent.send_audio(b"same utterance")
        agent.send_audio(b"same utterance")
    assert len(list(tmp_path.iterdir())) == 2


def test_transcript_is_logged_and_audio_is_saved_for_explicit_transcription(tmp_path, caplog):
    with patch.object(Agent, "_update_model_support"), patch(
        "llm_toolkit.agents.agent.transcription", return_value={"text": "Hello from the microphone"}
    ), caplog.at_level(logging.INFO, logger="llm_toolkit"):
        agent = Agent(recording_dir=tmp_path)
        assert agent.transcribe_audio(b"audio") == "Hello from the microphone"
    assert "User audio transcript: Hello from the microphone" in caplog.text
    assert Path(agent.last_input_audio_path).read_bytes() == b"audio"


def test_realtime_chunks_and_raw_audio_events_are_saved_in_one_readable_wav(tmp_path, caplog):
    async def run():
        session = RealtimeSession(recording_dir=tmp_path)
        socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock())
        session._websocket = socket
        await session.append_audio(b"\x01\x00\x02\x00")
        with wave.open(session.last_input_audio_path) as file:
            assert file.readframes(2) == b"\x01\x00\x02\x00"
        await session.send_event({"type": "input_audio_buffer.append", "audio": base64.b64encode(b"\x03\x00").decode()})
        session._log_transcript({"type": "conversation.item.input_audio_transcription.completed", "transcript": "Hello"})
        session._log_transcript({"type": "response.output_audio_transcript.done", "transcript": "Hi there"})
        # The WAV is readable before close, as well as after a transport error.
        with wave.open(session.last_input_audio_path) as file:
            assert file.getnframes() == 3
        socket.close.side_effect = OSError("Disconnected")
        with pytest.raises(OSError):
            await session.close()
        return session.last_input_audio_path

    with caplog.at_level(logging.INFO, logger="llm_toolkit"):
        path = asyncio.run(run())
    with wave.open(path) as file:
        assert (file.getnchannels(), file.getsampwidth(), file.getframerate()) == (1, 2, 24000)
        assert file.readframes(3) == b"\x01\x00\x02\x00\x03\x00"
    assert "User audio transcript: Hello" in caplog.text
    assert "Assistant audio transcript: Hi there" in caplog.text


def test_combined_recording_contains_playable_video_and_offset_audio(tmp_path):
    video, audio = tmp_path / "camera.mp4", tmp_path / "microphone.wav"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (32, 24))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()
    samples = 0.25 * np.sin(2 * np.pi * 440 * np.arange(48000) / 24000)
    sf.write(audio, samples, 24000)
    combined = mux_recordings(video, audio, audio_offset=0.2)
    capture = cv2.VideoCapture(combined)
    try:
        assert capture.read()[0]
    finally:
        capture.release()
    result = subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", combined,
        "-map", "0:a:0", "-af", "aresample=async=1:first_pts=0", "-ac", "1", "-ar", "24000",
        "-f", "s16le", "pipe:1",
    ], capture_output=True, check=True, timeout=20)
    decoded = np.frombuffer(result.stdout, dtype="<i2") / 32768
    assert len(decoded) >= 24000 * 1.8
    assert np.abs(decoded[:2400]).mean() < 0.01
    assert np.abs(decoded[12000:24000]).mean() > 0.1
    assert video.is_file() and audio.is_file()


def test_reconnecting_after_remote_close_starts_a_new_audio_archive(tmp_path):
    async def run():
        socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock(), close_code=None)
        factory = AsyncMock(return_value=socket)
        session = RealtimeSession(url="ws://localhost", websocket_factory=factory, recording_dir=tmp_path)
        await session.connect()
        await session.append_audio(b"\x01\x00")
        first = session.last_input_audio_path
        socket.close_code = 1000
        factory.return_value = SimpleNamespace(send=AsyncMock(), close=AsyncMock(), close_code=None)
        await session.connect()
        await session.append_audio(b"\x02\x00")
        second = session.last_input_audio_path
        await session.close()
        assert first != second
        for path, expected in ((first, b"\x01\x00"), (second, b"\x02\x00")):
            with wave.open(path) as file:
                assert file.readframes(2) == expected
    asyncio.run(run())


def test_record_command_does_not_invert_sources_that_are_out_of_sync():
    microphone = Mock(is_recording=True)
    microphone.toggle_recording.return_value = "audio.wav"
    camera = Mock(is_recording=False)
    completed = toggle_recording_sources([("audio", microphone), ("video", camera)])
    assert completed == [{"type": "audio", "value": "audio.wav"}]
    camera.toggle_recording.assert_not_called()
