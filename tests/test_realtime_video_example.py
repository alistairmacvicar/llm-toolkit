import asyncio
import base64
import importlib.util
import json
import queue
import sys
import threading
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pytest

from llm_toolkit import RealtimeSession


@pytest.fixture
def example():
    examples = Path(__file__).resolve().parents[1] / "examples"
    spec = importlib.util.spec_from_file_location("realtime_video_example", examples / "realtime_video.py")
    module = importlib.util.module_from_spec(spec)
    with patch.object(sys, "path", [str(examples), *sys.path]):
        spec.loader.exec_module(module)
    return module


def test_defaults_device_names_and_invalid_arguments(example):
    args = example.parse_args([])
    assert args.model == "openai/gpt-realtime"
    assert args.frame_interval == 1
    assert not hasattr(args, "port")
    args = example.parse_args(["--input-device", "MacBook Air Microphone", "--output-device", "3"])
    assert args.input_device == "MacBook Air Microphone" and args.output_device == 3
    for option, value in [("--frame-interval", "nan"), ("--frame-interval", "0"),
                          ("--timeout", "inf"), ("--camera", "-1"), ("--model", "gemini/test")]:
        with pytest.raises(SystemExit):
            example.parse_args([option, value])


def test_device_listing_does_not_capture_or_connect(example):
    with patch.object(example.sd, "query_devices", return_value="Mac microphone"), patch.object(
        example, "LiveAudio"
    ) as audio, patch.object(example, "RealtimeSession") as session:
        assert example.main(example.parse_args(["--list-devices"])) == 0
    audio.assert_not_called()
    session.assert_not_called()


def test_vad_setup_enables_streamed_voice_and_interruption(example):
    session = SimpleNamespace(connect=AsyncMock(), update_session=AsyncMock(), receive_event=AsyncMock(
        side_effect=[{"type": "session.created"}, {"type": "session.updated"}]))
    asyncio.run(example.configure_session(session, example.parse_args([])))
    settings = session.update_session.call_args.kwargs
    assert settings["output_modalities"] == ["audio"]
    assert settings["audio"]["input"]["turn_detection"]["interrupt_response"]
    assert settings["audio"]["input"]["turn_detection"]["create_response"]
    assert settings["audio"]["output"]["format"] == {"type": "audio/pcm", "rate": 24000}


def test_recordings_stream_to_disk_preserve_transcripts_and_mux_on_exit(example, tmp_path, capsys):
    store = example.Recordings(tmp_path, "test-model")
    store.capture(b"\x01\x00\x02\x00", 10.25)
    store.reply(("reply", 0), b"\x03\x00")
    store.transcript("user", "Can you see this?", "user-1")
    store.transcript("user", "Can you see this?", "user-1")
    with wave.open(str(store.microphone.path)) as file:
        assert file.readframes(2) == b"\x01\x00\x02\x00"
    camera = SimpleNamespace(last_recording_path="camera.mp4", recording_started_at=10)
    with patch.object(example, "mux_recordings", return_value="combined.mp4") as mux:
        store.finish(camera)
    assert mux.call_args.kwargs["audio_offset"] == 0.25
    assert json.loads((store.directory / "session.json").read_text())["status"] == "finished"
    assert len((store.directory / "transcripts.jsonl").read_text().splitlines()) == 1
    assert "You: Can you see this?" in capsys.readouterr().out


def test_interruption_truncates_at_played_time_and_ignores_late_audio(example):
    delta = {"type": "response.output_audio.delta", "item_id": "item-1", "response_id": "response-1",
             "delta": base64.b64encode(b"\x01\x00").decode()}
    events = [{"type": "response.created", "response": {"id": "response-1"}}, delta,
              {"type": "input_audio_buffer.speech_started"}, delta,
              {"type": "response.created", "response": {"id": "response-2"}},
              {**delta, "item_id": "item-2", "response_id": "response-2"}]

    async def receive():
        for event in events:
            yield event

    session = SimpleNamespace(events=receive, send_event=AsyncMock())
    audio = Mock()
    audio.interrupt.return_value = (("item-1", 0), 120)
    recordings = Mock()
    with pytest.raises(RuntimeError, match="connection closed"):
        asyncio.run(example.receive_events(session, audio, recordings))
    assert audio.play.call_count == 2
    session.send_event.assert_awaited_once_with({"type": "conversation.item.truncate", "item_id": "item-1",
                                               "content_index": 0, "audio_end_ms": 120})


def test_frames_are_bounded_and_do_not_trigger_spoken_replies(example):
    sent = []
    sleeps = 0

    async def send(event):
        sent.append(event)

    async def sleep(interval):
        nonlocal sleeps
        sleeps += 1
        if sleeps == 33:
            raise asyncio.CancelledError

    camera = SimpleNamespace(get_frame=lambda: np.zeros((720, 1280, 3), dtype=np.uint8))
    with patch.object(example.asyncio, "sleep", side_effect=sleep), pytest.raises(asyncio.CancelledError):
        asyncio.run(example.send_camera(SimpleNamespace(send_event=send), camera, 1))
    creates = [event for event in sent if event["type"] == "conversation.item.create"]
    assert len(creates) == 33
    assert len([event for event in sent if event["type"] == "conversation.item.delete"]) == 3
    assert not any(event["type"] == "response.create" for event in sent)
    encoded = creates[0]["item"]["content"][0]["image_url"].split(",")[1]
    image = example.cv2.imdecode(np.frombuffer(base64.b64decode(encoded), dtype=np.uint8), 1)
    assert image.shape[:2] == (360, 640)


@pytest.mark.parametrize("outcome", ["quit", "connect_failure", "cancel_startup", "disk_full", "cancel_running"])
def test_native_lifecycle_releases_devices_and_finalizes_audio(example, tmp_path, outcome):
    pcm = b"\x01\x00" * 480
    capture = SimpleNamespace(inputs=queue.Queue(), input_info={"name": "Mac microphone"},
                              output_info={"name": "AirPods"}, check=Mock(), play=Mock(), interrupt=Mock(return_value=None))
    capture.start = Mock(side_effect=lambda: capture.inputs.put((pcm, 10.25)))
    capture.close = Mock()
    camera = SimpleNamespace(preview_visible=True, last_recording_path=None, recording_started_at=10,
                             get_frame=lambda: np.zeros((24, 32, 3), dtype=np.uint8), toggle_capture=Mock(), quit=Mock())
    preview_threads = []
    camera.poll_preview = lambda: (preview_threads.append(threading.current_thread()) or -1)

    def begin_video():
        camera.last_recording_path = str(tmp_path / "camera.mp4")
    camera.toggle_recording = Mock(side_effect=begin_video)
    sessions = []

    async def exercise():
        sent_audio = asyncio.Event()

        class Session(RealtimeSession):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                sessions.append(self)
                self.socket = SimpleNamespace(send=AsyncMock(), close=AsyncMock(), close_code=None)

            async def connect(self):
                if outcome == "connect_failure":
                    raise RuntimeError("No provider connection")
                if outcome == "cancel_startup":
                    await asyncio.Event().wait()
                self._websocket = self.socket

            async def receive_event(self):
                return {"type": "session.updated"}

            async def append_audio(self, audio):
                await super().append_audio(audio)
                sent_audio.set()

            async def events(self):
                yield {"type": "conversation.item.input_audio_transcription.completed", "item_id": "user-1", "transcript": "Hello"}
                await asyncio.Event().wait()

        async def terminal(stop):
            if outcome == "cancel_startup":
                await asyncio.sleep(0.02)
            else:
                await sent_audio.wait()
                await asyncio.sleep(0.02)
            if outcome != "cancel_running":
                stop.set()
            else:
                await asyncio.Event().wait()

        with patch.object(example, "RealtimeSession", Session), patch.object(example, "terminal_loop", side_effect=terminal):
            task = asyncio.create_task(example.run(example.parse_args(["--save-dir", str(tmp_path)])))
            if outcome == "cancel_running":
                await asyncio.wait_for(sent_audio.wait(), timeout=1)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                return
            return await asyncio.wait_for(task, timeout=2)

    with patch.object(example, "LiveAudio", return_value=capture), patch.object(example, "CameraSource", return_value=camera), patch.object(
        example, "mux_recordings", return_value="combined.mp4"
    ) as mux:
        if outcome == "disk_full":
            with patch.object(example.Recordings, "capture", side_effect=OSError("Disk full")):
                result = asyncio.run(exercise())
        else:
            result = asyncio.run(exercise())
    capture.close.assert_called_once()
    camera.quit.assert_called_once()
    assert preview_threads and all(thread is threading.main_thread() for thread in preview_threads)
    directory = next(tmp_path.glob("live_*"))
    metadata = json.loads((directory / "session.json").read_text())
    if outcome in {"connect_failure", "disk_full"}:
        assert result == 1 and metadata["status"] == "interrupted"
    elif outcome == "cancel_startup":
        capture.start.assert_not_called()
        camera.toggle_recording.assert_not_called()
    else:
        assert result in (0, None)
        mux.assert_called_once()
        sessions[0].socket.close.assert_awaited_once()
        with wave.open(sessions[0].last_input_audio_path) as file:
            assert file.readframes(480) == pcm
        with wave.open(str(next(directory.glob("microphone_*.wav")))) as file:
            assert file.readframes(480) == pcm
