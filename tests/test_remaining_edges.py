from unittest.mock import Mock, patch
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from llm_toolkit.agents.agent import Agent
from llm_toolkit.media import camera_source
from llm_toolkit.media.audio_output import AudioOutput
from llm_toolkit.media.microphone_source import _MonitorBuffer
from llm_toolkit.util.action_handler import ActionHandler


def test_constructor_does_not_share_nested_state_between_agents():
    messages = [{"role": "user", "content": [{"type": "text", "text": "original"}]}]
    tools = [{"type": "function", "function": {"name": "work"}}]
    audio = {"voice": "alloy", "format": "wav"}
    with patch.object(Agent, "_update_model_support"):
        first = Agent(messages=messages, tools=tools, audio=audio)
        second = Agent(messages=messages, tools=tools, audio=audio)
    first.messages[0]["content"][0]["text"] = "changed"
    first.tools[0]["function"]["name"] = "changed"
    first.audio["voice"] = "changed"
    assert second.messages == messages
    assert second.messages[0]["content"][0]["text"] == "original"
    assert second.tools[0]["function"]["name"] == "work"
    assert second.audio["voice"] == "alloy"


@pytest.mark.parametrize("bad_field", [{"audio": None}, {"messages": 1}, {"token_usage": None}])
def test_invalid_session_fields_do_not_partially_mutate_agent(bad_field):
    with patch.object(Agent, "_update_model_support"):
        agent = Agent(tool_handlers={"work": lambda: "done"})
    before = agent.get_session()
    handlers = agent.tool_handlers.copy()
    with pytest.raises((TypeError, ValueError)):
        agent.set_session({"model": "changed", **bad_field})
    assert agent.get_session() == before
    assert agent.tool_handlers == handlers


def test_failed_capability_lookup_does_not_replace_existing_session():
    with patch.object(Agent, "_update_model_support"):
        agent = Agent()
    before = agent.get_session()
    with patch.object(Agent, "_update_model_support", side_effect=RuntimeError("lookup failed")), pytest.raises(RuntimeError):
        agent.set_session({"model": "changed"})
    assert agent.get_session() == before


def test_audio_playback_preserves_samples_across_arbitrary_chunk_boundaries():
    written = []
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    stream.write.side_effect = lambda data: written.append(data.copy())
    pcm = np.array([16384, -16384, 100, -200], dtype="<i2").tobytes()
    output = AudioOutput(channels=1, stream_factory=Mock(return_value=stream))
    output.play_pcm([pcm[:1], b"", pcm[1:4], pcm[4:7], pcm[7:]])
    np.testing.assert_allclose(np.concatenate(written)[:, 0], [0.5, -0.5, 100/32768, -200/32768])
    stream.stop.assert_called_once()


def test_pcm_playback_bounds_each_device_write():
    sizes = []
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    stream.write.side_effect = lambda data: sizes.append(len(data))
    AudioOutput(stream_factory=Mock(return_value=stream)).play_pcm([b"\0\0" * 100000])
    assert sum(sizes) == 100000
    assert max(sizes) <= 4096


def test_incomplete_pcm_sample_closes_stream_and_reports_idle():
    states = []
    stream = Mock()
    stream.__enter__ = Mock(return_value=stream)
    stream.__exit__ = Mock(return_value=False)
    with pytest.raises(ValueError, match="incomplete"):
        AudioOutput(stream_factory=Mock(return_value=stream)).play_pcm([b"\0"], on_state=states.append)
    assert states == [True, False]
    stream.__exit__.assert_called_once()


@pytest.mark.parametrize("failed_resource", ["action_handler", "microphone_source", "camera_source"])
def test_action_shutdown_attempts_every_resource_and_joins_listener(failed_resource):
    handler = ActionHandler.__new__(ActionHandler)
    handler.running = True
    handler.action_handler = Mock()
    handler.microphone_source = Mock()
    handler.camera_source = Mock()
    resource = getattr(handler, failed_resource)
    (resource.stop if failed_resource == "action_handler" else resource.quit).side_effect = OSError("cleanup failed")
    with pytest.raises(OSError, match="cleanup failed"):
        handler.quit()
    assert not handler.running
    handler.action_handler.stop.assert_called_once()
    handler.action_handler.join.assert_called_once()
    handler.microphone_source.quit.assert_called_once()
    handler.camera_source.quit.assert_called_once()


@pytest.fixture
def camera(tmp_path):
    cap = Mock()
    cap.isOpened.return_value = True
    cap.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
    with patch.object(camera_source.cv2, "VideoCapture", return_value=cap), patch.object(
        camera_source.threading, "Thread"
    ), patch.object(camera_source.cv2, "destroyWindow"):
        source = camera_source.CameraSource(save_dir=str(tmp_path))
        source._capture_thread.is_alive.return_value = False
        try:
            yield source
        finally:
            source.quit()


def test_failed_camera_open_releases_capture_before_starting_thread():
    cap = Mock()
    cap.isOpened.return_value = False
    cap.read.return_value = (False, None)
    with patch.object(camera_source.cv2, "VideoCapture", return_value=cap), patch.object(
        camera_source.threading, "Thread"
    ) as thread, pytest.raises(RuntimeError, match="open"):
        camera_source.CameraSource()
    cap.release.assert_called_once()
    thread.assert_not_called()


def test_camera_writer_failure_releases_writer(camera):
    camera._capture.get.return_value = 30
    writer = Mock()
    writer.isOpened.return_value = False
    with patch.object(camera_source.cv2, "VideoWriter", return_value=writer), pytest.raises(RuntimeError):
        camera.toggle_recording()
    writer.release.assert_called_once()
    assert camera._video_writer is None
    assert not camera.is_recording


def test_camera_recording_uses_frame_dimensions_and_valid_fps(camera):
    camera._capture.get.return_value = float("nan")
    writer = Mock()
    writer.isOpened.return_value = True
    with patch.object(camera_source.cv2, "VideoWriter", return_value=writer) as factory:
        camera.toggle_recording()
    assert factory.call_args.args[2:] == (30, (32, 24))


def test_camera_quit_resets_recording_and_releases_capture_after_writer_error(camera):
    writer = Mock()
    writer.release.side_effect = RuntimeError("writer failed")
    camera._video_writer = writer
    camera._recording = True
    with pytest.raises(RuntimeError, match="writer failed"):
        camera.quit()
    assert not camera.is_recording
    camera._capture.release.assert_called_once()


def test_camera_operations_after_quit_reject_stale_frames(camera):
    camera.quit()
    with pytest.raises(RuntimeError, match="closed"):
        camera.snapshot()
    with pytest.raises(RuntimeError, match="closed"):
        camera.toggle_recording()


def test_monitor_buffer_randomized_block_sizes_match_bounded_fifo():
    rng = np.random.default_rng(42)
    ring = _MonitorBuffer(37, 1)
    expected = []
    for _ in range(2000):
        count = int(rng.integers(0, 90))
        if rng.random() < 0.6:
            values = rng.integers(-100, 100, count).astype(np.float32)
            ring.write(values[:, None])
            expected = (expected + values.tolist())[-37:]
        else:
            output = np.empty((count, 1), dtype=np.float32)
            ring.read_into(output)
            wanted = expected[:count] + [0] * max(0, count - len(expected))
            expected = expected[count:]
            np.testing.assert_array_equal(output[:, 0], wanted)
        assert ring.size == len(expected)


def test_capture_example_cleans_up_devices_if_later_initialization_fails(monkeypatch, tmp_path):
    examples = Path(__file__).parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location("capture_example", examples / "local_capture.py")
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    camera = Mock()
    arguments = SimpleNamespace(camera=True, microphone=True, save=False, save_dir=str(tmp_path))
    with patch.object(example, "CameraSource", return_value=camera), patch.object(
        example, "MicrophoneSource", side_effect=RuntimeError("microphone unavailable")
    ), pytest.raises(RuntimeError):
        example.main(arguments)
    camera.quit.assert_called_once()
