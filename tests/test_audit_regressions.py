"""Offline regressions for session, media, and connection boundary conditions."""

import asyncio
import base64
import io
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import soundfile as sf

from llm_toolkit.agents.agent import Agent
from llm_toolkit.agents.director import Director
from llm_toolkit.agents.realtime import RealtimeSession
from llm_toolkit.agents.realtime_ensemble import RealtimeEnsemble
from llm_toolkit.media import file_source
from llm_toolkit.media.audio_output import AudioOutput
from llm_toolkit.util.action_handler import ActionHandler


class AgentBoundaryTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.object(Agent, "_update_model_support")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_long_base64_audio_does_not_get_treated_as_a_filename(self):
        payload = b"audio" * 1000
        encoded, audio_file, fmt = Agent()._prepare_audio(base64.b64encode(payload).decode())
        self.assertEqual(base64.b64decode(encoded), payload)
        self.assertEqual(audio_file.read(), payload)
        self.assertEqual(fmt, "wav")

    def test_session_restoration_does_not_share_nested_state(self):
        original = Agent(messages=[{"role": "user", "content": [{"type": "text", "text": "hello"}]}])
        session = original.get_session()
        first, second = Agent(), Agent()
        first.set_session(session)
        second.set_session(session)
        first.messages[0]["content"][0]["text"] = "changed"
        self.assertEqual(second.messages[0]["content"][0]["text"], "hello")
        self.assertEqual(session["messages"][0]["content"][0]["text"], "hello")

    def test_partial_session_update_preserves_generation_options(self):
        agent = Agent(completion_options={"temperature": 0.2})
        agent.set_session({"system_prompt": "New prompt"})
        self.assertEqual(agent.completion_options, {"temperature": 0.2})

    def test_invalid_session_options_do_not_partially_update_agent(self):
        agent = Agent(model="original")
        with self.assertRaises(ValueError):
            agent.set_session({"model": "changed", "completion_options": {"stream": True}})
        self.assertEqual(agent.model, "original")

    def test_mixed_manual_and_automatic_tools_leave_batch_for_caller(self):
        handler = Mock(return_value="done")
        agent = Agent(tool_handlers={"local": handler})
        message = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "function": {"name": "local", "arguments": "{}"}},
            {"id": "2", "function": {"name": "manual", "arguments": "{}"}},
        ]}
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)
        with patch("llm_toolkit.agents.agent.completion", return_value=response) as complete:
            self.assertIs(agent.get_response(), response)
        complete.assert_called_once()
        handler.assert_not_called()
        self.assertEqual(agent.messages, [message])


class VideoSamplingTests(unittest.TestCase):
    def capture(self, frames, count=None):
        cap = Mock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: 30 if prop == file_source.cv2.CAP_PROP_FPS else (len(frames) if count is None else count)
        cap.read.side_effect = [(True, frame) for frame in frames] + [(False, None)]
        return cap

    def test_short_clip_includes_first_frame(self):
        cap = self.capture(["first"])
        with patch.object(file_source.cv2, "VideoCapture", return_value=cap), patch.object(file_source.cv2, "imencode", return_value=(True, "jpeg")):
            self.assertEqual(file_source._sample_frames("short.mp4", 2), ["jpeg"])
        cap.release.assert_called_once()

    def test_unknown_frame_count_reads_until_eof(self):
        cap = self.capture(["first", "second"], count=0)
        with patch.object(file_source.cv2, "VideoCapture", return_value=cap), patch.object(file_source.cv2, "imencode", side_effect=lambda _, frame: (True, frame)):
            self.assertEqual(file_source._sample_frames("unknown.mp4", 30), ["first", "second"])
        cap.release.assert_called_once()

    def test_encoding_failure_releases_capture(self):
        cap = self.capture(["first"])
        with patch.object(file_source.cv2, "VideoCapture", return_value=cap), patch.object(file_source.cv2, "imencode", return_value=(False, None)):
            with self.assertRaises(ValueError):
                file_source._sample_frames("broken.mp4", 30)
        cap.release.assert_called_once()


class AudioBoundaryTests(unittest.TestCase):
    def test_stereo_realtime_audio_is_downmixed_to_mono(self):
        session = RealtimeSession("ws://localhost", "test")
        encoded = session._prepare_audio(np.array([[0.5, -0.5], [1.0, 0.0]]))
        samples = np.frombuffer(base64.b64decode(encoded), dtype="<i2")
        np.testing.assert_array_equal(samples, [0, 16383])

    def test_pcm16_stereo_downmix_preserves_amplitude(self):
        session = RealtimeSession("ws://localhost", "test")
        encoded = session._prepare_audio(np.array([[2000, 4000], [-2000, 2000]], dtype=np.int16))
        np.testing.assert_array_equal(np.frombuffer(base64.b64decode(encoded), dtype="<i2"), [3000, 0])

    def test_playback_open_failure_reports_idle(self):
        events = []
        output = AudioOutput(stream_factory=Mock(side_effect=OSError("No device")))
        with self.assertRaises(OSError):
            output.play_pcm([b"\0\0"], on_state=events.append)
        self.assertEqual(events, [False])

    def test_real_wav_decode_works_alongside_agent_tests(self):
        wav = io.BytesIO()
        sf.write(wav, np.zeros((24, 2)), 24000, format="WAV")
        output = AudioOutput()
        with patch.object(output, "play_pcm") as play:
            output.play_wav(wav.getvalue())
        self.assertEqual(len(play.call_args.args[0][0]), 48)


class ConnectionBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_concurrent_connects_share_one_socket(self):
        socket = SimpleNamespace(close=AsyncMock())
        async def make_socket(*args, **kwargs):
            await asyncio.sleep(0)
            return socket
        factory = AsyncMock(side_effect=make_socket)
        session = RealtimeSession("ws://localhost", "test", websocket_factory=factory)
        await asyncio.gather(session.connect(), session.connect())
        factory.assert_awaited_once()
        await session.close()

    async def test_failed_close_clears_connected_state(self):
        socket = SimpleNamespace(close=AsyncMock(side_effect=OSError("Disconnected")))
        session = RealtimeSession("ws://localhost", "test", websocket_factory=AsyncMock(return_value=socket))
        await session.connect()
        with self.assertRaises(OSError):
            await session.close()
        self.assertFalse(session.connected)

    async def test_overlapping_recording_starts_are_rejected(self):
        director = Director({"Ada": Mock()}, director_agent=Mock())
        ensemble = RealtimeEnsemble(director, model="test")
        connecting, release = asyncio.Event(), asyncio.Event()
        async def connect():
            connecting.set()
            await release.wait()
            ensemble.sessions[director.root] = SimpleNamespace(send_event=AsyncMock(), close=AsyncMock())
        ensemble.connect = connect
        first = asyncio.create_task(ensemble.begin())
        await connecting.wait()
        try:
            with self.assertRaises(RuntimeError):
                await asyncio.wait_for(ensemble.begin(), 0.05)
        finally:
            release.set()
            await first
            await ensemble.close()

    async def test_invalid_ensemble_is_rejected_before_opening_sockets(self):
        factory = Mock()
        ensemble = RealtimeEnsemble(Director({"Ada": Mock()}), model="test", session_factory=factory)
        with self.assertRaisesRegex(ValueError, "coordinator"):
            await ensemble.connect()
        factory.assert_not_called()


class KeyboardBoundaryTests(unittest.TestCase):
    def setUp(self):
        keyboard = SimpleNamespace(
            KeyCode=SimpleNamespace(from_char=lambda char: char),
            Key=SimpleNamespace(enter="enter", esc="esc"),
            Listener=Mock(),
        )
        self.microphone = Mock()
        with patch.dict(sys.modules, {"pynput": SimpleNamespace(keyboard=keyboard)}):
            self.actions = ActionHandler(microphone_source=self.microphone)
        self.addCleanup(self.actions.stop)

    def test_key_repeat_does_not_toggle_monitor_repeatedly(self):
        self.actions._on_press("m")
        self.actions._on_press("m")
        self.microphone.toggle_monitor.assert_called_once()
        self.actions._on_release("m")
        self.actions._on_press("m")
        self.assertEqual(self.microphone.toggle_monitor.call_count, 2)

    def test_terminal_typing_does_not_trigger_global_shortcuts(self):
        with patch("llm_toolkit.util.action_handler.threading.Thread") as thread:
            self.actions._on_press("t")
            self.actions._on_release("t")
            for key in ("t", "q", "a", "m", "enter"):
                self.actions._on_press(key)
                self.actions._on_release(key)
        thread.assert_called_once()
        self.assertTrue(self.actions.running)
        self.microphone.start_recording.assert_not_called()
        self.microphone.stop_recording.assert_not_called()
        self.microphone.toggle_monitor.assert_not_called()
        with patch("builtins.input", return_value="question"):
            self.actions._read_terminal_message()
        self.assertFalse(self.actions._reading_terminal)
        self.assertEqual(self.actions.get_pending_inputs(), [{"type": "text", "value": "question"}])

    def test_terminal_eof_stops_listener_and_emits_quit(self):
        self.actions._reading_terminal = True
        with patch("builtins.input", side_effect=EOFError):
            self.actions._read_terminal_message()
        self.assertFalse(self.actions.running)
        self.assertFalse(self.actions._reading_terminal)
        self.assertEqual(self.actions.get_event(block=False)["type"], "quit")

    def test_push_to_talk_release_still_stops_capture_after_terminal_opens(self):
        self.microphone.stop_recording.return_value = b"audio"
        self.actions._on_press("a")
        with patch("llm_toolkit.util.action_handler.threading.Thread"):
            self.actions._on_press("t")
        self.actions._on_release("a")
        self.microphone.stop_recording.assert_called_once()
        self.assertEqual(self.actions.get_pending_inputs(), [{"type": "audio", "value": b"audio"}])
