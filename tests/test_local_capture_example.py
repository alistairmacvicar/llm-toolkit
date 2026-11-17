import contextlib
import importlib.util
import io
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from llm_toolkit import ActionHandler


class LocalCaptureExampleTests(unittest.TestCase):
    def setUp(self):
        examples = Path(__file__).parents[1] / "examples"
        spec = importlib.util.spec_from_file_location("capture_demo", examples / "local_capture.py")
        self.example = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(examples), *sys.path]):
            spec.loader.exec_module(self.example)
        self.agent = mock.Mock(model="test/model")
        self.agent.get_response.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Hello!"))]
        )
        self.arguments = SimpleNamespace(
            model="test/model", camera=False, microphone=False, auto_send=False,
            save=False, save_dir="recordings", global_hotkeys=False,
        )

    def run_demo(self, commands):
        commands = iter(commands)
        output = io.StringIO()
        with (
            mock.patch.object(self.example, "Agent", return_value=self.agent),
            mock.patch("builtins.input", side_effect=lambda _: next(commands)),
            contextlib.redirect_stdout(output),
        ):
            self.example.main(self.arguments)
        return output.getvalue()

    def test_terminal_default_needs_no_listener_and_confirm_requests_one_reply(self):
        def commands():
            yield "Hello"
            self.agent.send_inputs.assert_not_called()
            yield ""
            self.agent.send_inputs.assert_called_once_with([{"type": "text", "value": "Hello"}])
            self.agent.get_response.assert_called_once()
            yield "/quit"

        with (
            mock.patch.dict(sys.modules, {"pynput": None}),
            mock.patch.object(self.example, "ActionHandler") as listener,
            mock.patch.object(self.example, "CameraSource") as camera,
            mock.patch.object(self.example, "MicrophoneSource") as microphone,
        ):
            output = self.run_demo(commands())
        listener.assert_not_called()
        camera.assert_not_called()
        microphone.assert_not_called()
        self.assertIn("Agent: Hello!", output)
        self.assertIn("Microphone: off | Camera: off", output)

    def test_discard_empty_confirm_and_disabled_devices_make_no_requests(self):
        output = self.run_demo([
            "Don't send this", "/discard", "/send", "/audio", "/snapshot", "/record", "/quit",
        ])
        self.agent.get_response.assert_not_called()
        self.assertIn("Nothing staged", output)
        self.assertIn("--microphone", output)
        self.assertIn("--camera", output)

    def test_auto_send_does_not_require_confirmation_or_resend_on_enter(self):
        self.arguments.auto_send = True
        self.run_demo(["Hello", "", "/quit"])
        self.agent.send_inputs.assert_called_once_with([{"type": "text", "value": "Hello"}])
        self.agent.get_response.assert_called_once()

    def test_recorded_sources_and_text_are_confirmed_as_one_batch_and_closed(self):
        self.arguments.camera = self.arguments.microphone = True
        camera = mock.Mock(is_recording=False, recording_started_at=1.0)
        microphone = mock.Mock(is_recording=False, recording_started_at=1.0)
        camera.snapshot.return_value = "snapshot.jpg"

        def toggle(source, value):
            was_recording = source.is_recording
            source.is_recording = not was_recording
            return value if was_recording else None

        camera.toggle_recording.side_effect = lambda: toggle(camera, "recording.mp4")
        microphone.toggle_recording.side_effect = lambda: toggle(microphone, "recorded-audio.wav")
        with (
            mock.patch.object(self.example, "CameraSource", return_value=camera),
            mock.patch.object(self.example, "MicrophoneSource", return_value=microphone),
            mock.patch("llm_toolkit.media.recording.mux_recordings", return_value="recording-with-audio.mp4") as mux,
        ):
            self.run_demo(["/record", "What is happening?", "/snapshot", "/record", "/send", "/quit"])
        self.agent.send_inputs.assert_called_once_with([
            {"type": "text", "value": "What is happening?"},
            {"type": "image", "value": "snapshot.jpg"},
            {"type": "audio", "value": "recorded-audio.wav"},
            {"type": "video", "value": "recording-with-audio.mp4"},
        ])
        mux.assert_called_once_with("recording.mp4", "recorded-audio.wav", audio_offset=0.0)
        camera.quit.assert_called_once()
        microphone.quit.assert_called_once()

    def test_denied_global_hotkeys_fall_back_to_working_terminal_controls(self):
        self.arguments.global_hotkeys = True
        with mock.patch.object(self.example, "ActionHandler", side_effect=PermissionError("Access denied")):
            output = self.run_demo(["Hello", "/send", "/quit"])
        self.agent.get_response.assert_called_once()
        self.assertIn("Using terminal controls instead", output)

    def test_camera_only_recording_explains_how_to_include_audio(self):
        camera = mock.Mock(is_recording=False)
        output = io.StringIO()
        with (
            mock.patch("builtins.input", side_effect=["/record", "/quit"]),
            contextlib.redirect_stdout(output),
        ):
            self.example.run_terminal(self.agent, camera, None, auto_send=False)
        self.assertIn("Video only", output.getvalue())
        self.assertIn("--microphone", output.getvalue())

    def test_preview_keeps_pumping_on_main_thread_during_blocking_work(self):
        release = threading.Event()
        polled_threads = []
        camera = mock.Mock()
        def poll():
            polled_threads.append(threading.current_thread())
            if len(polled_threads) >= 2:
                release.set()
        camera.poll_preview.side_effect = poll
        result = self.example.with_preview(camera, lambda: release.wait(timeout=1))
        self.assertTrue(result)
        self.assertGreaterEqual(len(polled_threads), 2)
        self.assertTrue(all(thread is threading.main_thread() for thread in polled_threads))

    def test_empty_microphone_capture_is_reported_and_not_sent(self):
        microphone = mock.Mock(is_recording=True)
        microphone.toggle_recording.return_value = None
        output = io.StringIO()
        with (
            mock.patch("builtins.input", side_effect=["/record", "/send", "/quit"]),
            contextlib.redirect_stdout(output),
        ):
            self.example.run_terminal(self.agent, None, microphone, auto_send=False)
        self.assertIn("No audio was captured", output.getvalue())
        self.agent.send_inputs.assert_not_called()

    def test_eof_and_interrupt_close_capture_devices(self):
        self.arguments.microphone = True
        for ending in (EOFError, KeyboardInterrupt):
            with self.subTest(ending=ending):
                microphone = mock.Mock()
                with (
                    mock.patch.object(self.example, "Agent", return_value=self.agent),
                    mock.patch.object(self.example, "MicrophoneSource", return_value=microphone),
                    mock.patch("builtins.input", side_effect=ending),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.example.main(self.arguments)
                microphone.quit.assert_called_once()
        self.agent.get_response.assert_not_called()

    def test_untrusted_listener_is_stopped_before_permission_error(self):
        listener = mock.Mock(IS_TRUSTED=False)
        keyboard = SimpleNamespace(
            KeyCode=SimpleNamespace(from_char=lambda char: char),
            Key=SimpleNamespace(enter="enter", esc="esc"),
            Listener=mock.Mock(return_value=listener),
        )
        with mock.patch.dict(sys.modules, {"pynput": SimpleNamespace(keyboard=keyboard)}):
            with self.assertRaisesRegex(PermissionError, "Accessibility"):
                ActionHandler()
        listener.start.assert_called_once()
        listener.wait.assert_called_once()
        listener.stop.assert_called_once()
        listener.join.assert_called_once()


if __name__ == "__main__":
    unittest.main()
