import contextlib
import importlib.util
import io
import sys
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
        camera = mock.Mock(is_recording=False)
        microphone = mock.Mock(is_recording=False)
        camera.snapshot.return_value = "snapshot.jpg"

        def toggle(source, value):
            was_recording = source.is_recording
            source.is_recording = not was_recording
            return value if was_recording else None

        camera.toggle_recording.side_effect = lambda: toggle(camera, "recording.mp4")
        microphone.toggle_recording.side_effect = lambda: toggle(microphone, b"recorded audio")
        with (
            mock.patch.object(self.example, "CameraSource", return_value=camera),
            mock.patch.object(self.example, "MicrophoneSource", return_value=microphone),
        ):
            self.run_demo(["/record", "What is happening?", "/snapshot", "/record", "/send", "/quit"])
        self.agent.send_inputs.assert_called_once_with([
            {"type": "text", "value": "What is happening?"},
            {"type": "image", "value": "snapshot.jpg"},
            {"type": "audio", "value": b"recorded audio"},
            {"type": "video", "value": "recording.mp4"},
        ])
        camera.quit.assert_called_once()
        microphone.quit.assert_called_once()

    def test_denied_global_hotkeys_fall_back_to_working_terminal_controls(self):
        self.arguments.global_hotkeys = True
        with mock.patch.object(self.example, "ActionHandler", side_effect=PermissionError("Access denied")):
            output = self.run_demo(["Hello", "/send", "/quit"])
        self.agent.get_response.assert_called_once()
        self.assertIn("Using terminal controls instead", output)

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
