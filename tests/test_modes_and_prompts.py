import base64
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import litellm


def response(text, *, audio=False):
    message = {"role": "assistant", "content": text}
    if audio:
        message.update(content=None, audio={
            "id": "audio-test", "expires_at": 2000000000,
            "data": base64.b64encode(b"test wav bytes").decode(), "transcript": text,
        })
    return litellm.ModelResponse(choices=[{"message": message}])


class ModeExampleTests(unittest.TestCase):
    def setUp(self):
        examples = Path(__file__).parents[1] / "examples"
        spec = importlib.util.spec_from_file_location("modes_demo", examples / "modes_and_prompts.py")
        self.example = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(examples), *sys.path]):
            spec.loader.exec_module(self.example)
        for name in ("supports_audio_input", "supports_audio_output"):
            patcher = mock.patch.object(litellm, name, side_effect=lambda model: model == "test/voice")
            patcher.start()
            self.addCleanup(patcher.stop)
        vision = mock.patch.object(litellm, "supports_vision", return_value=False)
        vision.start()
        self.addCleanup(vision.stop)
        completion = mock.patch("llm_toolkit.agents.agent.completion",
                                side_effect=AssertionError("Unexpected provider call"))
        self.completion = completion.start()
        self.addCleanup(completion.stop)

    def test_modes_switch_models_preserve_transcripts_and_play_saved_audio(self):
        self.completion.side_effect = [
            response("I'll remember tea."), response("You prefer tea.", audio=True),
            response("You still prefer tea."),
        ]
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("builtins.input", side_effect=[
                "I prefer tea.", "/mode voice", "What do I prefer?",
                "/mode assistant", "And now?", "/quit",
            ]),
            mock.patch.object(self.example, "AudioOutput") as playback,
            contextlib.redirect_stdout(output),
        ):
            self.example.main("test/text", "test/voice", output_dir=directory)
            files = list(Path(directory).glob("voice-*.wav"))
            self.assertEqual(len(files), 1)
            self.assertEqual(files[0].read_bytes(), b"test wav bytes")
            playback.return_value.play_wav.assert_called_once_with(b"test wav bytes")
        calls = self.completion.call_args_list
        self.assertEqual([call.kwargs["model"] for call in calls],
                         ["test/text", "test/voice", "test/text"])
        self.assertEqual(calls[1].kwargs["modalities"], ["text", "audio"])
        self.assertNotIn("modalities", calls[2].kwargs)
        self.assertIn({"role": "assistant", "content": "You prefer tea."}, calls[2].kwargs["messages"])
        self.assertTrue(all("audio" not in message for message in calls[2].kwargs["messages"]))
        self.assertIn("Mode: voice | Model: test/voice", output.getvalue())

    def test_unsupported_voice_model_leaves_text_mode_usable(self):
        self.completion.return_value = response("Hello")
        self.completion.side_effect = None
        output = io.StringIO()
        with mock.patch("builtins.input", side_effect=["/mode voice", "Hello", "/quit"]), \
                contextlib.redirect_stdout(output):
            self.example.main("test/text", "test/text", playback=False)
        self.assertIn("Unable to change mode:", output.getvalue())
        self.assertIn("Agent: Hello", output.getvalue())
        self.assertEqual(self.completion.call_args.kwargs["model"], "test/text")
        self.assertNotIn("modalities", self.completion.call_args.kwargs)

    def test_audio_model_text_only_error_is_local_and_user_can_switch_to_voice(self):
        self.completion.side_effect = [response("Hello", audio=True)]
        output = io.StringIO()
        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch("builtins.input", side_effect=[
                "This text-only request should fail", "/mode voice", "Hello", "/quit",
            ]),
            contextlib.redirect_stdout(output),
        ):
            self.example.main("openai/gpt-audio-1.5", "test/voice", playback=False, output_dir=directory)
        self.assertEqual(self.completion.call_count, 1)
        self.assertIn("needs audio input or audio output", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())
        self.assertNotIn("This text-only request should fail", str(self.completion.call_args.kwargs["messages"]))

    def test_provider_failure_keeps_previous_conversation_and_accepts_next_message(self):
        self.completion.side_effect = [
            response("Tea noted."),
            litellm.exceptions.BadRequestError(message="Unsupported setting", model="test/text", llm_provider="test"),
            response("You said tea."),
        ]
        output = io.StringIO()
        with mock.patch("builtins.input", side_effect=[
            "I like tea", "Failed message", "What did I say?", "/quit",
        ]), contextlib.redirect_stdout(output):
            self.example.main("test/text", "test/voice", playback=False)
        self.assertEqual(self.completion.call_count, 3)
        history = str(self.completion.call_args.kwargs["messages"])
        self.assertIn("I like tea", history)
        self.assertIn("Tea noted.", history)
        self.assertNotIn("Failed message", history)
        self.assertIn("Unable to get a reply:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
