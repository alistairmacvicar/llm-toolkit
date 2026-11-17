import base64
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from litellm import AuthenticationError, NotFoundError

from llm_toolkit import Agent


class GeneratedMediaExampleTests(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).parents[1] / "examples" / "generated_media.py"
        spec = importlib.util.spec_from_file_location("generated_media", source)
        self.example = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.example)
        support = mock.patch.object(Agent, "_update_model_support")
        support.start()
        self.addCleanup(support.stop)
        generation = mock.patch(
            "llm_toolkit.agents.agent.litellm.image_generation",
            return_value={
                "data": [{"b64_json": base64.b64encode(b"image bytes").decode()}],
            },
        )
        self.generation = generation.start()
        self.addCleanup(generation.stop)

    def test_command_line_image_defaults_to_an_image_model_and_saves(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "image.png"
            arguments = SimpleNamespace(
                model="gpt-5", kind="image", media_model=None,
                prompt="A watercolor robot", output=str(destination),
            )
            with (
                mock.patch("builtins.input", side_effect=AssertionError("Unexpected prompt")),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = self.example.main(arguments)

            self.generation.assert_called_once_with(
                prompt="A watercolor robot", model="openai/gpt-image-2.5-flare",
            )
            self.assertEqual(destination.read_bytes(), b"image bytes")
            self.assertEqual(result, 0)

    def test_command_line_image_model_override_is_preserved(self):
        arguments = SimpleNamespace(
            model="gpt-5", kind="image", media_model="custom/image-model",
            prompt="A watercolor robot", output="custom.png",
        )
        with (
            mock.patch.object(self.example, "generate") as generate,
            mock.patch("builtins.input", side_effect=AssertionError("Unexpected prompt")),
        ):
            self.example.main(arguments)

        self.assertEqual(generate.call_args.args[1:], (
            "image", "custom/image-model", "A watercolor robot", "custom.png",
        ))

    def test_interactive_image_accepts_default_or_explicit_model(self):
        for supplied, entered, expected in (
            (None, "", "openai/gpt-image-2.5-flare"),
            ("custom/image-model", "", "custom/image-model"),
            (None, "custom/image-model", "custom/image-model"),
        ):
            with self.subTest(supplied=supplied, entered=entered):
                arguments = SimpleNamespace(
                    model="gpt-5", kind=None, media_model=supplied,
                    prompt=None, output=None,
                )
                with (
                    mock.patch.object(self.example, "generate") as generate,
                    mock.patch("builtins.input", side_effect=[
                        "image", entered, "A watercolor robot", "", "/quit",
                    ]) as user_input,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    self.example.main(arguments)

                self.assertEqual(generate.call_args.args[1:], (
                    "image", expected, "A watercolor robot", "recordings/generated-image.png",
                ))
                self.assertIn(
                    f"[{supplied or 'openai/gpt-image-2.5-flare'}]",
                    user_input.call_args_list[1].args[0],
                )

    def test_sora_404_explains_retirement_and_returns_failure(self):
        for model in ("sora-2", "openai/sora-2", "openai/sora-2-pro"):
            with self.subTest(model=model):
                arguments = SimpleNamespace(
                    model="gpt-5", kind="video", media_model=model,
                    prompt="Australia at dusk", output="video.mp4",
                )
                output = io.StringIO()
                with (
                    mock.patch("llm_toolkit.agents.agent.litellm.video_generation",
                               side_effect=NotFoundError("", model=model, llm_provider="openai")),
                    mock.patch("builtins.input", side_effect=AssertionError("Unexpected prompt")),
                    contextlib.redirect_stdout(output),
                ):
                    result = self.example.main(arguments)
                self.assertEqual(result, 1)
                self.assertIn("HTTP 404", output.getvalue())
                self.assertIn("24 September 2026", output.getvalue())
                self.assertIn("adding openai/ will not fix", output.getvalue())

    def test_interactive_video_failure_allows_next_generation(self):
        arguments = SimpleNamespace(
            model="gpt-5", kind=None, media_model=None, prompt=None, output=None,
        )
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "image.png"
            with (
                mock.patch("llm_toolkit.agents.agent.litellm.video_generation",
                           side_effect=NotFoundError("", model="sora-2", llm_provider="openai")),
                mock.patch("builtins.input", side_effect=[
                    "video", "sora-2", "Australia at dusk", "",
                    "image", "", "A watercolor robot", str(destination), "/quit",
                ]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(self.example.main(arguments), 0)
            self.assertEqual(destination.read_bytes(), b"image bytes")

    def test_other_provider_errors_are_not_misdiagnosed_as_sora_retirement(self):
        model = "vertex_ai/veo-3.1-generate-001"
        for error in (
            NotFoundError("Model not available", model=model, llm_provider="vertex_ai"),
            AuthenticationError("Invalid credentials", model=model, llm_provider="vertex_ai"),
        ):
            with self.subTest(error=type(error).__name__):
                arguments = SimpleNamespace(
                    model="gpt-5", kind="video", media_model=model,
                    prompt="Australia at dusk", output="video.mp4",
                )
                output = io.StringIO()
                with (
                    mock.patch("llm_toolkit.agents.agent.litellm.video_generation", side_effect=error),
                    contextlib.redirect_stdout(output),
                ):
                    self.assertEqual(self.example.main(arguments), 1)
                self.assertIn("Video generation failed", output.getvalue())
                self.assertNotIn("retired", output.getvalue())

    def test_video_status_and_download_errors_retain_the_same_job(self):
        model = "vertex_ai/veo-3.1-generate-001"
        agent = mock.Mock()
        agent.generate_video.return_value = SimpleNamespace(id="video-1")
        agent.get_video_status.side_effect = [
            NotFoundError("Job temporarily unavailable", model=model, llm_provider="vertex_ai"),
            {"id": "video-1", "status": "completed"},
        ]
        agent.save_video.side_effect = [OSError("Disk unavailable"), "video.mp4"]
        output = io.StringIO()
        with (
            mock.patch("builtins.input", side_effect=["status", "status", "save", "save"]),
            contextlib.redirect_stdout(output),
        ):
            self.example.generate(agent, "video", model, "Australia at dusk", "video.mp4")
        agent.generate_video.assert_called_once()
        self.assertEqual(agent.get_video_status.call_args_list, [mock.call("video-1")] * 2)
        self.assertEqual(agent.save_video.call_args_list, [mock.call("video-1", "video.mp4")] * 2)
        self.assertIn("The video job is still selected", output.getvalue())
        self.assertIn("Saved video: video.mp4", output.getvalue())


if __name__ == "__main__":
    unittest.main()
