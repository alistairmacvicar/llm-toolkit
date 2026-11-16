import base64
import contextlib
import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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
                self.example.main(arguments)

            self.generation.assert_called_once_with(
                prompt="A watercolor robot", model="openai/gpt-image-2.5-flare",
            )
            self.assertEqual(destination.read_bytes(), b"image bytes")

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


if __name__ == "__main__":
    unittest.main()
