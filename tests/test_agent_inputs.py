import base64
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


from llm_toolkit.agents import agent as agent_module
from llm_toolkit.agents.agent import Agent


class AgentInputTests(unittest.TestCase):
    def test_generation_parameters_are_passed_and_session_copies_are_independent(self):
        options = {"temperature": 0.3, "max_tokens": 120, "response_format": {"type": "json_object"}}
        agent = Agent(completion_options=options)
        options["response_format"]["type"] = "changed"
        response = types.SimpleNamespace(choices=[types.SimpleNamespace(message={"role": "assistant", "content": "{}"})], usage=None)
        with mock.patch.object(agent_module, "completion", return_value=response) as request:
            agent.get_response()
        self.assertEqual(request.call_args.kwargs["temperature"], 0.3)
        self.assertEqual(request.call_args.kwargs["response_format"], {"type": "json_object"})
        restored = Agent()
        restored.set_session(agent.get_session())
        agent.completion_options["max_tokens"] = 999
        self.assertEqual(restored.completion_options["max_tokens"], 120)

    def test_generation_parameters_cannot_override_routing_or_stream(self):
        for options in ({"messages": []}, {"stream": True}, {"api_key": "secret"}, {"temperature": float("nan")}, []):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    Agent.validate_completion_options(options)

    def setUp(self):
        # Another suite may already have imported real LiteLLM. Patch per test
        # instead of relying on the import-time stand-in or collection order.
        for name, supported in (("supports_audio_input", True), ("supports_audio_output", False), ("supports_vision", True)):
            patcher = mock.patch.object(agent_module.litellm, name, return_value=supported)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name in ("completion", "transcription"):
            patcher = mock.patch.object(agent_module, name, side_effect=AssertionError("Unexpected provider call in an offline test"))
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_input_and_output_modalities_are_tracked_separately(self):
        agent = Agent(modalities=["text", "audio"])

        self.assertEqual(agent.input_modalities, ["text", "audio", "image"])
        self.assertEqual(agent.modalities, ["text"])

    def test_model_can_be_changed_and_capabilities_are_refreshed(self):
        agent = Agent(model="first/model")
        agent_module.litellm.supports_audio_input.return_value = False
        agent_module.litellm.supports_vision.return_value = False

        agent.set_model("second/model")

        self.assertEqual(agent.model, "second/model")
        self.assertEqual(agent.input_modalities, ["text"])
        agent_module.litellm.supports_vision.assert_called_with(model="second/model")

    def test_mode_updates_prompt_output_and_transcription_together(self):
        agent_module.litellm.supports_audio_output.return_value = True
        agent = Agent()

        agent.set_mode(
            system_prompt="Answer as a concise critic.",
            modalities=["text", "audio"],
            audio={"voice": "nova", "format": "mp3"},
            transcription_enabled=True,
        )

        self.assertEqual(agent.system_prompt, "Answer as a concise critic.")
        self.assertEqual(agent.modalities, ["text", "audio"])
        self.assertEqual(agent.audio, {"voice": "nova", "format": "mp3"})
        self.assertTrue(agent.transcription_enabled)

    def test_audio_bytes_are_encoded_for_supported_models(self):
        agent = Agent()

        agent.send_audio(b"wav data", format="wav")

        audio = agent.messages[0]["content"][0]["input_audio"]
        self.assertEqual(audio["data"], base64.b64encode(b"wav data").decode("utf-8"))
        self.assertEqual(audio["format"], "wav")

    def test_transcribe_audio_returns_text_without_mutating_history(self):
        agent = Agent()
        agent.send_text("Existing history")
        before = agent.get_messages()
        with mock.patch.object(agent_module, "transcription", return_value={"text": "Hello"}) as transcribe:
            text = agent.transcribe_audio(b"audio", model="test/transcribe", format="webm", language="en")
        self.assertEqual(text, "Hello")
        self.assertEqual(agent.get_messages(), before)
        self.assertEqual(transcribe.call_args.kwargs["model"], "test/transcribe")
        self.assertEqual(transcribe.call_args.kwargs["file"].name, "audio.webm")
        self.assertEqual(transcribe.call_args.kwargs["language"], "en")

    def test_transcription_error_leaves_history_unchanged(self):
        agent = Agent()
        with mock.patch.object(agent_module, "transcription", side_effect=RuntimeError("Failed")):
            with self.assertRaises(RuntimeError):
                agent.transcribe_audio(b"audio")
        self.assertEqual(agent.get_messages(), [])

    def test_enabled_transcription_sends_text(self):
        agent = Agent(transcription_enabled=True)
        transcription = types.SimpleNamespace(text="transcribed speech")

        with mock.patch.object(agent_module, "transcription", return_value=transcription):
            agent.send_audio(b"wav data", caption="Context")

        self.assertEqual(
            agent.messages[0]["content"],
            [{"type": "text", "text": "Context\n\ntranscribed speech"}],
        )

    def test_local_image_is_converted_to_a_data_url(self):
        agent = Agent()

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "image.jpg"
            image_path.write_bytes(b"image data")
            agent.send_image(image_path)

        image_url = agent.messages[0]["content"][0]["image_url"]["url"]
        self.assertEqual(
            image_url,
            "data:image/jpeg;base64," + base64.b64encode(b"image data").decode("utf-8"),
        )

    def test_video_limit_is_applied_by_loader_before_retaining_frames(self):
        agent = Agent()
        frames = [f"data:image/jpeg;base64,{index}" for index in (0, 2, 4)]

        with mock.patch.object(agent_module, "load_video_urls", return_value=frames) as loader:
            agent.send_video("video.mp4", max_frames=3)
        loader.assert_called_once_with("video.mp4", sample_fps=2, max_frames=3)

        content = agent.messages[0]["content"]
        selected_frames = [item["image_url"]["url"] for item in content[1:]]
        self.assertEqual(selected_frames, frames)

    def test_confirmed_inputs_can_be_sent_directly_to_agent(self):
        agent = Agent()

        agent.send_inputs(
            [
                {"type": "text", "value": "hello"},
                {"type": "audio", "value": b"wav data"},
            ]
        )

        self.assertEqual(agent.messages[0]["content"][0]["text"], "hello")
        self.assertEqual(agent.messages[1]["content"][0]["type"], "input_audio")

    def test_text_completion_omits_provider_specific_modality_parameters(self):
        agent = Agent(model="anthropic/test-model")
        response = self._response({"role": "assistant", "content": "Hello"})
        agent.send_text("Hello")

        with mock.patch.object(agent_module, "completion", return_value=response) as call:
            agent.get_response()

        self.assertNotIn("modalities", call.call_args.kwargs)
        self.assertNotIn("audio", call.call_args.kwargs)

    def test_registered_tool_is_executed_and_returned_to_model(self):
        agent = Agent()
        tool = {
            "type": "function",
            "function": {
                "name": "add",
                "description": "Adds two numbers.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "left": {"type": "integer"},
                        "right": {"type": "integer"},
                    },
                    "required": ["left", "right"],
                },
            },
        }
        agent.add_tool(tool, handler=lambda left, right: {"total": left + right})
        agent.send_text("Add two and three")
        responses = [
            self._response(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "add",
                                "arguments": '{"left": 2, "right": 3}',
                            },
                        }
                    ],
                }
            ),
            self._response({"role": "assistant", "content": "The total is five."}),
        ]

        with mock.patch.object(agent_module, "completion", side_effect=responses) as completion:
            response = agent.get_response()

        self.assertIs(response, responses[1])
        self.assertEqual(completion.call_count, 2)
        self.assertEqual(agent.messages[-2]["role"], "tool")
        self.assertEqual(agent.messages[-2]["content"], '{"total": 5}')
        follow_up_messages = completion.call_args_list[1].kwargs["messages"]
        self.assertEqual(follow_up_messages[-1]["tool_call_id"], "call-1")
        self.assertEqual(agent.token_usage["total_tokens"], 6)

    def test_register_tool_generates_definition(self):
        agent = Agent()

        def ping():
            """Returns pong."""

            return "pong"

        tool = agent.register_tool(ping)

        self.assertEqual(tool["function"]["name"], "ping")
        self.assertIs(agent.tool_handlers["ping"], ping)

    def test_session_round_trips_through_json(self):
        agent = Agent(
            model="test/model",
            transcription_enabled=True,
            system_prompt="Session prompt",
        )
        agent.send_text("Remember this")
        agent.token_usage["total_tokens"] = 12
        agent.tool_handlers["temporary"] = lambda: None

        with tempfile.TemporaryDirectory() as temp_dir:
            session_path = Path(temp_dir) / "session.json"
            saved_path = agent.save_session(str(session_path))
            restored = Agent()
            session = restored.load_session(saved_path)

        self.assertEqual(session["model"], "test/model")
        self.assertEqual(restored.system_prompt, "Session prompt")
        self.assertEqual(restored.messages, agent.messages)
        self.assertEqual(restored.token_usage["total_tokens"], 12)
        self.assertTrue(restored.transcription_enabled)
        self.assertEqual(restored.tool_handlers, {})

    def test_generated_image_can_be_saved_from_base64_response(self):
        agent = Agent()
        response = types.SimpleNamespace(
            data=[
                types.SimpleNamespace(
                    b64_json=base64.b64encode(b"generated image").decode("utf-8"),
                    url=None,
                )
            ],
            usage=types.SimpleNamespace(
                prompt_tokens=1,
                completion_tokens=0,
                total_tokens=1,
            ),
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "generated.png"
            with mock.patch.object(
                agent_module.litellm,
                "image_generation",
                return_value=response,
            ):
                result = agent.generate_image(
                    "A geometric landscape",
                    model="image/model",
                    file_path=str(image_path),
                )
            saved_image = image_path.read_bytes()

        self.assertIs(result, response)
        self.assertEqual(saved_image, b"generated image")
        self.assertEqual(agent.token_usage["total_tokens"], 1)

    def test_generated_speech_can_be_streamed_to_a_file(self):
        agent = Agent()
        response = types.SimpleNamespace(stream_to_file=mock.Mock())

        with tempfile.TemporaryDirectory() as temp_dir:
            speech_path = Path(temp_dir) / "speech.mp3"
            with mock.patch.object(
                agent_module.litellm,
                "speech",
                return_value=response,
            ):
                agent.generate_speech(
                    "Hello",
                    model="speech/model",
                    file_path=str(speech_path),
                )

            response.stream_to_file.assert_called_once_with(speech_path.resolve())
            self.assertEqual(agent.last_speech_path, str(speech_path.resolve()))

    def test_video_generation_status_and_download_are_exposed(self):
        agent = Agent()
        job = types.SimpleNamespace(id="video-1", status="queued")
        status = types.SimpleNamespace(id="video-1", status="completed")

        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "video.mp4"
            with (
                mock.patch.object(
                    agent_module.litellm,
                    "video_generation",
                    return_value=job,
                ),
                mock.patch.object(
                    agent_module.litellm,
                    "video_status",
                    return_value=status,
                ),
                mock.patch.object(
                    agent_module.litellm,
                    "video_content",
                    return_value=b"generated video",
                ),
            ):
                generated = agent.generate_video("A quiet street", model="video/model")
                current_status = agent.get_video_status(job.id)
                saved_path = agent.save_video(job.id, str(video_path))
            saved_video = video_path.read_bytes()

        self.assertIs(generated, job)
        self.assertIs(current_status, status)
        self.assertEqual(saved_video, b"generated video")
        self.assertEqual(saved_path, str(video_path.resolve()))

    def _response(self, message):
        usage = types.SimpleNamespace(
            prompt_tokens=1,
            completion_tokens=2,
            total_tokens=3,
        )
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)],
            usage=usage,
        )


if __name__ == "__main__":
    unittest.main()
