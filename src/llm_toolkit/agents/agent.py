import base64
import copy
import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, final
from urllib.request import urlopen

import litellm
import soundfile as sf
from litellm import completion, transcription

from llm_toolkit.logging import get_logger
from llm_toolkit.media.file_source import (
    load_image_url,
    load_video_urls,
    save_base64_file,
    save_file,
)
from llm_toolkit.types import Content, InputEvent, Message, SessionData, TokenUsage

logger = get_logger(__name__)


@final
class Agent:
    def __init__(
        self,
        model="gpt-5",
        modalities=None,
        audio=None,
        messages=None,
        transcription_model="whisper-1",
        transcription_enabled=False,
        vision_fallback_model="gpt-4o-mini",
        tools=None,
        tool_handlers=None,
        tool_choice=None,
        system_prompt="You are a friendly assistant.",
    ):
        self.model = model
        self.modalities = list(dict.fromkeys(modalities or ["text"]))
        self.audio = audio or {"voice": "alloy", "format": "wav"}
        self.messages = list(messages or [])
        self.transcription_model = transcription_model
        self.transcription_enabled = transcription_enabled
        self.vision_fallback_model = vision_fallback_model
        self.tools = list(tools or [])
        self.tool_handlers = dict(tool_handlers or {})
        self.tool_choice = tool_choice
        self.system_prompt = system_prompt
        self.token_usage: TokenUsage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self._model_support = {}
        self.input_modalities = ["text"]
        self.last_image_path = None
        self.last_speech_path = None
        self.last_video_path = None

        self._update_model_support()
        logger.info(
            "Agent initialized with model %s, inputs %s, and outputs %s",
            self.model,
            self.input_modalities,
            self.modalities,
        )

    def _send_message(self, content: str | list[Content], role="user"):
        self.messages.append({"role": role, "content": content})
        logger.debug("Added %s message; history length is now %s", role, len(self.messages))

    def send_message(self, content: str | list[Content], role="user"):
        self._send_message(content, role=role)

    def get_messages(self) -> list[Message]:
        """
        Gets the current conversation history.

        Returns:
            list[Message]: The stored conversation messages.
        """

        return list(self.messages)

    def clear_messages(self):
        """
        Clears the current conversation history.
        """

        self.messages = []
        logger.info("Agent message history cleared")

    def get_session(self) -> SessionData:
        """
        Gets the current serializable agent session.

        Returns:
            SessionData: The current agent session.
        """

        session = {
            "model": self.model,
            "modalities": self.modalities,
            "audio": self.audio,
            "messages": self.messages,
            "transcription_model": self.transcription_model,
            "transcription_enabled": self.transcription_enabled,
            "vision_fallback_model": self.vision_fallback_model,
            "tools": self.tools,
            "tool_choice": self.tool_choice,
            "system_prompt": self.system_prompt,
            "token_usage": self.token_usage,
            "last_image_path": self.last_image_path,
            "last_speech_path": self.last_speech_path,
            "last_video_path": self.last_video_path,
        }
        return copy.deepcopy(session)

    def set_session(self, session: SessionData):
        """
        Updates the agent from a serialized session.

        Args:
            `session` (SessionData): Session values to apply.
        """

        self.model = session.get("model", self.model)
        self.modalities = list(session.get("modalities", self.modalities))
        self.audio = dict(session.get("audio", self.audio))
        self.messages = list(session.get("messages", self.messages))
        self.transcription_model = session.get(
            "transcription_model",
            self.transcription_model,
        )
        self.transcription_enabled = session.get(
            "transcription_enabled",
            self.transcription_enabled,
        )
        self.vision_fallback_model = session.get(
            "vision_fallback_model",
            self.vision_fallback_model,
        )
        self.tools = list(session.get("tools", self.tools))
        self.tool_handlers = {}
        self.tool_choice = session.get("tool_choice", self.tool_choice)
        self.system_prompt = session.get("system_prompt", self.system_prompt)
        self.token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            **session.get("token_usage", {}),
        }
        self.last_image_path = session.get("last_image_path")
        self.last_speech_path = session.get("last_speech_path")
        self.last_video_path = session.get("last_video_path")
        self._model_support = {}
        self.input_modalities = ["text"]
        self._update_model_support()
        logger.info("Agent session updated")

    def save_session(self, file_path: str) -> str:
        """
        Saves the current agent session as JSON.

        Args:
            `file_path` (str): Path used to save the session.

        Returns:
            str: The absolute path of the saved session.
        """

        session = json.dumps(self.get_session(), indent=2)
        saved_path = save_file(session, file_path)
        logger.info("Agent session saved to %s", saved_path)
        return saved_path

    def load_session(self, file_path: str) -> SessionData:
        """
        Loads an agent session from JSON.

        Args:
            `file_path` (str): Path to the saved session.

        Returns:
            SessionData: The loaded agent session.
        """

        with open(Path(file_path).expanduser(), encoding="utf-8") as session_file:
            session = json.load(session_file)

        self.set_session(session)
        logger.info("Agent session loaded from %s", file_path)
        return self.get_session()

    def set_system_prompt(self, system_prompt: str):
        """
        Updates the system prompt used for future responses.

        Args:
            `system_prompt` (str): The new system prompt.
        """

        self.system_prompt = system_prompt
        logger.info("Agent system prompt updated")

    def set_model(self, model: str):
        """
        Changes the model used for future responses.

        Args:
            `model` (str): The new LiteLLM model name.
        """

        self.model = model
        self._model_support = {}
        self.input_modalities = ["text"]
        self._update_model_support()
        logger.info("Agent model changed to %s", self.model)

    def set_output_modalities(self, modalities: list[str], audio=None):
        """
        Changes the output modalities used for future responses.

        Args:
            `modalities` (list[str]): Requested output modalities.
            `audio` (dict): Optional audio output configuration.
        """

        self.modalities = list(dict.fromkeys(modalities))
        if audio is not None:
            self.audio = dict(audio)

        if "audio" in self.modalities and not self._supports_audio_output():
            self.modalities.remove("audio")
            logger.warning("Model %s does not support audio output", self.model)

        logger.info("Agent output modalities changed to %s", self.modalities)

    def set_mode(
        self,
        model: str = None,
        system_prompt: str = None,
        modalities: list[str] = None,
        audio=None,
        transcription_enabled: bool = None,
    ):
        """
        Updates related model, prompt, and output settings together.

        Args:
            `model` (str): Optional LiteLLM model name.
            `system_prompt` (str): Optional system prompt.
            `modalities` (list[str]): Optional output modalities.
            `audio` (dict): Optional audio output configuration.
            `transcription_enabled` (bool): Optional transcription setting.
        """

        if model is not None:
            self.set_model(model)
        if system_prompt is not None:
            self.set_system_prompt(system_prompt)
        if modalities is not None or audio is not None:
            self.set_output_modalities(modalities or self.modalities, audio=audio)
        if transcription_enabled is not None:
            self.set_transcription(transcription_enabled)

        logger.info("Agent mode updated")

    def set_tools(
        self,
        tools: list[dict[str, Any]],
        tool_choice=None,
        tool_handlers: dict[str, Callable] = None,
    ):
        """
        Sets the tools passed to the model.

        Args:
            `tools` (list[dict[str, Any]]): The tool definitions.
            `tool_choice`: Optional tool choice configuration.
            `tool_handlers` (dict[str, Callable]): Optional handlers keyed by tool name.
        """

        self.tools = list(tools)
        self.tool_choice = tool_choice
        if tool_handlers is not None:
            self.tool_handlers = dict(tool_handlers)
        logger.info("Agent tools updated")

    def add_tool(self, tool: dict[str, Any], handler: Callable = None):
        """
        Adds a tool definition to the model.

        Args:
            `tool` (dict[str, Any]): The tool definition.
            `handler` (Callable): Optional function used to execute the tool.
        """

        self.tools.append(tool)
        if handler is not None:
            tool_name = tool["function"]["name"]
            self.tool_handlers[tool_name] = handler
        logger.info("Agent tool added")

    def register_tool(self, handler: Callable):
        """
        Creates and registers a tool definition from a function.

        Args:
            `handler` (Callable): Function to expose to the model.

        Returns:
            dict[str, Any]: The generated tool definition.
        """

        function = litellm.utils.function_to_dict(handler)
        tool = {"type": "function", "function": function}
        self.add_tool(tool, handler=handler)
        return tool

    def set_transcription(self, enabled: bool):
        """
        Updates whether audio is transcribed before being sent to the model.

        Args:
            `enabled` (bool): Whether to transcribe audio before sending it.
        """

        self.transcription_enabled = enabled
        logger.info("Audio transcription %s", "enabled" if enabled else "disabled")

    def toggle_transcription(self) -> bool:
        """
        Toggles whether audio is transcribed before being sent to the model.

        Returns:
            bool: Whether audio transcription is enabled.
        """

        self.set_transcription(not self.transcription_enabled)
        return self.transcription_enabled

    def send_audio(self, audio, caption=None, format=None):
        encoded_audio, audio_file, audio_format = self._prepare_audio(
            audio,
            format=format,
        )

        if self.transcription_enabled or not self._supports_audio_input():
            reason = "Transcription is enabled." if self.transcription_enabled else (
                f"Model {self.model} does not support audio input."
            )
            logger.warning(
                "%s Transcribing with %s first.",
                reason,
                self.transcription_model,
            )
            transcript = self._transcribe_audio(audio_file)
            text = transcript if caption is None else f"{caption}\n\n{transcript}"
            self.send_text(text)
            return

        message = [
            {
                "type": "input_audio",
                "input_audio": {"data": encoded_audio, "format": audio_format},
            }
        ]

        if caption:
            message.append({"type": "text", "text": caption})

        self._send_message(message)

    def send_text(self, text):
        message = [{"type": "text", "text": text}]

        self._send_message(message)

    def send_image(self, image, caption=None):
        image = self._prepare_image(image)

        if not self._supports_image_input():
            logger.warning(
                "Model %s does not support image input. Describing the image with %s first.",
                self.model,
                self.vision_fallback_model,
            )
            description = self._describe_image(image, caption=caption)
            self.send_text(description)
            return

        message = [{"type": "image_url", "image_url": {"url": image}}]

        if caption:
            message.append({"type": "text", "text": caption})

        self._send_message(message)

    def send_video(self, video, caption=None, sample_fps=2, max_frames=20):
        """
        Sends a video as sampled image frames.

        Args:
            `video` (str): Path to the video file.
            `caption` (str): Optional text to send with the video.
            `sample_fps` (int): The number of frames to sample per second.
            `max_frames` (int): Maximum number of frames to send.
        """

        if max_frames <= 0:
            raise ValueError("max_frames must be greater than 0")

        frames = load_video_urls(video, sample_fps=sample_fps)
        if len(frames) > max_frames:
            if max_frames == 1:
                frames = frames[:1]
            else:
                step = (len(frames) - 1) / (max_frames - 1)
                frames = [frames[round(index * step)] for index in range(max_frames)]
            logger.info("Limited video input to %s sampled frames", max_frames)

        prompt = caption or "Describe what happens in this video."
        message = [{"type": "text", "text": prompt}]
        message.extend(
            {"type": "image_url", "image_url": {"url": frame}}
            for frame in frames
        )

        if self._supports_image_input():
            self._send_message(message)
            return

        logger.warning(
            "Model %s does not support video frames. Describing them with %s first.",
            self.model,
            self.vision_fallback_model,
        )
        description = self._describe_content(message)
        self.send_text(description)

    def send_inputs(self, inputs: list[InputEvent]):
        """
        Sends confirmed inputs from an InputBuffer to the agent.

        Args:
            `inputs` (list[InputEvent]): Confirmed text, audio, image, or video inputs.
        """

        for input_event in inputs:
            match input_event["type"]:
                case "text":
                    self.send_text(input_event["value"])
                case "audio":
                    self.send_audio(input_event["value"])
                case "image":
                    self.send_image(input_event["value"])
                case "video":
                    self.send_video(input_event["value"])

    def generate_image(self, prompt: str, model=None, file_path=None, **options):
        """
        Generates an image and optionally saves the first result.

        Args:
            `prompt` (str): Description of the image to generate.
            `model` (str): Image model to use. Defaults to the agent model.
            `file_path` (str): Optional path used to save the first image.
            `options`: Additional options passed to LiteLLM.

        Returns:
            The image generation response.
        """

        response = litellm.image_generation(
            prompt=prompt,
            model=model or self.model,
            **options,
        )
        self._track_token_usage(response)

        if file_path is not None:
            self.last_image_path = self._save_generated_image(response, file_path)

        logger.info("Image generated with model %s", model or self.model)
        return response

    def generate_speech(
        self,
        text: str,
        model=None,
        voice="alloy",
        file_path=None,
        **options,
    ):
        """
        Generates speech and optionally streams it to a file.

        Args:
            `text` (str): Text to convert into speech.
            `model` (str): Speech model to use. Defaults to the agent model.
            `voice` (str): Voice used for speech generation. Defaults to alloy.
            `file_path` (str): Optional path used to save the generated speech.
            `options`: Additional options passed to LiteLLM.

        Returns:
            The speech generation response.
        """

        response = litellm.speech(
            model=model or self.model,
            voice=voice,
            input=text,
            **options,
        )

        if file_path is not None:
            output_path = Path(file_path).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            response.stream_to_file(output_path)
            self.last_speech_path = str(output_path)

        logger.info("Speech generated with model %s", model or self.model)
        return response

    def generate_video(self, prompt: str, model=None, **options):
        """
        Starts a video generation job.

        Args:
            `prompt` (str): Description of the video to generate.
            `model` (str): Video model to use. Defaults to the agent model.
            `options`: Additional options passed to LiteLLM.

        Returns:
            The video generation job response.
        """

        response = litellm.video_generation(
            prompt=prompt,
            model=model or self.model,
            **options,
        )
        logger.info("Video generation started with model %s", model or self.model)
        return response

    def get_video_status(self, video_id: str, **options):
        """
        Gets the current status of a video generation job.

        Args:
            `video_id` (str): Identifier returned by `generate_video`.
            `options`: Additional options passed to LiteLLM.

        Returns:
            The video status response.
        """

        return litellm.video_status(video_id=video_id, **options)

    def save_video(self, video_id: str, file_path: str, **options) -> str:
        """
        Downloads and saves a completed generated video.

        Args:
            `video_id` (str): Identifier of the completed video.
            `file_path` (str): Path used to save the video.
            `options`: Additional options passed to LiteLLM.

        Returns:
            str: The absolute path of the saved video.
        """

        video_bytes = litellm.video_content(video_id=video_id, **options)
        self.last_video_path = save_file(video_bytes, file_path)
        logger.info("Generated video saved to %s", self.last_video_path)
        return self.last_video_path

    def _save_generated_image(self, response, file_path: str) -> str:
        images = self._get_value(response, "data") or []
        if not images:
            raise ValueError("Image generation response did not contain an image")

        image = images[0]
        encoded_image = self._get_value(image, "b64_json")
        if encoded_image:
            return save_base64_file(encoded_image, file_path)

        image_url = self._get_value(image, "url")
        if image_url:
            with urlopen(image_url) as image_response:
                return save_file(image_response.read(), file_path)

        raise ValueError("Image generation response did not contain image data")

    def _update_model_support(self):
        self._sync_input_modality("audio", self._supports_audio_input())
        self._sync_input_modality("image", self._supports_image_input())

        if "audio" in self.modalities and not self._supports_audio_output():
            self.modalities.remove("audio")
            logger.warning("Model %s does not support audio output", self.model)

    def _sync_input_modality(self, modality: str, is_supported: bool):
        if is_supported and modality not in self.input_modalities:
            self.input_modalities.append(modality)
        elif not is_supported and modality in self.input_modalities:
            self.input_modalities.remove(modality)

    def _supports_audio_input(self):
        return self._call_litellm_supports("supports_audio_input")

    def _supports_image_input(self):
        return self._call_litellm_supports("supports_vision")

    def _supports_audio_output(self):
        return self._call_litellm_supports("supports_audio_output")

    def _call_litellm_supports(self, function_name: str):
        if function_name in self._model_support:
            return self._model_support[function_name]

        support_check = getattr(litellm, function_name)
        self._model_support[function_name] = bool(support_check(model=self.model))
        logger.debug(
            "Model %s %s=%s",
            self.model,
            function_name,
            self._model_support[function_name],
        )
        return self._model_support[function_name]

    def _prepare_audio(self, audio, format=None):
        audio_format = format

        if hasattr(audio, "read"):
            audio_bytes = audio.read()
            audio_name = getattr(audio, "name", "audio.wav")
            audio_format = audio_format or Path(audio_name).suffix[1:]
        elif isinstance(audio, bytes):
            audio_bytes = audio
        elif isinstance(audio, (str, Path)) and Path(audio).expanduser().exists():
            audio_path = Path(audio).expanduser()
            with open(audio_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_format = audio_format or audio_path.suffix[1:]
        elif isinstance(audio, str):
            audio_bytes = base64.b64decode(audio)
        else:
            audio_buffer = io.BytesIO()
            sf.write(audio_buffer, audio, 24000, format="WAV")
            audio_bytes = audio_buffer.getvalue()

        audio_format = audio_format or "wav"
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")
        transcription_file = io.BytesIO(audio_bytes)
        transcription_file.name = f"audio.{audio_format}"
        return encoded_audio, transcription_file, audio_format

    def _prepare_image(self, image):
        if isinstance(image, Path):
            return load_image_url(str(image))
        if isinstance(image, str) and not image.startswith(
            ("http://", "https://", "data:")
        ):
            return load_image_url(image)
        return image

    def _transcribe_audio(self, audio_file):
        logger.info("Transcribing audio with %s", self.transcription_model)
        response = transcription(model=self.transcription_model, file=audio_file)
        return response.text

    def _describe_image(self, image, caption=None):
        logger.info("Describing image with %s", self.vision_fallback_model)
        prompt = caption or "Describe this image in enough detail to answer questions about it."
        return self._describe_content(
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image}},
            ]
        )

    def _describe_content(self, content):
        response = completion(
            model=self.vision_fallback_model,
            messages=[
                {
                    "role": "user",
                    "content": content,
                }
            ],
        )
        return self._get_message_content(response.choices[0].message)

    def _get_message_content(self, message):
        if hasattr(message, "content"):
            return message.content
        return message["content"]

    def get_response(self, execute_tools: bool = True, max_tool_rounds: int = 5):
        """
        Gets a model response and optionally executes registered tool calls.

        Args:
            `execute_tools` (bool): Whether to execute registered tool calls.
            `max_tool_rounds` (int): Maximum consecutive rounds of tool calls.

        Returns:
            The final model response.
        """

        if max_tool_rounds < 0:
            raise ValueError("max_tool_rounds cannot be less than 0")

        logger.info("Requesting response from model %s", self.model)
        response = self._request_completion()
        tool_rounds = 0

        while execute_tools and self.tool_handlers:
            response_message = self.messages[-1]
            tool_calls = response_message.get("tool_calls") or []
            if not tool_calls:
                break
            if tool_rounds >= max_tool_rounds:
                logger.warning("Maximum tool call rounds reached")
                break

            self._execute_tool_calls(tool_calls)
            response = self._request_completion()
            tool_rounds += 1

        return response

    def _request_completion(self):
        messages = [{"role": "system", "content": self.system_prompt}, *self.messages]
        completion_args = {
            "model": self.model,
            "messages": messages,
        }

        if "audio" in self.modalities:
            completion_args["modalities"] = self.modalities
            completion_args["audio"] = self.audio

        if self.tools:
            completion_args["tools"] = self.tools
        if self.tool_choice is not None:
            completion_args["tool_choice"] = self.tool_choice

        response = completion(**completion_args)

        response_message = response.choices[0].message
        if hasattr(response_message, "model_dump"):
            response_message = response_message.model_dump(exclude_none=True)
        elif not isinstance(response_message, dict):
            response_message = dict(response_message)

        self.messages.append(response_message)
        self._track_token_usage(response)
        logger.debug("Stored assistant response; history length is now %s", len(self.messages))
        return response

    def _execute_tool_calls(self, tool_calls):
        for tool_call in tool_calls:
            tool_call_id = self._get_value(tool_call, "id")
            function = self._get_value(tool_call, "function")
            tool_name = self._get_value(function, "name")
            arguments = self._get_value(function, "arguments") or "{}"
            handler = self.tool_handlers.get(tool_name)

            if handler is None:
                result = {"error": f"No handler registered for tool {tool_name}"}
                logger.warning("No handler registered for tool %s", tool_name)
            else:
                try:
                    result = handler(**json.loads(arguments))
                except Exception as exc:
                    result = {"error": str(exc)}
                    logger.exception("Tool %s failed", tool_name)
                else:
                    logger.info("Tool %s executed", tool_name)

            if not isinstance(result, str):
                result = json.dumps(result, default=str)

            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "content": result,
                }
            )

    def _get_value(self, value, key):
        if isinstance(value, dict):
            return value.get(key)
        return getattr(value, key, None)

    def _track_token_usage(self, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return

        for key in self.token_usage:
            value = getattr(usage, key, None)
            if value is None and isinstance(usage, dict):
                value = usage.get(key)
            if value is not None:
                self.token_usage[key] += value

        logger.debug("Token usage updated: %s", self.token_usage)
