import asyncio
import base64
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from llm_toolkit.logging import get_logger
from llm_toolkit.types import InputEvent

logger = get_logger(__name__)


@final
class RealtimeSession:
    def __init__(
        self,
        url: str,
        model: str,
        api_key: str = None,
        headers: dict[str, str] = None,
        websocket_factory: Callable = None,
        sample_rate: int = 24000,
    ):
        self.url = self._get_url(url, model)
        self.model = model
        self.headers = dict(headers or {})
        self.sample_rate = sample_rate
        self._websocket_factory = websocket_factory
        self._websocket = None

        if api_key is not None:
            self.headers.setdefault("Authorization", f"Bearer {api_key}")

    @property
    def connected(self) -> bool:
        """
        Gets whether the realtime websocket is connected.

        Returns:
            bool: Whether the websocket is connected.
        """

        return self._websocket is not None

    async def connect(self):
        """
        Connects to the realtime websocket.

        Returns:
            RealtimeSession: The connected realtime session.
        """

        websocket_factory = self._websocket_factory
        if websocket_factory is None:
            from websockets.asyncio.client import connect

            websocket_factory = connect

        self._websocket = await websocket_factory(
            self.url,
            additional_headers=self.headers,
        )
        logger.info("Realtime session connected to %s", self.url)
        return self

    async def close(self):
        """
        Closes the realtime websocket.
        """

        if self._websocket is None:
            return

        await self._websocket.close()
        self._websocket = None
        logger.info("Realtime session closed")

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exception_type, exception, traceback):
        await self.close()

    async def send_event(self, event: dict[str, Any]):
        """
        Sends a raw client event to the realtime websocket.

        Args:
            `event` (dict[str, Any]): Realtime client event.
        """

        if self._websocket is None:
            raise RuntimeError("Realtime session is not connected")

        await self._websocket.send(json.dumps(event))
        logger.debug("Realtime event sent: %s", event.get("type"))

    async def receive_event(self) -> dict[str, Any]:
        """
        Receives the next server event from the realtime websocket.

        Returns:
            dict[str, Any]: The next realtime server event.
        """

        if self._websocket is None:
            raise RuntimeError("Realtime session is not connected")

        event = json.loads(await self._websocket.recv())
        logger.debug("Realtime event received: %s", event.get("type"))
        return event

    async def events(self) -> AsyncIterator[dict[str, Any]]:
        """
        Iterates over server events until the websocket closes.

        Yields:
            dict[str, Any]: Realtime server events.
        """

        if self._websocket is None:
            raise RuntimeError("Realtime session is not connected")

        async for message in self._websocket:
            event = json.loads(message)
            logger.debug("Realtime event received: %s", event.get("type"))
            yield event

    async def update_session(
        self,
        instructions: str = None,
        output_modalities: list[str] = None,
        **options,
    ):
        """
        Updates the active realtime session.

        Args:
            `instructions` (str): Optional system instructions.
            `output_modalities` (list[str]): Optional output modalities.
            `options`: Additional realtime session options.
        """

        session = {"type": "realtime", **options}
        if instructions is not None:
            session["instructions"] = instructions
        if output_modalities is not None:
            session["output_modalities"] = output_modalities

        await self.send_event({"type": "session.update", "session": session})

    async def send_text(self, text: str, create_response: bool = True):
        """
        Sends a text message to the realtime conversation.

        Args:
            `text` (str): Text to send.
            `create_response` (bool): Whether to request a response immediately.
        """

        await self._send_content([{"type": "input_text", "text": text}])
        if create_response:
            await self.request_response()

    async def append_audio(self, audio):
        """
        Appends PCM16 audio to the realtime input buffer.

        Args:
            `audio`: PCM16 bytes, a base64 string, an audio file, or an audio array.
        """

        encoded_audio = self._prepare_audio(audio)
        await self.send_event(
            {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio,
            }
        )

    async def commit_audio(self, create_response: bool = True):
        """
        Commits buffered audio to the realtime conversation.

        Args:
            `create_response` (bool): Whether to request a response immediately.
        """

        await self.send_event({"type": "input_audio_buffer.commit"})
        if create_response:
            await self.request_response()

    async def send_image(self, image, create_response: bool = True):
        """
        Sends an image to the realtime conversation.

        Args:
            `image`: Image URL, data URL, or local image path.
            `create_response` (bool): Whether to request a response immediately.
        """

        image = self._prepare_image(image)

        await self._send_content([{"type": "input_image", "image_url": image}])
        if create_response:
            await self.request_response()

    async def send_inputs(
        self,
        inputs: list[InputEvent],
        create_response: bool = True,
        video_sample_fps: int = 2,
        video_max_frames: int = 20,
    ):
        """
        Sends confirmed InputBuffer values to the realtime conversation.

        Args:
            `inputs` (list[InputEvent]): Confirmed inputs to send.
            `create_response` (bool): Whether to request a response after sending.
            `video_sample_fps` (int): Number of video frames sampled per second.
            `video_max_frames` (int): Maximum video frames sent as images.
        """

        content = []
        audio_inputs = []

        for input_event in inputs:
            match input_event["type"]:
                case "text":
                    content.append({"type": "input_text", "text": input_event["value"]})
                case "audio":
                    audio_inputs.append(input_event["value"])
                case "image":
                    image = self._prepare_image(input_event["value"])
                    content.append({"type": "input_image", "image_url": image})
                case "video":
                    frames = self._prepare_video(
                        input_event["value"],
                        sample_fps=video_sample_fps,
                        max_frames=video_max_frames,
                    )
                    content.extend(
                        {"type": "input_image", "image_url": frame}
                        for frame in frames
                    )

        if content:
            await self._send_content(content)

        for audio in audio_inputs:
            await self.append_audio(audio)
        if audio_inputs:
            await self.commit_audio(create_response=False)

        if create_response and (content or audio_inputs):
            await self.request_response()

    async def request_response(self, **response_options):
        """
        Requests a response from the realtime model.

        Args:
            `response_options`: Optional response configuration.
        """

        await self.send_event(
            {
                "type": "response.create",
                "response": response_options,
            }
        )

    def create_send_handler(self, loop=None):
        """
        Creates an ActionHandler callback that schedules realtime sends.

        Args:
            `loop`: Event loop used to schedule sends. Defaults to the running loop.

        Returns:
            Callable: Callback accepting confirmed InputBuffer values.
        """

        event_loop = loop or asyncio.get_running_loop()

        def send_handler(inputs):
            return asyncio.run_coroutine_threadsafe(
                self.send_inputs(inputs),
                event_loop,
            )

        return send_handler

    async def _send_content(self, content):
        await self.send_event(
            {
                "type": "conversation.item.create",
                "item": {
                    "type": "message",
                    "role": "user",
                    "content": content,
                },
            }
        )

    def _prepare_image(self, image):
        from llm_toolkit.media.file_source import load_image_url

        if isinstance(image, Path):
            return load_image_url(str(image))
        if isinstance(image, str) and not image.startswith(
            ("http://", "https://", "data:")
        ):
            return load_image_url(image)
        return image

    def _prepare_video(self, video, sample_fps: int, max_frames: int):
        from llm_toolkit.media.file_source import load_video_urls

        if max_frames <= 0:
            raise ValueError("video_max_frames must be greater than 0")

        frames = load_video_urls(video, sample_fps=sample_fps)
        if len(frames) <= max_frames:
            return frames
        if max_frames == 1:
            return frames[:1]

        step = (len(frames) - 1) / (max_frames - 1)
        return [frames[round(index * step)] for index in range(max_frames)]

    def _prepare_audio(self, audio) -> str:
        if isinstance(audio, str):
            audio_path = Path(audio).expanduser()
            try:
                is_audio_file = audio_path.exists()
            except OSError:
                is_audio_file = False

            if not is_audio_file:
                return audio
            audio = audio_path

        if isinstance(audio, bytes):
            audio_bytes = audio
        else:
            import numpy as np
            import soundfile as sf

            if isinstance(audio, Path):
                audio, sample_rate = sf.read(audio, dtype="float32")
                if sample_rate != self.sample_rate:
                    raise ValueError(
                        f"Audio sample rate must be {self.sample_rate} Hz, "
                        f"got {sample_rate} Hz"
                    )

            audio_array = np.asarray(audio)
            if np.issubdtype(audio_array.dtype, np.integer):
                audio_array = audio_array.astype(np.int16)
            else:
                audio_array = (np.clip(audio_array, -1, 1) * 32767).astype(np.int16)
            audio_bytes = audio_array.tobytes()

        return base64.b64encode(audio_bytes).decode("utf-8")

    def _get_url(self, url: str, model: str) -> str:
        parsed_url = urlsplit(url)
        query = dict(parse_qsl(parsed_url.query))
        query["model"] = model
        return urlunsplit(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                urlencode(query),
                parsed_url.fragment,
            )
        )
