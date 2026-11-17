import asyncio
import base64
import copy
import json
import math
import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, final
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from llm_toolkit.logging import get_logger
from llm_toolkit.types import InputEvent

logger = get_logger(__name__)


@final
class RealtimeSession:
    @staticmethod
    def create_openai_webrtc_answer(
        offer_sdp: str,
        *,
        model: str,
        instructions: str = "",
        voice: str = "marin",
        transcription_model: str = "gpt-4o-mini-transcribe",
        api_key: str = None,
        timeout: float = 30,
    ) -> str:
        """Exchange a browser's SDP offer for an OpenAI Realtime SDP answer.

        Call on the application server. The standard API key stays on the
        server; only the SDP answer is returned to the browser. Audio and
        interruption handling then use WebRTC directly. This OpenAI-specific
        helper is independent of the existing LiteLLM websocket connection.
        No connection is created until this method is called.
        """
        import httpx

        if not isinstance(offer_sdp, str) or not offer_sdp.startswith("v=0"):
            raise ValueError("A valid SDP offer is required")
        if len(offer_sdp) > 1_000_000:
            raise ValueError("SDP offer is too large")
        if not model.strip():
            raise ValueError("A realtime model is required")
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("Set OPENAI_API_KEY on the server to use live voice")
        session = {
            "type": "realtime",
            "model": model,
            "instructions": instructions,
            "output_modalities": ["audio"],
            "audio": {
                "input": {
                    "transcription": {"model": transcription_model},
                    "turn_detection": {
                        "type": "server_vad",
                        "create_response": True,
                        "interrupt_response": True,
                    },
                },
                "output": {"voice": voice},
            },
        }
        response = httpx.post(
            "https://api.openai.com/v1/realtime/calls",
            headers={"Authorization": f"Bearer {key}"},
            files={
                "sdp": (None, offer_sdp),
                "session": (None, json.dumps(session), "application/json"),
            },
            timeout=timeout,
        )
        if not response.is_success:
            # Do not include request headers or response bodies containing secrets.
            raise RuntimeError(
                f"Realtime connection failed (HTTP {response.status_code}). "
                "Check the API key, model access, and account quota."
            )
        if not response.text.startswith("v=0"):
            raise RuntimeError("Realtime provider did not return an SDP answer")
        return response.text

    def __init__(
        self,
        url: str = None,
        model: str = "openai/gpt-realtime",
        api_key: str = None,
        headers: dict[str, str] = None,
        websocket_factory: Callable = None,
        sample_rate: int = 24000,
        *,
        provider_options: dict[str, Any] = None,
        connect_timeout: float = 30,
        recording_dir=None,
    ):
        """Connect by model through LiteLLM, or supply an explicit proxy URL.

        With no URL, LiteLLM resolves provider credentials from the environment
        and translates realtime events in process. `provider_options` accepts
        provider settings such as `api_base`, `api_version`, or `vertex_project`.
        An explicit URL retains the OpenAI-compatible WebSocket transport.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("A realtime model is required")
        if not math.isfinite(connect_timeout) or connect_timeout <= 0:
            raise ValueError("connect_timeout must be finite and positive")
        if provider_options is not None and not isinstance(provider_options, dict):
            raise ValueError("provider_options must be an object")
        if url is not None and provider_options:
            raise ValueError("Use provider_options with a model-only connection; use headers/api_key with a proxy URL")
        if url is None and websocket_factory is not None:
            raise ValueError("websocket_factory requires an explicit URL")
        reserved = {"model", "websocket", "api_key", "headers", "timeout"}
        if reserved.intersection(provider_options or {}):
            raise ValueError("provider_options cannot override model, websocket, api_key, headers, or timeout")
        self.url = self._get_url(url, model) if url is not None else None
        self.model = model
        self.headers = dict(headers or {})
        self._api_key = api_key
        self.provider_options = copy.deepcopy(provider_options or {})
        self.connect_timeout = connect_timeout
        self.sample_rate = sample_rate
        self._websocket_factory = websocket_factory
        self._websocket = None
        self._connection_lock = asyncio.Lock()
        self.recording_dir = recording_dir
        self.last_input_audio_path = None
        self._audio_archive = None

        if api_key is not None and self.url is not None:
            self.headers.setdefault("Authorization", f"Bearer {api_key}")

    @property
    def connected(self) -> bool:
        """
        Gets whether the realtime websocket is connected.

        Returns:
            bool: Whether the websocket is connected.
        """

        return self._websocket is not None and getattr(self._websocket, "close_code", None) is None

    async def connect(self):
        """
        Connects to the realtime websocket.

        Returns:
            RealtimeSession: The connected realtime session.
        """

        async with self._connection_lock:
            if self.connected:
                return self
            websocket, self._websocket = self._websocket, None
            try:
                if websocket is not None:
                    await websocket.close()
            finally:
                self._close_audio_archive()
            if self.url is None:
                from llm_toolkit.agents.realtime_transport import ProviderConnection

                self._websocket = await ProviderConnection.connect(
                    model=self.model, api_key=self._api_key, headers=self.headers,
                    timeout=self.connect_timeout, options=copy.deepcopy(self.provider_options),
                )
            else:
                websocket_factory = self._websocket_factory
                if websocket_factory is None:
                    from websockets.asyncio.client import connect

                    websocket_factory = connect
                self._websocket = await asyncio.wait_for(websocket_factory(
                    self.url,
                    additional_headers=self.headers,
                ), timeout=self.connect_timeout)
        logger.info("Realtime session connected with model %s", self.model)
        return self

    async def close(self):
        """
        Closes the realtime websocket.
        """

        async with self._connection_lock:
            websocket, self._websocket = self._websocket, None
            try:
                if websocket is not None:
                    await websocket.close()
            finally:
                self._close_audio_archive()
        logger.info("Realtime session closed")

    async def __aenter__(self):
        return await self.connect()

    async def __aexit__(self, exception_type, exception, traceback):
        await self.close()

    async def send_event(self, event: dict[str, Any]):
        """
        Sends a raw client event to the realtime websocket.

        Args:
            event (dict[str, Any]): Realtime client event.
        """

        if self._websocket is None:
            raise RuntimeError("Realtime session is not connected")

        if event.get("type") == "input_audio_buffer.append":
            self._archive_audio(event["audio"])
        elif event.get("type") == "conversation.item.create":
            for part in event.get("item", {}).get("content", []):
                if part.get("type") == "input_audio" and part.get("audio"):
                    self._archive_audio(part["audio"])
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
        self._log_transcript(event)
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
            self._log_transcript(event)
            yield event

    def _archive_audio(self, encoded):
        from llm_toolkit.media.recording import PCMArchive

        data = base64.b64decode(encoded, validate=True)
        if not data:
            return
        if self._audio_archive is None:
            self._audio_archive = PCMArchive(self.recording_dir, self.sample_rate)
            self.last_input_audio_path = str(self._audio_archive.path.resolve())
        self._audio_archive.write(data)

    def _close_audio_archive(self):
        archive, self._audio_archive = self._audio_archive, None
        if archive is not None:
            archive.close()

    @staticmethod
    def _log_transcript(event):
        kind = event.get("type")
        if kind == "conversation.item.input_audio_transcription.completed":
            logger.info("User audio transcript: %s", event.get("transcript", ""))
        elif kind in {"response.output_audio_transcript.done", "response.audio_transcript.done"}:
            logger.info("Assistant audio transcript: %s", event.get("transcript", ""))

    async def update_session(
        self,
        instructions: str = None,
        output_modalities: list[str] = None,
        **options,
    ):
        """
        Updates the active realtime session.

        Args:
            instructions (str): Optional system instructions.
            output_modalities (list[str]): Optional output modalities.
            options (Any): Additional realtime session options.
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
            text (str): Text to send.
            create_response (bool): Whether to request a response immediately.
        """

        await self._send_content([{"type": "input_text", "text": text}])
        if create_response:
            await self.request_response()

    async def append_audio(self, audio):
        """
        Appends PCM16 audio to the realtime input buffer.

        Args:
            audio (Any): PCM16 bytes, a base64 string, an audio file, or an audio array.
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
            create_response (bool): Whether to request a response immediately.
        """

        await self.send_event({"type": "input_audio_buffer.commit"})
        if create_response:
            await self.request_response()

    async def send_image(self, image, create_response: bool = True):
        """
        Sends an image to the realtime conversation.

        Args:
            image (str | Path): Image URL, data URL, or local image path.
            create_response (bool): Whether to request a response immediately.
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
            inputs (list[InputEvent]): Confirmed inputs to send.
            create_response (bool): Whether to request a response after sending.
            video_sample_fps (int): Number of video frames sampled per second.
            video_max_frames (int): Maximum video frames sent as images.
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
            response_options (Any): Optional response configuration.
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
            loop (asyncio.AbstractEventLoop | None): Event loop used to schedule sends. Defaults to the running loop.

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

        return load_video_urls(video, sample_fps=sample_fps, max_frames=max_frames)

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
            if audio_array.ndim not in (1, 2) or (audio_array.ndim == 2 and audio_array.shape[1] == 0):
                raise ValueError("Audio must have shape (samples,) or (samples, channels)")
            if np.issubdtype(audio_array.dtype, np.integer):
                # Integer arrays are PCM16 sample values, even after downmixing.
                if audio_array.ndim == 2:
                    audio_array = audio_array.mean(axis=1)
                audio_array = np.clip(audio_array, -32768, 32767).astype("<i2")
            else:
                if audio_array.ndim == 2:
                    audio_array = audio_array.mean(axis=1)
                audio_array = (np.clip(audio_array, -1, 1) * 32767).astype("<i2")
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
