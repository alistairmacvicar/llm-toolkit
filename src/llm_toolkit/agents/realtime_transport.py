"""Adapt LiteLLM's realtime bridge to an in-process, bounded event transport.

LiteLLM currently exposes its provider routing through the internal `_arealtime`
entry point and a small WebSocket protocol intended for proxy hosts. Keep that
dependency here, so applications need neither an ASGI server nor a local port.
"""

import asyncio
import json
from datetime import datetime
from uuid import uuid4

import litellm


class RealtimeConnectionError(ConnectionError):
    """A realtime provider failed to connect or closed the event stream."""


async def _run_bridge(bridge, options):
    # LiteLLM's Google adapters default to sending setup before the application
    # can supply its instructions. Google accepts only one setup. Configure an
    # adapter instance per connection instead of changing LiteLLM's global flag.
    model = options["model"]
    if not model.startswith(("gemini/", "vertex_ai/")):
        return await bridge(**options)

    from litellm.realtime_api import main as realtime

    model, provider, key, base = litellm.get_llm_provider(
        model=model, api_key=options.get("api_key"), api_base=options.get("api_base"),
    )
    if provider == "gemini":
        from litellm.llms.gemini.realtime.transformation import GeminiRealtimeConfig

        config = GeminiRealtimeConfig()
    else:
        credentials = options.get("vertex_credentials") or options.get("vertex_ai_credentials") or realtime.get_secret_str("VERTEXAI_CREDENTIALS")
        project = options.get("vertex_project") or options.get("vertex_ai_project") or litellm.vertex_project or realtime.get_secret_str("VERTEXAI_PROJECT")
        location = options.get("vertex_location") or options.get("vertex_ai_location") or litellm.vertex_location or realtime.get_secret_str("VERTEXAI_LOCATION")

        async def access_token():
            token, _ = await realtime._resolve_vertex_access_token_bounded(
                credentials=credentials, project_id=project,
                resolver=realtime.vertex_access_token_resolver,
                timeout_seconds=options["timeout"],
            )
            return token

        token, project = await realtime._resolve_vertex_access_token_bounded(
            credentials=credentials, project_id=project,
            resolver=realtime.vertex_access_token_resolver,
            timeout_seconds=options["timeout"],
        )
        config = realtime.vertex_realtime_config(
            model, access_token=token, resolve_access_token=access_token,
            project=project, location=location,
        )

    config.requires_session_configuration = lambda: False
    logging = realtime.LiteLLMLogging(
        model=model, messages=[], stream=True, call_type="arealtime",
        start_time=datetime.now(), litellm_call_id=str(uuid4()), function_id=str(uuid4()),
        kwargs=options, supports_correlation_logging=False,
    )
    logging.update_from_kwargs(
        kwargs=options, model=model, user=options.get("user"), optional_params={},
        litellm_params=options, custom_llm_provider=provider,
    )
    await realtime.base_llm_http_handler.async_realtime(
        model=model, websocket=options["websocket"], logging_obj=logging,
        provider_config=config, headers=options["headers"],
        api_base=base or options.get("api_base"), api_key=key or options.get("api_key"),
        timeout=options["timeout"], query_params=options.get("query_params"),
        litellm_metadata=realtime._build_litellm_metadata(options),
    )


class _BridgeSocket:
    """The server-facing side of LiteLLM's WebSocket protocol."""

    def __init__(self, connection, headers):
        self.connection = connection
        self.scope = {"headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()]}

    async def receive_text(self):
        # Provider handlers start reading client events after opening upstream.
        self.connection._ready.set()
        return await self.connection._outbound.get()

    async def send_text(self, data):
        event = json.loads(data)
        if not self.connection._ready.is_set() and event.get("type") == "error":
            self.connection._failure = RealtimeConnectionError(
                event.get("error", {}).get("message", "Realtime provider rejected the connection")
            )
        await self.connection._inbound.put(data)
        self.connection._ready.set()

    async def close(self, code=1000, reason=None):
        self.connection.close_code = code
        self.connection.close_reason = reason or ""


class ProviderConnection:
    """A socket-like client using LiteLLM's provider auth and event translation."""

    def __init__(self):
        self.close_code = None
        self.close_reason = ""
        self._inbound = asyncio.Queue(maxsize=128)
        self._outbound = asyncio.Queue(maxsize=128)
        self._ready = asyncio.Event()
        self._failure = None
        self._task = None
        self._google = False
        self._setup_sent = False

    @classmethod
    async def connect(cls, *, model, api_key=None, headers=None, timeout=30, options=None):
        bridge = getattr(litellm, "_arealtime", None)
        if bridge is None:
            raise RuntimeError("This LiteLLM version has no realtime bridge. Upgrade LiteLLM or supply a proxy URL.")
        connection = cls()
        connection._google = model.startswith(("gemini/", "vertex_ai/"))
        headers = dict(headers or {})
        socket = _BridgeSocket(connection, headers)
        connection._task = asyncio.create_task(connection._run(bridge, {
            **(options or {}), "model": model, "websocket": socket,
            "api_key": api_key, "headers": headers, "timeout": timeout,
        }))
        try:
            await asyncio.wait_for(connection._while_open(connection._ready.wait()), timeout=timeout)
            if connection._failure is not None:
                raise connection._failure
            if connection.close_code is not None:
                connection._raise_closed()
            return connection
        except BaseException:
            await connection.close()
            raise

    async def _run(self, bridge, options):
        try:
            await _run_bridge(bridge, options)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self._failure = error
            self.close_code = 1011
        finally:
            if self.close_code is None:
                self.close_code = 1000

    def _raise_closed(self):
        if self._failure is not None:
            raise RealtimeConnectionError(str(self._failure)) from self._failure
        raise RealtimeConnectionError(self.close_reason or "Realtime connection closed")

    async def _while_open(self, operation):
        """Wake blocked readers/writers when the provider ends or is cancelled."""
        pending = asyncio.create_task(operation)
        try:
            done, _ = await asyncio.wait((pending, self._task), return_when=asyncio.FIRST_COMPLETED)
            if pending in done:
                return pending.result()
            self._raise_closed()
        finally:
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)

    async def send(self, message):
        if self.close_code is not None:
            self._raise_closed()
        event = json.loads(message)
        if self._google:
            if event.get("type") == "session.update" and self._setup_sent:
                raise ValueError("Google Live accepts one initial session configuration. Start a new session to change it.")
            content = event.get("item", {}).get("content", [])
            if any(part.get("type") == "input_image" for part in content):
                raise ValueError("The installed LiteLLM Google realtime adapter does not translate image inputs; they were not sent.")
        await self._while_open(self._outbound.put(message))
        if event.get("type") == "session.update":
            self._setup_sent = True

    async def recv(self):
        # Preserve queued final events, even if the provider has already closed.
        if not self._inbound.empty():
            return self._inbound.get_nowait()
        if self.close_code is not None:
            self._raise_closed()
        return await self._while_open(self._inbound.get())

    async def close(self):
        if self.close_code is None:
            self.close_code = 1000
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def __aiter__(self):
        while True:
            try:
                message = await self.recv()
            except RealtimeConnectionError:
                if self.close_code == 1000 and self._failure is None:
                    return
                raise
            yield message
