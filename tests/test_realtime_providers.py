"""Exercise installed LiteLLM routing/translations without contacting providers."""

import asyncio
import json
import os
import unittest
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import litellm

from llm_toolkit import RealtimeSession
from llm_toolkit.agents.realtime_transport import RealtimeConnectionError


class Backend:
    def __init__(self, *, gemini=False):
        self.gemini = gemini
        self.incoming = asyncio.Queue()
        self.sent = []
        self.closed = False
        if not gemini:
            self.emit({"type": "session.created", "session": {"id": "test-session"}})

    def emit(self, event):
        self.incoming.put_nowait(json.dumps(event))

    async def send(self, message):
        event = json.loads(message)
        self.sent.append(event)
        if self.gemini:
            if "setup" in event:
                self.emit({"setupComplete": {}})
            elif "clientContent" in event:
                self.emit({"serverContent": {"modelTurn": {"parts": [{"text": "Hello from Gemini"}]}}})
                self.emit({"serverContent": {"turnComplete": True}})
        elif event.get("type") == "session.update":
            self.emit({"type": "session.updated", "session": event["session"]})
        elif event.get("type") == "response.create":
            self.emit({"type": "response.output_text.delta", "delta": "Hello"})
            self.emit({"type": "response.done", "response": {"status": "completed", "output": []}})

    async def recv(self, decode=None):
        return await self.incoming.get()

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()


class ConnectResult:
    def __init__(self, backend):
        self.backend = backend

    def __await__(self):
        async def connected():
            return self.backend
        return connected().__await__()

    async def __aenter__(self):
        return await self.backend.__aenter__()

    async def __aexit__(self, *args):
        await self.backend.__aexit__(*args)


class ProviderRoutingTests(unittest.IsolatedAsyncioTestCase):
    async def collect_reply(self, session):
        events = []
        async with asyncio.timeout(2):
            async for event in session.events():
                events.append(event)
                if event.get("type") == "response.done":
                    return events
        self.fail("No completed reply")

    async def test_openai_azure_and_xai_resolve_models_credentials_and_endpoints(self):
        cases = [
            ("openai/gpt-realtime", {"OPENAI_API_KEY": "openai-test"}, "api.openai.com", "/v1/realtime", "Authorization", "Bearer openai-test"),
            ("azure/my-deployment", {"AZURE_API_KEY": "azure-test", "AZURE_API_BASE": "https://example.openai.azure.com"}, "example.openai.azure.com", "/openai/v1/realtime", "api-key", "azure-test"),
            ("xai/grok-voice-latest", {"XAI_API_KEY": "xai-test"}, "api.x.ai", "/v1/realtime", "Authorization", "Bearer xai-test"),
        ]
        for model, environment, host, path, header, credential in cases:
            with self.subTest(model=model):
                backend = Backend()
                with mock.patch.dict(os.environ, environment, clear=True), mock.patch(
                    "websockets.connect", return_value=ConnectResult(backend)
                ) as connect:
                    async with RealtimeSession(model=model, connect_timeout=2) as session:
                        self.assertTrue(session.connected)
                        await session.update_session(instructions="Be brief", output_modalities=["text"])
                        await session.send_text("Hi")
                        events = await self.collect_reply(session)
                url = urlsplit(connect.call_args.args[0])
                self.assertEqual((url.hostname, url.path), (host, path))
                self.assertEqual(parse_qs(url.query)["model"], [model.split("/", 1)[1]])
                self.assertEqual(connect.call_args.kwargs["additional_headers"][header], credential)
                self.assertTrue(any(event.get("delta") == "Hello" for event in events))
                self.assertTrue(backend.closed)
                self.assertFalse(session.connected)

    async def test_gemini_translates_session_text_media_and_returned_events(self):
        backend = Backend(gemini=True)
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": "gemini-test"}, clear=True), mock.patch(
            "websockets.connect", return_value=ConnectResult(backend)
        ) as connect:
            async with RealtimeSession(model="gemini/gemini-2.0-flash-live-001", connect_timeout=2) as session:
                await session.update_session(instructions="Be brief", output_modalities=["text"])
                await session.append_audio(b"\x00\x00" * 10)
                await session.send_text("Hi Gemini")
                events = await self.collect_reply(session)
                with self.assertRaisesRegex(ValueError, "image inputs"):
                    await session.send_image("data:image/jpeg;base64,aW1hZ2U=")
                with self.assertRaisesRegex(ValueError, "initial session configuration"):
                    await session.update_session(instructions="A second setup must not be silently ignored")
        url = urlsplit(connect.call_args.args[0])
        self.assertEqual(url.hostname, "generativelanguage.googleapis.com")
        self.assertEqual(parse_qs(url.query)["key"], ["gemini-test"])
        setup = next(event["setup"] for event in backend.sent if "setup" in event)
        self.assertEqual(setup["model"], "models/gemini-2.0-flash-live-001")
        self.assertEqual(setup["systemInstruction"]["parts"], [{"text": "Be brief"}])
        self.assertTrue(any("realtimeInput" in event for event in backend.sent))
        self.assertTrue(any("Hi Gemini" in json.dumps(event.get("clientContent")) for event in backend.sent))
        self.assertTrue(any(event.get("delta") == "Hello from Gemini" for event in events))
        self.assertTrue(any(event.get("type") == "session.updated" for event in events))
        self.assertTrue(backend.closed)

    async def test_vertex_uses_its_credentials_and_preserves_initial_instructions(self):
        from litellm.realtime_api import main as realtime

        backend = Backend(gemini=True)
        with mock.patch.object(realtime, "_resolve_vertex_access_token_bounded", return_value=("vertex-token", "demo-project")), mock.patch(
            "websockets.connect", return_value=ConnectResult(backend)
        ) as connect:
            async with RealtimeSession(
                model="vertex_ai/gemini-2.0-flash-live-001",
                provider_options={"vertex_project": "demo-project", "vertex_location": "us-central1"},
                connect_timeout=2,
            ) as session:
                await session.update_session(instructions="Keep this instruction", output_modalities=["text"])
                await session.send_text("Hello")
                await self.collect_reply(session)
        self.assertIn("aiplatform.googleapis.com", connect.call_args.args[0])
        self.assertEqual(connect.call_args.kwargs["additional_headers"]["Authorization"], "Bearer vertex-token")
        setup = next(event["setup"] for event in backend.sent if "setup" in event)
        self.assertIn("Keep this instruction", json.dumps(setup))
        self.assertTrue(backend.closed)

    async def test_bedrock_cloud_options_are_routed_without_a_proxy(self):
        from litellm.realtime_api import main as realtime

        async def handler(**kwargs):
            await kwargs["websocket"].receive_text()

        with mock.patch.object(realtime.bedrock_realtime, "async_realtime", side_effect=handler) as backend:
            async with RealtimeSession(
                model="bedrock/amazon.nova-sonic-v1:0",
                provider_options={"aws_region_name": "us-east-1", "aws_profile_name": "demo"},
            ):
                self.assertEqual(backend.call_args.kwargs["model"], "amazon.nova-sonic-v1:0")
                self.assertEqual(backend.call_args.kwargs["aws_profile_name"], "demo")
                self.assertEqual(backend.call_args.kwargs["aws_region_name"], "us-east-1")

    async def test_provider_options_reach_litellm_and_do_not_mutate_callers_settings(self):
        options = {"api_base": "https://custom.example", "metadata": {"tag": "demo"}}
        observed = {}

        async def bridge(**kwargs):
            observed.update(kwargs)
            kwargs["metadata"]["tag"] = "changed"
            await kwargs["websocket"].receive_text()

        with mock.patch.object(litellm, "_arealtime", side_effect=bridge):
            async with RealtimeSession(model="test/voice", provider_options=options, api_key="explicit"):
                self.assertEqual(observed["api_base"], options["api_base"])
                self.assertEqual(observed["api_key"], "explicit")
                self.assertEqual(options["metadata"]["tag"], "demo")


class ProviderLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_startup_failure_reaches_caller_and_leaves_no_connection(self):
        with mock.patch.object(litellm, "_arealtime", side_effect=ValueError("Missing provider key")):
            session = RealtimeSession(model="test/voice")
            with self.assertRaisesRegex(RealtimeConnectionError, "Missing provider key"):
                await session.connect()
        self.assertFalse(session.connected)

    async def test_timeout_and_cancellation_close_the_upstream_task(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                started, stopped = asyncio.Event(), asyncio.Event()

                async def bridge(**kwargs):
                    started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        stopped.set()

                with mock.patch.object(litellm, "_arealtime", side_effect=bridge):
                    session = RealtimeSession(model="test/voice", connect_timeout=0.05 if not cancel else 30)
                    task = asyncio.create_task(session.connect())
                    await started.wait()
                    if cancel:
                        task.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancel else TimeoutError):
                        await task
                    self.assertTrue(stopped.is_set())
                    self.assertFalse(session.connected)

    async def test_remote_close_keeps_queued_final_event_and_reconnects_explicitly(self):
        async def bridge(**kwargs):
            socket = kwargs["websocket"]
            await socket.send_text(json.dumps({"type": "session.created"}))
            await socket.receive_text()
            await socket.send_text(json.dumps({"type": "response.done"}))
            await socket.close()

        with mock.patch.object(litellm, "_arealtime", side_effect=bridge) as start:
            async with RealtimeSession(model="test/voice") as session:
                await session.send_event({"type": "finish"})
                events = [event async for event in session.events()]
                self.assertEqual([event["type"] for event in events], ["session.created", "response.done"])
                self.assertFalse(session.connected)
                self.assertEqual(start.call_count, 1)
                await session.connect()
                self.assertEqual(start.call_count, 2)

    async def test_close_wakes_blocked_reader_and_writer_without_background_tasks(self):
        async def bridge(**kwargs):
            await kwargs["websocket"].send_text(json.dumps({"type": "session.created"}))
            await asyncio.Event().wait()

        with mock.patch.object(litellm, "_arealtime", side_effect=bridge):
            session = RealtimeSession(model="test/voice")
            await session.connect()
            await session.receive_event()
            for _ in range(128):
                await session.send_event({"type": "test"})
            reader = asyncio.create_task(session.receive_event())
            writer = asyncio.create_task(session.send_event({"type": "overflow"}))
            await asyncio.sleep(0)
            await session.close()
            for pending in (reader, writer):
                with self.assertRaises(RealtimeConnectionError):
                    await asyncio.wait_for(pending, timeout=1)


if __name__ == "__main__":
    unittest.main()
