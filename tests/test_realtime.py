import asyncio
import base64
import json
import queue
import threading
import unittest

from llm_toolkit.agents.realtime import RealtimeSession


class RealtimeSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.websocket = FakeWebSocket()
        self.connection = None

        async def websocket_factory(url, **options):
            self.connection = {"url": url, **options}
            return self.websocket

        self.session = RealtimeSession(
            url="ws://localhost:4000/v1/realtime?guardrails=safety",
            model="realtime-model",
            api_key="test-key",
            websocket_factory=websocket_factory,
        )
        await self.session.connect()

    async def asyncTearDown(self):
        await self.session.close()

    async def test_connect_adds_model_and_authorization(self):
        self.assertEqual(
            self.connection["url"],
            "ws://localhost:4000/v1/realtime?guardrails=safety&model=realtime-model",
        )
        self.assertEqual(
            self.connection["additional_headers"],
            {"Authorization": "Bearer test-key"},
        )
        self.assertTrue(self.session.connected)

    async def test_session_update_and_text_message_use_realtime_events(self):
        await self.session.update_session(
            instructions="Be concise.",
            output_modalities=["text"],
        )
        await self.session.send_text("Hello")

        events = self.websocket.decoded_events()
        self.assertEqual(events[0]["type"], "session.update")
        self.assertEqual(events[0]["session"]["instructions"], "Be concise.")
        self.assertEqual(events[1]["type"], "conversation.item.create")
        self.assertEqual(
            events[1]["item"]["content"],
            [{"type": "input_text", "text": "Hello"}],
        )
        self.assertEqual(events[2]["type"], "response.create")

    async def test_confirmed_text_and_audio_create_one_response(self):
        await self.session.send_inputs(
            [
                {"type": "text", "value": "Listen to this"},
                {"type": "audio", "value": b"pcm audio"},
            ]
        )

        events = self.websocket.decoded_events()
        self.assertEqual(
            [event["type"] for event in events],
            [
                "conversation.item.create",
                "input_audio_buffer.append",
                "input_audio_buffer.commit",
                "response.create",
            ],
        )
        self.assertEqual(
            events[1]["audio"],
            base64.b64encode(b"pcm audio").decode("utf-8"),
        )

    async def test_receive_event_decodes_server_message(self):
        await self.websocket.incoming.put(
            json.dumps({"type": "response.text.delta", "delta": "Hello"})
        )

        event = await self.session.receive_event()

        self.assertEqual(event["delta"], "Hello")

    async def test_action_handler_callback_schedules_send_on_event_loop(self):
        send_handler = self.session.create_send_handler()
        scheduled_sends = queue.Queue()

        thread = threading.Thread(
            target=lambda: scheduled_sends.put(
                send_handler([{"type": "text", "value": "From keyboard"}])
            )
        )
        thread.start()
        thread.join()

        scheduled = scheduled_sends.get()
        await asyncio.wrap_future(scheduled)

        self.assertEqual(
            self.websocket.decoded_events()[0]["type"],
            "conversation.item.create",
        )


class FakeWebSocket:
    def __init__(self):
        self.sent = []
        self.incoming = asyncio.Queue()
        self.closed = False

    async def send(self, message):
        self.sent.append(message)

    async def recv(self):
        return await self.incoming.get()

    async def close(self):
        self.closed = True

    def decoded_events(self):
        return [json.loads(message) for message in self.sent]


if __name__ == "__main__":
    unittest.main()
