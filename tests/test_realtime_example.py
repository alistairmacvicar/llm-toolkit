import asyncio
import base64
import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock


class Session:
    sample_rate = 24000

    def __init__(self, events=()):
        self.connect = mock.AsyncMock()
        self.close = mock.AsyncMock()
        self.update_session = mock.AsyncMock()
        self.send_text = mock.AsyncMock()
        self.receive_event = mock.AsyncMock(return_value={"type": "session.updated"})
        self.responses = list(events)

    async def events(self):
        for event in self.responses:
            yield event


class RealtimeExampleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        examples = Path(__file__).parents[1] / "examples"
        spec = importlib.util.spec_from_file_location("realtime_demo", examples / "realtime_chat.py")
        self.example = importlib.util.module_from_spec(spec)
        with mock.patch.object(sys, "path", [str(examples), *sys.path]):
            spec.loader.exec_module(self.example)

    async def run_demo(self, session, argv=()):
        output = io.StringIO()
        with mock.patch.object(self.example, "RealtimeSession", return_value=session) as factory, contextlib.redirect_stdout(output):
            result = await self.example.main(self.example.parse_args(["--message", "Hi", *argv]))
        return result, output.getvalue(), factory

    async def test_default_uses_model_routing_and_prints_streamed_response(self):
        session = Session([
            {"type": "response.output_text.delta", "delta": "Hello"},
            {"type": "response.done", "response": {"status": "completed"}},
        ])
        result, output, factory = await self.run_demo(session)
        self.assertEqual(result, 0)
        self.assertIsNone(factory.call_args.kwargs["url"])
        self.assertIsNone(factory.call_args.kwargs["api_key"])
        self.assertIn("Agent: Hello", output)
        session.send_text.assert_awaited_once_with("Hi")
        session.close.assert_awaited_once()

    async def test_proxy_is_explicit_and_uses_only_its_own_key(self):
        session = Session([{"type": "response.done"}])
        with mock.patch.dict(os.environ, {"LITELLM_API_KEY": "proxy-key", "OPENAI_API_KEY": "provider-key"}):
            result, _, factory = await self.run_demo(session, ["--url", "ws://localhost:4000/v1/realtime", "--model", "my-alias"])
        self.assertEqual(result, 0)
        self.assertEqual(factory.call_args.kwargs["api_key"], "proxy-key")
        self.assertEqual(factory.call_args.kwargs["model"], "my-alias")

    async def test_missing_proxy_reports_setup_without_traceback_or_retry(self):
        session = Session()
        session.connect.side_effect = ConnectionRefusedError("Connection refused")
        result, output, _ = await self.run_demo(session, ["--url", "ws://localhost:4000/v1/realtime"])
        self.assertEqual(result, 1)
        self.assertIn("Omit --url", output)
        self.assertNotIn("Traceback", output)
        session.connect.assert_awaited_once()
        session.send_text.assert_not_awaited()

    async def test_server_configuration_error_is_shown_before_sending_a_message(self):
        session = Session()
        session.receive_event.return_value = {"type": "error", "error": {"message": "Unsupported modality"}}
        result, output, _ = await self.run_demo(session)
        self.assertEqual(result, 1)
        self.assertIn("Unsupported modality", output)
        session.send_text.assert_not_awaited()
        session.close.assert_awaited_once()

    async def test_error_failed_response_and_remote_close_end_the_wait(self):
        for events in (
            [{"type": "error", "error": {"message": "Invalid model"}}],
            [{"type": "response.done", "response": {"status": "failed", "status_details": {"error": {"message": "Quota exceeded"}}}}],
            [],
        ):
            with self.subTest(events=events):
                session = Session(events)
                result, _, _ = await self.run_demo(session)
                self.assertEqual(result, 1)
                session.close.assert_awaited_once()

    async def test_silent_server_times_out_and_closes_without_retrying(self):
        session = Session()

        async def never():
            await asyncio.Event().wait()

        session.receive_event.side_effect = never
        result, output, _ = await self.run_demo(session, ["--timeout", "0.01"])
        self.assertEqual(result, 1)
        self.assertIn("Timed out", output)
        session.close.assert_awaited_once()

    async def test_audio_reply_is_saved_as_valid_wav_and_playback_is_optional(self):
        pcm = b"\x00\x00" * 480
        session = Session([
            {"type": "response.output_audio_transcript.delta", "delta": "Hello"},
            {"type": "response.output_audio.delta", "delta": base64.b64encode(pcm).decode()},
            {"type": "response.done", "response": {"status": "completed"}},
        ])
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(self.example, "AudioOutput") as playback:
            result, output, _ = await self.run_demo(session, ["--output", "audio", "--no-playback", "--output-dir", directory])
            files = list(Path(directory).glob("realtime-*.wav"))
            self.assertEqual(len(files), 1)
            with wave.open(str(files[0]), "rb") as wav:
                self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 24000))
                self.assertEqual(wav.readframes(wav.getnframes()), pcm)
            playback.assert_not_called()
        self.assertEqual(result, 0)
        self.assertIn("Hello", output)
        self.assertEqual(session.update_session.call_args.kwargs["output_modalities"], ["audio"])

    @unittest.skipIf(sys.platform == "win32", "POSIX terminal reader")
    async def test_waiting_for_terminal_input_keeps_background_tasks_running_and_handles_eof(self):
        read_fd, write_fd = os.pipe()
        try:
            with os.fdopen(read_fd) as stream, mock.patch.object(sys, "stdin", stream), contextlib.redirect_stdout(io.StringIO()):
                pending = asyncio.create_task(self.example.read_message())
                await asyncio.sleep(0)
                self.assertFalse(pending.done())
                os.write(write_fd, b"Hello from the terminal\n")
                self.assertEqual(await asyncio.wait_for(pending, timeout=1), "Hello from the terminal")
                os.close(write_fd)
                write_fd = None
                with self.assertRaises(EOFError):
                    await asyncio.wait_for(self.example.read_message(), timeout=1)
        finally:
            if write_fd is not None:
                os.close(write_fd)


if __name__ == "__main__":
    unittest.main()
