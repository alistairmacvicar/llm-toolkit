import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from llm_toolkit import RealtimeSession


class WebRTCTests(unittest.TestCase):
    def test_answer_uses_server_key_and_configures_interruptible_voice(self):
        response = SimpleNamespace(is_success=True, text="v=0\r\nanswer")
        with patch("httpx.post", return_value=response) as post:
            answer = RealtimeSession.create_openai_webrtc_answer(
                "v=0\r\noffer", model="voice-model", instructions="Play Mira.",
                voice="coral", api_key="secret-server-key",
            )
        self.assertEqual(answer, response.text)
        request = post.call_args.kwargs
        self.assertEqual(request["headers"]["Authorization"], "Bearer secret-server-key")
        self.assertNotIn("secret-server-key", answer)
        config = json.loads(request["files"]["session"][1])
        self.assertEqual(config["model"], "voice-model")
        self.assertEqual(config["instructions"], "Play Mira.")
        self.assertTrue(config["audio"]["input"]["turn_detection"]["interrupt_response"])
        self.assertEqual(config["audio"]["output"]["voice"], "coral")

    def test_provider_error_does_not_expose_credentials_or_response_body(self):
        response = SimpleNamespace(is_success=False, status_code=401, text="sensitive provider body")
        with patch("httpx.post", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as caught:
                RealtimeSession.create_openai_webrtc_answer("v=0\r\n", model="voice-model", api_key="secret")
        self.assertNotIn("sensitive", str(caught.exception))
        self.assertNotIn("secret", str(caught.exception))

    def test_missing_key_and_invalid_offer_make_no_request(self):
        with patch.dict("os.environ", {}, clear=True), patch("httpx.post") as post:
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                RealtimeSession.create_openai_webrtc_answer("v=0\r\n", model="voice-model")
            with self.assertRaisesRegex(ValueError, "SDP"):
                RealtimeSession.create_openai_webrtc_answer("bad", model="voice-model", api_key="test")
        post.assert_not_called()

    def test_invalid_answer_is_rejected(self):
        with patch("httpx.post", return_value=SimpleNamespace(is_success=True, text="unexpected")):
            with self.assertRaisesRegex(RuntimeError, "SDP answer"):
                RealtimeSession.create_openai_webrtc_answer("v=0\r\n", model="voice-model", api_key="test")
