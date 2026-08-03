import queue
import unittest

from llm_toolkit.util.input_buffer import InputBuffer


class InputBufferTests(unittest.TestCase):
    def test_confirm_sends_all_staged_inputs(self):
        sent = []
        input_buffer = InputBuffer(send_handler=sent.append)

        input_buffer.stage("text", "hello")
        input_buffer.stage("image", "photo.jpg")
        confirmed = input_buffer.confirm()

        self.assertEqual(
            confirmed,
            [
                {"type": "text", "value": "hello"},
                {"type": "image", "value": "photo.jpg"},
            ],
        )
        self.assertEqual(sent, [confirmed])
        self.assertEqual(input_buffer.get_pending(), [])

    def test_new_capture_replaces_pending_input_of_the_same_type(self):
        input_buffer = InputBuffer()

        input_buffer.stage("audio", "first.wav")
        input_buffer.stage("audio", "second.wav")

        self.assertEqual(
            input_buffer.get_pending(),
            [{"type": "audio", "value": "second.wav"}],
        )

    def test_discard_clears_pending_inputs(self):
        input_buffer = InputBuffer()
        input_buffer.stage("video", "capture.mp4")

        discarded = input_buffer.discard()

        self.assertEqual(discarded, [{"type": "video", "value": "capture.mp4"}])
        self.assertEqual(input_buffer.get_pending(), [])

    def test_auto_send_dispatches_completed_capture(self):
        sent = []
        input_buffer = InputBuffer(auto_send=True, send_handler=sent.append)

        staged = input_buffer.stage_many(
            [
                {"type": "audio", "value": "capture.wav"},
                {"type": "video", "value": "capture.mp4"},
            ]
        )

        self.assertEqual(sent, [staged])
        self.assertEqual(input_buffer.get_pending(), [])
        self.assertEqual(input_buffer.get_event()["type"], "send")

    def test_empty_confirm_does_not_queue_an_event(self):
        input_buffer = InputBuffer()

        self.assertEqual(input_buffer.confirm(), [])
        with self.assertRaises(queue.Empty):
            input_buffer.get_event(block=False)

    def test_enabling_auto_send_confirms_existing_inputs(self):
        sent = []
        input_buffer = InputBuffer(send_handler=sent.append)
        input_buffer.stage("text", "ready")

        input_buffer.set_auto_send(True)

        self.assertEqual(sent, [[{"type": "text", "value": "ready"}]])
        self.assertEqual(input_buffer.get_pending(), [])


if __name__ == "__main__":
    unittest.main()
