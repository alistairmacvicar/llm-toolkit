import unittest
from unittest.mock import patch

import numpy as np

from llm_toolkit.media.audio_output import AudioOutput


class AudioOutputTests(unittest.TestCase):
    def test_microphone_resolves_explicit_and_default_devices(self):
        from llm_toolkit.media.microphone_source import MicrophoneSource
        source = MicrophoneSource.__new__(MicrophoneSource)
        with patch("llm_toolkit.media.microphone_source.sd.query_devices", return_value={"name": "Input", "max_input_channels": 1}) as query:
            with patch("llm_toolkit.media.microphone_source.sd.default.device", (4, 6)):
                self.assertEqual(source._get_input_source()["index"], 4)
                query.assert_called_with(device=4, kind="input")
            self.assertEqual(source._get_input_source(2)["index"], 2)
            query.assert_called_with(device=2, kind="input")

    def test_channels_order_and_drain_before_idle(self):
        events, chunks = [], []
        class Stream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
            def __enter__(self):
                return self
            def __exit__(self, *args):
                events.append("closed")
            def write(self, value):
                chunks.append(value.copy())
            def stop(self):
                events.append("drained")
        output = AudioOutput(device=7, channels=3, stream_factory=Stream)
        output.play_pcm([b"\x00\x40", b"\x00\x20"], channel=2, on_state=events.append)
        self.assertEqual(events, [True, "drained", "closed", False])
        np.testing.assert_allclose(chunks[0], [[0, 0, 0.5]])
        np.testing.assert_allclose(chunks[1], [[0, 0, 0.25]])
        with self.assertRaises(ValueError):
            output.play_pcm([], channel=3)

    def test_playback_failure_clears_talking_state(self):
        events = []
        class Broken:
            def __init__(self, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def write(self, value):
                raise OSError("Device removed")
        with self.assertRaises(OSError):
            AudioOutput(stream_factory=Broken).play_pcm([b"\0\0"], on_state=events.append)
        self.assertEqual(events, [True, False])
