from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from llm_toolkit.media import live_audio


class Stream:
    def __init__(self, **kwargs):
        self.options = kwargs
        self.active = False
        self.closed = False

    def start(self):
        self.active = True

    def close(self):
        self.closed = True
        self.active = False


@pytest.fixture
def audio():
    def device(requested, kind):
        return {"name": kind, "max_output_channels": 2, "max_input_channels": 1}

    with patch.object(live_audio.sys, "platform", "linux"), patch.object(
        live_audio.sd, "query_devices", side_effect=device
    ), patch.object(live_audio.sd, "InputStream", side_effect=Stream), patch.object(
        live_audio.sd, "OutputStream", side_effect=Stream
    ):
        instance = live_audio.LiveAudio(input_device=1, output_device=2)
        try:
            yield instance
        finally:
            instance.close()


def test_macos_prefers_builtin_input_without_overriding_explicit_device():
    devices = [
        {"name": "AirPods Pro", "max_input_channels": 1},
        {"name": "MacBook Air Microphone", "max_input_channels": 1},
    ]
    with patch.object(live_audio.sys, "platform", "darwin"), patch.object(live_audio.sd, "query_devices", return_value=devices):
        assert live_audio.choose_input_device() == 1
        assert live_audio.choose_input_device("External Mic") == "External Mic"
        assert live_audio.choose_input_device(0) == 0


def test_capture_keeps_exact_pcm_and_adc_timestamp_without_opening_devices_early(audio):
    assert audio._input is None and audio._output is None
    audio.start()
    assert audio._input.options["device"] == 1
    assert audio._output.options["device"] == 2
    pcm = np.array([[-32768], [0], [32767]], dtype="<i2")
    timing = SimpleNamespace(currentTime=10, inputBufferAdcTime=9.875)
    with patch.object(live_audio.time, "monotonic", return_value=100):
        audio._capture(pcm, 3, timing, SimpleNamespace(input_overflow=False))
    assert audio.inputs.get_nowait() == (pcm.tobytes(), 99.875)


def test_interrupt_counts_only_heard_audio_and_discards_queued_samples(audio):
    pcm = np.full(960, 16384, dtype="<i2").tobytes()  # 40ms, only half submitted.
    audio.play(("reply-1", 0), pcm)
    output = np.zeros((480, 2), dtype=np.float32)
    timing = SimpleNamespace(currentTime=10, outputBufferDacTime=10.125)
    with patch.object(live_audio.time, "monotonic", return_value=100):
        audio._playback(output, 480, timing, None)
    np.testing.assert_array_equal(output, 0.5)
    with patch.object(live_audio.time, "monotonic", return_value=100.1328125):
        assert audio.interrupt() == (("reply-1", 0), 7)
        audio._playback(output, 480, timing, None)
    assert not output.any()
    assert audio.interrupt() is None


def test_interruption_before_device_playback_reports_zero(audio):
    audio.play(("reply", 0), b"\x01\x00" * 480)
    timing = SimpleNamespace(currentTime=10, outputBufferDacTime=11)
    with patch.object(live_audio.time, "monotonic", return_value=100):
        audio._playback(np.zeros((480, 2)), 480, timing, None)
        assert audio.interrupt() == (("reply", 0), 0)


def test_capture_backlog_stops_instead_of_silently_dropping_sent_audio(audio):
    for _ in range(audio.inputs.maxsize):
        audio.inputs.put((b"audio", 0))
    with pytest.raises(live_audio.sd.CallbackAbort):
        audio._capture(np.zeros((480, 1), dtype="<i2"), 480,
                       SimpleNamespace(currentTime=10, inputBufferAdcTime=10), SimpleNamespace(input_overflow=False))
    with pytest.raises(RuntimeError, match="cannot keep up"):
        audio.check()


def test_playback_buffer_is_bounded_and_rejects_partial_samples(audio):
    with pytest.raises(ValueError, match="complete PCM16"):
        audio.play(("reply", 0), b"\x01")
    with pytest.raises(RuntimeError, match="buffer limit"):
        audio.play(("reply", 0), b"\x00\x00" * (24000 * 30 + 1))


def test_failed_start_releases_already_opened_output(audio):
    output = Stream()
    with patch.object(live_audio.sd, "OutputStream", return_value=output), patch.object(
        live_audio.sd, "InputStream", side_effect=RuntimeError("No microphone")
    ), pytest.raises(RuntimeError, match="No microphone"):
        audio.start()
    assert output.closed


def test_close_releases_both_devices_even_if_input_close_raises(audio):
    audio.start()
    output = audio._output
    audio._input.close = Mock(side_effect=RuntimeError("Device disconnected"))
    with pytest.raises(RuntimeError, match="Device disconnected"):
        audio.close()
    assert output.closed
    assert audio._input is None and audio._output is None
    audio.close()
