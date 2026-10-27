from unittest.mock import Mock, patch

import numpy as np
import pytest

from llm_toolkit.media import microphone_source as microphone


class Stream:
    def __init__(self, **kwargs):
        self.callback = kwargs["callback"]
        self.active = False
        self.closed = False
        self.starts = 0

    def start(self):
        self.active = True
        self.starts += 1

    def stop(self):
        self.active = False

    def close(self):
        self.active = False
        self.closed = True


@pytest.fixture
def source():
    with patch.object(microphone.sd, "InputStream", side_effect=Stream), patch.object(
        microphone.sd, "OutputStream", side_effect=Stream
    ) as output_factory, patch.object(microphone.MicrophoneSource, "_get_input_source", return_value={"index": 7}), patch.object(
        microphone.sd, "play", side_effect=AssertionError("No convenience playback in callbacks")
    ):
        source = microphone.MicrophoneSource(monitor_device=8, monitor_buffer_seconds=4 / 24000)
        source.output_factory = output_factory
        try:
            yield source
        finally:
            source.quit()


def capture(source, samples):
    data = np.asarray(samples, dtype=np.float32).reshape(-1, 1)
    source._callback(data, len(data), None, None)


def output(source, count):
    data = np.full((count, 1), -99, dtype=np.float32)
    source._monitor_callback(data, count, None, None)
    return data[:, 0]


def test_monitor_reuses_stream_and_handles_different_block_sizes(source):
    source.output_factory.assert_not_called()
    assert source.toggle_monitor()
    capture(source, [1, 2, 3])
    np.testing.assert_array_equal(output(source, 2), [1, 2])
    capture(source, [4, 5])
    np.testing.assert_array_equal(output(source, 5), [3, 4, 5, 0, 0])
    stream = source._output_stream
    assert not source.toggle_monitor()
    assert not source._input_stream.active
    assert source.toggle_monitor()
    assert source._output_stream is stream
    source.output_factory.assert_called_once()
    assert source.output_factory.call_args.kwargs["device"] == 8
    np.testing.assert_array_equal(output(source, 2), [0, 0])


def test_monitor_overflow_keeps_newest_audio_with_fixed_storage(source):
    source.toggle_monitor()
    storage = source._monitor_buffer.data
    capture(source, [1, 2, 3])
    capture(source, [4, 5, 6])
    np.testing.assert_array_equal(output(source, 4), [3, 4, 5, 6])
    capture(source, list(range(100)))
    np.testing.assert_array_equal(output(source, 4), [96, 97, 98, 99])
    assert source._monitor_buffer.data is storage
    assert storage.nbytes == 16


def test_monitor_callbacks_do_not_wait_for_buffer_lock(source):
    source.toggle_monitor()
    with source._monitor_buffer.lock:
        capture(source, [1, 2])
        np.testing.assert_array_equal(output(source, 3), [0, 0, 0])


def test_recording_continues_when_monitor_stops(source):
    source.toggle_monitor()
    source.start_recording()
    capture(source, [1, 2])
    source.toggle_monitor()
    assert source._input_stream.active
    assert not source._output_stream.active
    capture(source, [3, 4])
    recording = source.stop_recording()
    np.testing.assert_array_equal(recording[:, 0], [1, 2, 3, 4])
    assert not source._input_stream.active


def test_monitor_continues_when_recording_stops_and_next_recording_is_clean(source):
    source.toggle_monitor()
    source.start_recording()
    capture(source, [1, 2])
    source.start_recording()  # Idempotent during capture.
    np.testing.assert_array_equal(source.stop_recording()[:, 0], [1, 2])
    assert source._input_stream.active
    capture(source, [3, 4])
    source.start_recording()
    capture(source, [5])
    np.testing.assert_array_equal(source.stop_recording()[:, 0], [5])


def test_failed_output_open_leaves_recording_usable(source):
    source.start_recording()
    source.output_factory.side_effect = RuntimeError("no output")
    with pytest.raises(RuntimeError, match="no output"):
        source.toggle_monitor()
    assert not source._monitor_input
    assert source.is_recording and source._input_stream.active
    capture(source, [1])
    np.testing.assert_array_equal(source.stop_recording()[:, 0], [1])


def test_failed_input_start_closes_monitor_and_allows_retry(source):
    start = source._input_stream.start
    source._input_stream.start = Mock(side_effect=RuntimeError("no input"))
    stream = Stream(callback=source._monitor_callback)
    source.output_factory.side_effect = None
    source.output_factory.return_value = stream
    with pytest.raises(RuntimeError, match="no input"):
        source.toggle_monitor()
    assert stream.closed
    assert source._output_stream is None
    assert not source._monitor_input
    source._input_stream.start = start
    source.output_factory.side_effect = Stream
    assert source.toggle_monitor()


def test_failed_recording_start_restores_recording_state(source):
    source._input_stream.start = Mock(side_effect=RuntimeError("no input"))
    with pytest.raises(RuntimeError, match="no input"):
        source.start_recording()
    assert not source.is_recording


def test_failed_recording_stop_retains_audio_without_leaking_into_next_recording(source):
    source.start_recording()
    capture(source, [1, 2])
    stop = source._input_stream.stop
    source._input_stream.stop = Mock(side_effect=RuntimeError("stop failed"))
    with pytest.raises(RuntimeError, match="stop failed"):
        source.stop_recording()
    np.testing.assert_array_equal(source.last_recording[:, 0], [1, 2])
    assert not source.is_recording
    source._input_stream.stop = stop
    source.start_recording()
    capture(source, [3])
    np.testing.assert_array_equal(source.stop_recording()[:, 0], [3])


def test_quit_closes_both_streams_and_thread_even_if_output_close_fails(source):
    source.toggle_monitor()
    source._output_stream.close = Mock(side_effect=RuntimeError("close failed"))
    with pytest.raises(RuntimeError, match="close failed"):
        source.quit()
    assert source._input_stream.closed
    assert not source._microphone_thread.is_alive()
    source.quit()
    with pytest.raises(RuntimeError, match="closed"):
        source.toggle_monitor()
    with pytest.raises(RuntimeError, match="closed"):
        source.start_recording()


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_buffer_duration_never_opens_hardware(seconds):
    with patch.object(microphone.sd, "InputStream") as stream, pytest.raises(ValueError):
        microphone.MicrophoneSource(monitor_buffer_seconds=seconds)
    stream.assert_not_called()
