"""Bounded microphone capture and interruptible PCM playback for native apps."""

import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd


def choose_input_device(requested=None):
    """Prefer the Mac's built-in microphone without changing system settings."""
    if requested is not None:
        return requested
    if sys.platform == "darwin":
        for index, device in enumerate(sd.query_devices()):
            name = device["name"].lower()
            if device["max_input_channels"] and (
                "macbook" in name and "microphone" in name
                or "built-in microphone" in name
                or "imac microphone" in name
            ):
                return index
    return None


class LiveAudio:
    """Stream mono PCM16 at 24 kHz, with callbacks that never perform file/network I/O.

    Read ``inputs`` from the application loop; each entry contains PCM bytes and
    the monotonic capture time. ``play`` queues received audio, while ``interrupt``
    discards pending playback and reports how much of the current item was heard.
    Use headphones: this native transport does not provide echo cancellation.
    """

    sample_rate = 24000

    def __init__(self, input_device=None, output_device=None):
        self.input_device = choose_input_device(input_device)
        self.output_device = output_device
        self.input_info = sd.query_devices(self.input_device, "input")
        self.output_info = sd.query_devices(self.output_device, "output")
        self.inputs = queue.Queue(maxsize=100)  # At most two seconds of microphone audio.
        self.error = None
        self._input = self._output = None
        self._lock = threading.Lock()
        self._chunks = deque()
        self._scheduled = deque()
        self._item = None
        self._received = self._submitted = self._heard = self._queued = 0
        self._closed = False

    def start(self):
        if self._closed:
            raise RuntimeError("Live audio is closed")
        if self._input is not None:
            return
        settings = sd.CoreAudioSettings(change_device_parameters=False) if sys.platform == "darwin" else None
        try:
            self._output = sd.OutputStream(
                samplerate=self.sample_rate, device=self.output_device,
                channels=min(2, self.output_info["max_output_channels"]), dtype="float32",
                blocksize=480, latency="low", extra_settings=settings, callback=self._playback,
            )
            self._input = sd.InputStream(
                samplerate=self.sample_rate, device=self.input_device, channels=1, dtype="int16",
                blocksize=480, latency="low", extra_settings=settings, callback=self._capture,
            )
            self._output.start()
            self._input.start()
        except BaseException:
            self.close()
            raise

    def _capture(self, data, frames, timing, status):
        try:
            if status.input_overflow:
                raise RuntimeError("Microphone audio overflowed; stopping to avoid gaps in the recording")
            captured_at = time.monotonic() - max(0, timing.currentTime - timing.inputBufferAdcTime)
            self.inputs.put_nowait((data.astype("<i2", copy=False).tobytes(), captured_at))
        except Exception as error:
            self.error = error if not isinstance(error, queue.Full) else RuntimeError(
                "The connection cannot keep up with the microphone; stopping capture"
            )
            raise sd.CallbackAbort from error

    def play(self, item, pcm):
        """Queue one audio delta, identified by (item_id, content_index)."""
        if len(pcm) % 2:
            raise ValueError("Playback requires complete PCM16 samples")
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
        with self._lock:
            if item != self._item:
                self._reset()
                self._item = item
            if self._queued + len(samples) > self.sample_rate * 30:
                raise RuntimeError("Agent playback exceeded the 30-second buffer limit")
            if len(samples):
                self._chunks.append(samples)
                self._received += len(samples)
                self._queued += len(samples)

    def _update_heard(self, now):
        while self._scheduled:
            start, offset, count = self._scheduled[0]
            played = min(count, max(0, int((now - start) * self.sample_rate)))
            self._heard = max(self._heard, offset + played if now >= start else self._heard)
            if played < count:
                break
            self._scheduled.popleft()

    def _playback(self, output, frames, timing, status):
        output.fill(0)
        if not self._lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            self._update_heard(now)
            starts = now + max(0, timing.outputBufferDacTime - timing.currentTime)
            written = 0
            while self._chunks and written < frames:
                chunk = self._chunks[0]
                count = min(len(chunk), frames - written)
                output[written:written + count] = chunk[:count, None]
                self._scheduled.append((starts + written / self.sample_rate, self._submitted, count))
                self._submitted += count
                self._queued -= count
                written += count
                if count == len(chunk):
                    self._chunks.popleft()
                else:
                    self._chunks[0] = chunk[count:]
        except Exception as error:
            self.error = error
            raise sd.CallbackAbort from error
        finally:
            self._lock.release()

    def interrupt(self):
        """Return (item, heard_milliseconds), excluding queued and device-buffered audio."""
        with self._lock:
            self._update_heard(time.monotonic())
            result = None if self._item is None else (
                self._item, min(self._heard, self._received) * 1000 // self.sample_rate,
            )
            self._reset()
            return result

    def _reset(self):
        self._chunks.clear()
        self._scheduled.clear()
        self._item = None
        self._received = self._submitted = self._heard = self._queued = 0

    def check(self):
        if self.error is not None:
            raise self.error
        if self._input is not None and not self._closed and not self._input.active:
            raise RuntimeError("The microphone stopped unexpectedly")
        if self._output is not None and not self._closed and not self._output.active:
            raise RuntimeError("Audio playback stopped unexpectedly")

    def close(self):
        """Release both devices, including when startup or another close fails."""
        self._closed = True
        self.interrupt()
        error = None
        for name in ("_input", "_output"):
            stream = getattr(self, name)
            setattr(self, name, None)
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    error = error or exc
        if error is not None:
            raise error
