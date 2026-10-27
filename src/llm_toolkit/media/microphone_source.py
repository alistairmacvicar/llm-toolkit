import threading
import time
import math
from datetime import datetime
from pathlib import Path
from typing import final
import queue

import sounddevice as sd
import numpy as np
import soundfile as sf

from llm_toolkit.logging import get_logger

logger = get_logger(__name__)


class _MonitorBuffer:
    """Bounded PCM ring. Audio callbacks never wait for the other callback."""

    def __init__(self, capacity: int, channels: int):
        self.data = np.empty((capacity, channels), dtype=np.float32)
        self.lock = threading.Lock()
        self.start = 0
        self.size = 0

    def clear(self):
        with self.lock:
            self.start = self.size = 0

    def write(self, data):
        if not self.lock.acquire(blocking=False):
            return
        try:
            capacity = len(self.data)
            data = data[-capacity:]
            count = len(data)
            discarded = max(0, self.size + count - capacity)
            self.start = (self.start + discarded) % capacity
            self.size -= discarded
            end = (self.start + self.size) % capacity
            first = min(count, capacity - end)
            self.data[end:end + first] = data[:first]
            self.data[:count - first] = data[first:]
            self.size += count
        finally:
            self.lock.release()

    def read_into(self, output):
        output.fill(0)
        if not self.lock.acquire(blocking=False):
            return
        try:
            count = min(len(output), self.size)
            first = min(count, len(self.data) - self.start)
            output[:first] = self.data[self.start:self.start + first]
            output[first:count] = self.data[:count - first]
            self.start = (self.start + count) % len(self.data)
            self.size -= count
        finally:
            self.lock.release()


@final
class MicrophoneSource:
    """Capture mono audio with optional monitoring through a bounded PCM buffer.

    Monitoring drops old samples on overflow and emits silence on underflow.
    Completed recordings are kept in memory until the next recording starts.
    """

    def __init__(self, source: int = -1, save: bool = False, save_dir: str = "recordings",
                 *, monitor_device: int | None = None, monitor_buffer_seconds: float = 0.25):
        """Open an input source without starting capture.

        Args:
            source (int): Input device index, or -1 for the system default.
            save (bool): Save completed recordings as WAV files.
            save_dir (str): Recording output directory.
            monitor_device (int | None): Output device index, or None for the default.
            monitor_buffer_seconds (float): Maximum buffered monitor audio in seconds.
        """
        if not math.isfinite(monitor_buffer_seconds) or monitor_buffer_seconds <= 0:
            raise ValueError("monitor_buffer_seconds must be finite and greater than 0")
        self.SAMPLE_RATE = 24000  # https://developers.openai.com/api/docs/guides/realtime-transcription
        self.INPUT_CHANNELS = 1  # Mono for transcriptions should work on all devices
        self._audio_queue = queue.SimpleQueue()
        self._audio_data = []
        self._save = save
        self._save_dir = Path(save_dir)
        self.last_recording = None
        self.last_recording_path = None

        self._monitor_input = False
        self._monitor_device = monitor_device
        self._monitor_buffer = _MonitorBuffer(
            max(1, round(self.SAMPLE_RATE * monitor_buffer_seconds)), self.INPUT_CHANNELS)
        self._output_stream = None
        self._lock = threading.Lock()
        self._recording_lock = threading.Lock()
        self.running = True
        self.status = None
        self._recording = False
        self._input_device = self._get_input_source(source=source)
        self._input_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            device=self._input_device["index"],
            channels=self.INPUT_CHANNELS,
            dtype="float32",
            callback=self._callback,
        )

        self._microphone_thread = threading.Thread(target=self._listen)
        self._microphone_thread.daemon = True
        self._microphone_thread.start()
        logger.info(
            "Microphone source %s started at %s Hz",
            self._input_device["index"],
            self.SAMPLE_RATE,
        )

    def _listen(self):
        while self.running:
            with self._lock:
                if self._recording:
                    self._drain_audio_queue()
            time.sleep(0.01)

    def _drain_audio_queue(self):
        while True:
            try:
                self._audio_data.append(self._audio_queue.get_nowait())
            except queue.Empty:
                break

    def _get_input_source(self, source: int = -1):
        # Return the default input device set in the OS if the user hasn't chosen a specific one
        index = int(sd.default.device[0]) if source == -1 else source
        device = dict(sd.query_devices(device=index, kind="input"))
        device["index"] = index
        return device

    def toggle_monitor(self):
        with self._lock:
            if not self.running:
                raise RuntimeError("Microphone source is closed")
            if self._monitor_input:
                self._monitor_input = False
                try:
                    self._output_stream.stop()
                finally:
                    if not self._recording:
                        self._input_stream.stop()
            else:
                self._monitor_buffer.clear()
                try:
                    if self._output_stream is None:
                        self._output_stream = sd.OutputStream(
                            samplerate=self.SAMPLE_RATE, channels=self.INPUT_CHANNELS,
                            dtype="float32", device=self._monitor_device,
                            callback=self._monitor_callback,
                        )
                    self._output_stream.start()
                    if not self._input_stream.active:
                        self._input_stream.start()
                    self._monitor_input = True
                except BaseException:
                    output_stream, self._output_stream = self._output_stream, None
                    try:
                        if output_stream is not None:
                            output_stream.close()
                    finally:
                        if not self._recording and self._input_stream.active:
                            self._input_stream.stop()
                    raise

        logger.info("Microphone monitoring %s", "started" if self._monitor_input else "stopped")
        return self._monitor_input

    def _callback(self, indata, frames, time, status):
        if status:
            self.status = status
        if self._monitor_input:
            self._monitor_buffer.write(indata)
        if self._recording_lock.acquire(blocking=False):
            try:
                if self._recording:
                    self._audio_queue.put(indata.copy())
            finally:
                self._recording_lock.release()

    def _monitor_callback(self, outdata, frames, time, status):
        if status:
            self.status = status
        if self._monitor_input:
            self._monitor_buffer.read_into(outdata)
        else:
            outdata.fill(0)

    def start_recording(self):
        with self._lock:
            if not self.running:
                raise RuntimeError("Microphone source is closed")
            if self._recording:
                return
            self.last_recording = None
            self.last_recording_path = None
            with self._recording_lock:
                self._recording = True
            try:
                if not self._input_stream.active:
                    self._input_stream.start()
            except BaseException:
                with self._recording_lock:
                    self._recording = False
                self._drain_audio_queue()
                self._audio_data.clear()
                raise
        logger.info("Microphone recording started")

    def toggle_recording(self):
        if self._recording:
            return self.stop_recording()
        return self.start_recording()

    def stop_recording(self):
        error = None
        with self._lock:
            with self._recording_lock:
                self._recording = False
            if self.running and not self._monitor_input:
                try:
                    self._input_stream.stop()
                except Exception as exc:
                    error = exc

            self._drain_audio_queue()
            if self._audio_data:
                self.last_recording = np.concatenate(self._audio_data, axis=0)
                self._audio_data = []
                if self._save:
                    self.last_recording_path = self._save_recording(self.last_recording)
                logger.info(
                    "Microphone recording stopped with %s samples",
                    len(self.last_recording),
                )
            else:
                logger.info("Microphone recording stopped with no audio data")

        if error is not None:
            raise error
        if self._save:
            return self.last_recording_path
        return self.last_recording

    @property
    def is_recording(self) -> bool:
        """
        Gets whether the microphone is currently recording.

        Returns:
            bool: Whether the microphone is recording.
        """

        return self._recording

    def _save_recording(self, audio_array):
        self._save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        file_path = self._save_dir / f"microphone_{timestamp}.wav"
        sf.write(file_path, audio_array, self.SAMPLE_RATE)
        logger.info("Microphone recording saved to %s", file_path)
        return str(file_path)

    def quit(self):
        logger.info("Stopping microphone source")
        error = None
        with self._lock:
            if not self.running:
                return
            self.running = False
            self._monitor_input = False
            with self._recording_lock:
                self._recording = False
            for stream in (self._output_stream, self._input_stream):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception as exc:
                        error = error or exc
        if self._microphone_thread.is_alive():
            self._microphone_thread.join()
        if error is not None:
            raise error
        logger.info("Microphone source stopped")
