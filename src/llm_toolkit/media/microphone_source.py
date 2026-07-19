import threading
import time
from datetime import datetime
from pathlib import Path
from typing import final
import queue

import sounddevice as sd
import numpy as np
import soundfile as sf

from llm_toolkit.logging import get_logger

logger = get_logger(__name__)


@final
class MicrophoneSource:
    def __init__(self, source: int = -1, save: bool = False, save_dir: str = "recordings"):
        self.SAMPLE_RATE = 24000  # https://developers.openai.com/api/docs/guides/realtime-transcription
        self.INPUT_CHANNELS = 1  # Mono for transcriptions should work on all devices
        self._audio_queue = queue.Queue()  # buffered audio from the sd InputStream
        self._audio_data = []
        self._save = save
        self._save_dir = Path(save_dir)
        self.last_recording = None
        self.last_recording_path = None

        self._monitor_input = False
        self._input_device = self._get_input_source(source=source)
        self._input_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            device=self._input_device["index"],
            channels=self.INPUT_CHANNELS,
            callback=self._callback,
        )
        self._lock = threading.Lock()

        self.running = True
        self.status = None
        self._recording = False

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
        while not self._audio_queue.empty():
            self._audio_data.append(self._audio_queue.get())

    def _get_input_source(self, source: int = -1):
        # Return the default input device set in the OS if the user hasn't chosen a specific one
        if source == -1:
            return sd.get_devices(kind="input")
        else:
            return sd.get_devices(device=source)

    def toggle_monitor(self):
        with self._lock:
            self._monitor_input = not self._monitor_input

            if self._monitor_input and not self._input_stream.active:
                self._input_stream.start()
            elif not self._monitor_input and not self._recording:
                self._input_stream.stop()

        logger.info("Microphone monitoring %s", "started" if self._monitor_input else "stopped")
        return self._monitor_input

    def _callback(self, indata, frames, time, status):
        if status:
            self.status = status
            logger.warning("Microphone input status: %s", status)
        if self._monitor_input:
            sd.play(indata.copy(), samplerate=self.SAMPLE_RATE, blocking=False)
        if self._recording:
            self._audio_queue.put(indata.copy())

    def start_recording(self):
        with self._lock:
            self._recording = True
            if not self._input_stream.active:
                self._input_stream.start()
        logger.info("Microphone recording started")

    def toggle_recording(self):
        if self._recording:
            return self.stop_recording()
        return self.start_recording()

    def stop_recording(self):
        with self._lock:
            self._recording = False
            if not self._monitor_input:
                self._input_stream.stop()

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
        self.running = False

        if self._input_stream.active:
            self._input_stream.stop()
        self._input_stream.close()

        if self._microphone_thread.is_alive():
            self._microphone_thread.join()
        logger.info("Microphone source stopped")
