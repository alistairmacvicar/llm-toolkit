import threading
from datetime import datetime
from typing import final
import queue

import sounddevice as sd
import numpy as np
import soundfile as sf


@final
class MicrophoneSource:
    def __init__(self, source: int = -1, save: bool = False):
        self.SAMPLE_RATE = 24000  # https://developers.openai.com/api/docs/guides/realtime-transcription
        self.INPUT_CHANNELS = 1  # Mono for transcriptions should work on all devices
        self._audio_queue = queue.Queue()  # buffered audio from the sd InputStream
        self._audio_data = []
        self._save = save
        self.last_recording = None

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

    def _listen(self):
        while self.running:
            with self._lock:
                if self._recording:
                    while not self._audio_queue.empty():
                        self._audio_data.append(self._audio_queue.get())

    def _get_input_source(self, source: int = -1):
        # Return the default input device set in the OS if the user hasn't chosen a specific one
        if source == -1:
            return sd.get_devices(kind="input")
        else:
            return sd.get_devices(device=source)

    def _monitor(self):
        # TODO: allow for mic passthrough for debugging
        return

    def _callback(self, indata, frames, time, status):
        if status:
            self.status = status
        self._audio_queue.put(indata.copy())

    def start_recording(self):
        with self._lock:
            self._recording = True
            self._input_stream.start()

    def stop_recording(self):
        with self._lock:
            self._recording = False
            self._input_stream.stop()

            if self._audio_data:
                self.last_recording = np.concatenate(self._audio_data, axis=0)
                self._audio_data = []

    def _save_recording(self, audio_array):
        # TODO: Implement saving recording to a file (I thinkI want this to be handled by the app itself? or have some kind of experiment handler save everything in teh same directory when a new session starts?)
        return
