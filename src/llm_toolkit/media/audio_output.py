"""Explicit device selection and channel routing for local audio installations."""

import io
from threading import Lock
from typing import final

import numpy as np


@final
class AudioOutput:
    """Play PCM or WAV sequentially on a selected physical or virtual device.

    Hardware is opened only during playback. Channel numbers are zero based;
    ``None`` duplicates mono across all output channels.
    """

    def __init__(self, device=None, channels=2, sample_rate=24000, stream_factory=None):
        if channels < 1 or sample_rate < 1:
            raise ValueError("Channels and sample rate must be positive")
        self.device = device
        self.channels = channels
        self.sample_rate = sample_rate
        self._stream_factory = stream_factory
        self._lock = Lock()

    @staticmethod
    def devices():
        """Return PortAudio devices without opening a recording or playback stream."""
        import sounddevice as sd
        return [dict(device, index=index) for index, device in enumerate(sd.query_devices())]

    def play_pcm(self, chunks, *, channel=None, on_state=None):
        """Block until all PCM16 chunks drain, with accurate start/end callbacks."""
        if channel is not None and (type(channel) is not int or not 0 <= channel < self.channels):
            raise ValueError("Output channel is outside the device channel range")
        factory = self._stream_factory
        if factory is None:
            import sounddevice as sd
            factory = sd.OutputStream
        with self._lock:
            try:
                with factory(device=self.device, samplerate=self.sample_rate, channels=self.channels, dtype="float32") as stream:
                    if on_state:
                        on_state(True)
                    for chunk in chunks:
                        mono = np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768
                        output = np.zeros((len(mono), self.channels), dtype=np.float32)
                        if channel is None:
                            output[:] = mono[:, None]
                        else:
                            output[:, channel] = mono
                        stream.write(output)
                    stream.stop()  # PortAudio drains queued audio before returning.
            finally:
                if on_state:
                    on_state(False)

    def play_wav(self, audio, *, channel=None, on_state=None):
        """Decode WAV and resample to this device's configured rate."""
        import soundfile as sf
        values, rate = sf.read(io.BytesIO(audio), dtype="float32", always_2d=True)
        mono = values.mean(axis=1)
        if rate != self.sample_rate and len(mono):
            mono = np.interp(np.arange(round(len(mono) * self.sample_rate / rate)) * rate / self.sample_rate,
                             np.arange(len(mono)), mono)
        pcm = (np.clip(mono, -1, 1) * 32767).astype("<i2").tobytes()
        self.play_pcm([pcm], channel=channel, on_state=on_state)
