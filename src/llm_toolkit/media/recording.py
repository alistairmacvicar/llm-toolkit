"""Durable input-audio archives and audio/video recording assembly."""

import math
import os
import re
import subprocess
import wave
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from llm_toolkit.logging import get_logger

logger = get_logger(__name__)


def recording_directory(directory=None) -> Path:
    return Path(directory or os.getenv("LLM_TOOLKIT_RECORDING_DIR", "recordings")).expanduser()


def new_recording_path(directory, prefix, extension) -> Path:
    directory = recording_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return directory / f"{prefix}_{stamp}_{uuid4().hex[:8]}.{extension}"


def save_input_audio(data: bytes, format: str, directory=None) -> str:
    """Archive the encoded audio before it can be sent to a provider."""
    if not re.fullmatch(r"[a-zA-Z0-9]{1,16}", format):
        raise ValueError("Audio format must be a filename extension, such as wav or mp3")
    path = new_recording_path(directory, "input-audio", format.lower())
    path.write_bytes(data)
    logger.info("Input audio saved: %s", path)
    return str(path.resolve())


class PCMArchive:
    """Append streamed mono PCM16 to one WAV without retaining it in memory."""

    def __init__(self, directory=None, sample_rate=24000, *, prefix="input-audio"):
        self.path = new_recording_path(directory, prefix, "wav")
        self._output = self.path.open("wb")
        self.file = wave.open(self._output, "wb")
        try:
            self.file.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
        except BaseException:
            self._output.close()
            raise
        logger.info("Audio recording: %s", self.path)

    def write(self, data):
        # writeframes also updates the header, so an interrupted process leaves
        # a readable recording of the chunks already written.
        self.file.writeframes(data)
        self._output.flush()

    def close(self):
        try:
            self.file.close()
        finally:
            self._output.close()


def mux_recordings(video_path, audio_path, *, audio_offset=0.0) -> str:
    """Create an MP4 with the separate microphone track; preserve both originals."""
    import imageio_ffmpeg

    if not math.isfinite(audio_offset):
        raise ValueError("Audio offset must be finite")
    video = Path(video_path).expanduser().resolve()
    audio = Path(audio_path).expanduser().resolve()
    if not video.is_file() or not audio.is_file():
        raise FileNotFoundError("Both camera video and microphone audio must exist before combining them")
    output = video.with_name(f"{video.stem}-with-audio.mp4")
    temporary = output.with_name(f".{output.stem}-{uuid4().hex}.mp4")
    try:
        result = subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(video), "-itsoffset", str(audio_offset), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
            "-af", "apad", "-shortest", "-movflags", "+faststart", str(temporary),
        ], capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(f"Could not combine camera video and audio: {result.stderr.strip()}")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    logger.info("Video with microphone audio saved: %s", output)
    return str(output)


def toggle_recording_sources(sources):
    """Start idle sources together, or stop active sources and assemble their capture."""
    stopping = any(source.is_recording for _, source in sources)
    completed = []
    stopped = {}
    for kind, source in sources:
        if stopping and not source.is_recording:
            continue
        value = source.toggle_recording()
        if stopping and value is not None:
            completed.append({"type": kind, "value": value})
            stopped[kind] = source
    by_type = {item["type"]: item for item in completed}
    if "audio" in by_type and "video" in by_type:
        microphone, camera = stopped["audio"], stopped["video"]
        audio = by_type["audio"]["value"]
        if not isinstance(audio, (str, Path)):
            import soundfile as sf

            path = new_recording_path(Path(by_type["video"]["value"]).parent, "microphone", "wav")
            sf.write(path, audio, microphone.SAMPLE_RATE)
            audio = by_type["audio"]["value"] = str(path)
        offset = microphone.recording_started_at - camera.recording_started_at
        by_type["video"]["value"] = mux_recordings(
            by_type["video"]["value"], audio, audio_offset=offset,
        )
    return completed
