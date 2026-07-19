import base64
import mimetypes
import os.path
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import cv2

from llm_toolkit.logging import get_logger

logger = get_logger(__name__)


def _get_absolute_path(file_path: str) -> str:
    """
    Gets the absolute path of a file.

    Args:
        `file_path` (str): Path to the file (relative or absolute).

    Returns:
        str: The absolute path of the original file.
    """

    return os.path.abspath(file_path)


def _encode_file(file_path: str, format: str = "utf-8") -> str:
    """
    Encodes a file as a base64 string.

    Args:
        `file_path` (str): Path to the file (absolute).
        `format` (str): Character encoding format. Defaults to UTF-8.

    Returns:
        str: A base64 encoded string of the file.
    """

    with open(file_path, "rb") as file:
        return base64.b64encode(file.read()).decode(format)


def _encode_bytes(data: bytes, format: str = "utf-8") -> str:
    return base64.b64encode(data).decode(format)


def _guess_mime_type(file_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or "application/octet-stream"


def _format_data_url(encoded_data: str, mime_type: str) -> str:
    return f"data:{mime_type};base64,{encoded_data}"


def _sample_frames(file_path: str, sample_fps: int) -> list[Any]:
    """
    Samples frames from a video at a specified fps as a list of JPEG images.

    Args:
        `file_path` (str): Path to the file (absolute).
        `sample_fps` (int): How many frames to sample from the video every second.

    Returns:
        `list[np.ndarray[int]]`: A list of sampled frames from `video_path` as JPEG images.
    """

    if sample_fps <= 0:
        raise ValueError("sample_fps must be greater than 0")

    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {file_path}")

    frames: list[Any] = []

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or sample_fps
    sample_interval = max(1, round(video_fps / sample_fps))

    success, frame = cap.read()
    current_frame = 1

    while current_frame <= total_video_frames and success:
        if current_frame % sample_interval == 0:
            success_encode, frame_jpg = cv2.imencode(".jpg", frame)
            if not success_encode:
                raise ValueError(f"Unable to encode frame {current_frame} from {file_path}")
            frames.append(frame_jpg)

        success, frame = cap.read()
        current_frame += 1

    cap.release()

    logger.info("Sampled %s frames from video %s", len(frames), file_path)
    return frames


def load_video(file_path: str, sample_fps: int = 2) -> list[str]:
    """
    Loads a video file as a list of base64 encoded strings.

    Args:
        `file_path` (str): Path to the file (relative or absolute).
        `sample_fps` (int): The number of frames to sample from the video per second.

    Returns:
        list[str]: A list of base64 encoded strings sampled from `video_path`.
    """

    abs_path = _get_absolute_path(file_path)
    sampled_frames = _sample_frames(file_path=abs_path, sample_fps=sample_fps)
    encoded_frames: list[str] = []

    for frame in sampled_frames:
        encoded_frame = _encode_bytes(frame.tobytes())
        encoded_frames.append(encoded_frame)

    logger.info("Loaded video %s as %s encoded frames", abs_path, len(encoded_frames))
    return encoded_frames


def load_video_urls(file_path: str, sample_fps: int = 2) -> list[str]:
    """
    Loads a video file as a list of base64 JPEG data URLs.

    Args:
        `file_path` (str): Path to the file (relative or absolute).
        `sample_fps` (int): The number of frames to sample from the video per second.

    Returns:
        list[str]: A list of base64 JPEG data URLs sampled from `file_path`.
    """

    return [
        _format_data_url(encoded_frame, "image/jpeg")
        for encoded_frame in load_video(file_path, sample_fps=sample_fps)
    ]


def load_file(file_path: str) -> str:
    """
    Loads a file as a base64 encoded string.

    Args:
        `file_path` (str): Path to the file (relative or absolute).

    Returns:
        str: A base64 encoded string of the image file.
    """
    abs_path = _get_absolute_path(file_path)
    encoded_file = _encode_file(abs_path)

    logger.debug("Loaded file %s as base64", abs_path)
    return encoded_file


def load_image(file_path: str) -> str:
    """
    Loads an image file as a base64 encoded string. Currently uses the same logic as `load_file` with a use-case-friendly name.

    Args:
        `file_path` (str): Path to the file (relative or absolute).

    Returns:
        str: A base64 encoded string of the image file.
    """

    encoded_image = load_file(file_path)

    return encoded_image


def load_audio(file_path: str) -> str:
    """
    Loads an audio file as a base64 encoded string. Currently uses the same logic as `load_file` with a use-case-friendly name.

    Args:
        `file_path` (str): Path to the file (relative or absolute).

    Returns:
        str: A base64 encoded string of the audio file.
    """

    encoded_audio = load_file(file_path)

    return encoded_audio


def load_link(url: str) -> str:
    """
    Loads a URL as a base64 encoded string.

    Args:
        `url` (str): URL to load.

    Returns:
        str: A base64 encoded string of the response body.
    """

    with urlopen(url) as response:
        encoded_data = _encode_bytes(response.read())

    logger.info("Loaded link %s as base64", url)
    return encoded_data


def load_image_url(file_path: str) -> str:
    """
    Loads an image file as a base64 data URL.

    Args:
        `file_path` (str): Path to the file (relative or absolute).

    Returns:
        str: A base64 data URL of the image file.
    """

    abs_path = _get_absolute_path(file_path)
    encoded_image = load_image(abs_path)

    return _format_data_url(encoded_image, _guess_mime_type(abs_path))


def load_audio_url(file_path: str) -> str:
    """
    Loads an audio file as a base64 data URL.

    Args:
        `file_path` (str): Path to the file (relative or absolute).

    Returns:
        str: A base64 data URL of the audio file.
    """

    abs_path = _get_absolute_path(file_path)
    encoded_audio = load_audio(abs_path)

    return _format_data_url(encoded_audio, _guess_mime_type(abs_path))


def save_file(data: bytes | str, file_path: str, encoding: str = "utf-8") -> str:
    """
    Saves bytes or text to a file and returns the absolute path.
    """

    output_path = Path(file_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, bytes):
        with open(output_path, "wb") as file:
            file.write(data)
    else:
        with open(output_path, "w", encoding=encoding) as file:
            file.write(data)

    logger.info("Saved file to %s", output_path)
    return str(output_path)


def save_base64_file(encoded_data: str, file_path: str) -> str:
    """
    Decodes a base64 string into a file and returns the absolute path.
    """

    saved_path = save_file(base64.b64decode(encoded_data), file_path)
    logger.info("Saved base64 file to %s", saved_path)
    return saved_path
