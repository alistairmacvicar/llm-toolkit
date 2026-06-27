import base64
import os.path

import cv2
import numpy as np

# TODO: Add functionality to save the files.


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


def _sample_frames(file_path: str, sample_fps: int) -> list[np.ndarray[int]]:
    """
    Samples frames from a video at a specified fps as a list of JPEG images.

    Args:
        `file_path` (str): Path to the file (absolute).
        `sample_fps` (int): How many frames to sample from the video every second.

    Returns:
        `list[np.ndarray[int]]`: A list of sampled frames from `video_path` as JPEG images.
    """

    cap = cv2.VideoCapture(file_path)
    frames: list[str] = []

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    sample_interval = video_fps // sample_fps

    success, frame = cap.read()
    current_frame = 1

    while current_frame <= total_video_frames and success:
        if current_frame % sample_interval == 0:
            frame_jpg = cv2.imencode(".jpg", frame)[1]
            frames.append(frame_jpg)

        success, frame = cap.read()

    cap.release()

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
    sampled_frames = _sample_frames(file_path=abs_path, sampe_fps=sample_fps)
    encoded_frames: list[str] = []

    for frame in sampled_frames:
        encoded_frame = _encode_file(frame)
        encoded_frames.append(encoded_frame)

    return encoded_frames


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
