import base64
import io
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from llm_toolkit.agents.agent import Agent
from llm_toolkit.agents.realtime import RealtimeSession
from llm_toolkit.media import file_source


class Capture:
    def __init__(self, count):
        self.count = count
        self.position = 0
        self.closed = False

    def isOpened(self):
        return True

    def get(self, prop):
        # Deliberately unknown container frame count.
        return 30 if prop == cv2.CAP_PROP_FPS else 0

    def grab(self):
        self.position += 1
        return self.position <= self.count

    def read(self):
        return self.grab(), self.position - 1

    def retrieve(self):
        return True, self.position - 1

    def release(self):
        self.closed = True


@pytest.mark.parametrize("count,limit,expected", [
    (0, 3, []), (1, 3, [0]), (5, 3, [0, 2, 4]),
    (5, 1, [0]), (2, 20, [0, 1]),
    (5000, 4, [0, 1666, 3333, 4999]),
])
def test_video_retains_and_encodes_only_selected_samples(count, limit, expected):
    captures = []

    def open_capture(_):
        cap = Capture(count)
        captures.append(cap)
        return cap

    with patch.object(cv2, "VideoCapture", side_effect=open_capture), patch.object(
        cv2, "imencode", side_effect=lambda _, frame: (True, frame)
    ) as encode:
        assert file_source._sample_frames("clip", 30, max_frames=limit) == expected
    assert encode.call_count == len(expected)
    assert all(cap.closed for cap in captures)


def test_bounded_sampling_keeps_last_sample_at_requested_fps():
    with patch.object(cv2, "VideoCapture", side_effect=lambda _: Capture(100)), patch.object(
        cv2, "imencode", side_effect=lambda _, frame: (True, frame)
    ):
        assert file_source._sample_frames("clip", 2, max_frames=3) == [0, 45, 90]


@pytest.mark.parametrize("limit", [0, -1, 1.5, True])
def test_invalid_frame_limit_never_opens_capture(limit):
    with patch.object(cv2, "VideoCapture") as capture, pytest.raises(ValueError):
        file_source.load_video("clip", max_frames=limit)
    capture.assert_not_called()


def test_second_pass_encoding_failure_releases_both_captures():
    captures = [Capture(5), Capture(5)]
    with patch.object(cv2, "VideoCapture", side_effect=captures), patch.object(
        cv2, "imencode", return_value=(False, None)
    ), pytest.raises(ValueError, match="encode"):
        file_source._sample_frames("clip", 30, max_frames=3)
    assert all(cap.closed for cap in captures)


def test_failed_retrieval_does_not_silently_return_partial_video():
    captures = [Capture(5), Capture(5)]
    captures[1].retrieve = Mock(return_value=(False, None))
    with patch.object(cv2, "VideoCapture", side_effect=captures), pytest.raises(ValueError, match="retrieve"):
        file_source._sample_frames("clip", 30, max_frames=3)
    assert all(cap.closed for cap in captures)


def test_real_video_loader_returns_decodable_first_and_last_frames(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    try:
        for value in (0, 60, 120, 240):
            writer.write(np.full((32, 32, 3), value, dtype=np.uint8))
    finally:
        writer.release()
    urls = file_source.load_video_urls(str(path), sample_fps=10, max_frames=2)
    assert len(urls) == 2
    pixels = [cv2.imdecode(np.frombuffer(base64.b64decode(url.split(",")[1]),
                                        dtype=np.uint8), cv2.IMREAD_COLOR) for url in urls]
    assert pixels[0].mean() < 5
    assert pixels[1].mean() > 235


def test_realtime_passes_frame_limit_to_loader():
    session = RealtimeSession("ws://localhost", "test")
    with patch.object(file_source, "load_video_urls", return_value=["one"]) as loader:
        assert session._prepare_video("clip", 2, 1) == ["one"]
    loader.assert_called_once_with("clip", sample_fps=2, max_frames=1)


class Response(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.read_sizes = []

    def read(self, size=-1):
        assert 0 < size <= 65536
        self.read_sizes.append(size)
        return super().read(size)


def test_link_read_is_bounded_and_passes_timeout():
    data = b"abc" * 30000
    response = Response(data)
    with patch.object(file_source, "urlopen", return_value=response) as open_url:
        result = file_source.load_link("https://example.test/image", max_bytes=len(data), timeout=2)
    assert base64.b64decode(result) == data
    open_url.assert_called_once_with("https://example.test/image", timeout=2)
    assert response.closed
    assert len(response.read_sizes) >= 3


def test_link_rejects_oversized_response_even_without_content_length():
    response = Response(b"a" * 101)
    with patch.object(file_source, "urlopen", return_value=response), pytest.raises(ValueError, match="max_bytes"):
        file_source.load_link("https://example.test/image", max_bytes=100)
    assert response.closed


@pytest.mark.parametrize("options", [{"max_bytes": 0}, {"max_bytes": 1.5},
                                     {"timeout": 0}, {"timeout": float("inf")}])
def test_invalid_download_limits_do_not_open_connection(options):
    with patch.object(file_source, "urlopen") as open_url, pytest.raises(ValueError):
        file_source.load_link("https://example.test/image", **options)
    open_url.assert_not_called()


@pytest.mark.parametrize("failure", ["size", "timeout"])
def test_failed_download_preserves_destination_and_removes_partial_file(tmp_path, failure):
    destination = tmp_path / "image.png"
    destination.write_bytes(b"original")
    response = Response(b"too big")
    if failure == "timeout":
        response.read = Mock(side_effect=[b"part", TimeoutError("timed out")])
    with patch.object(file_source, "urlopen", return_value=response), pytest.raises((ValueError, TimeoutError)):
        file_source.save_link("https://example.test/image", str(destination), max_bytes=4)
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]
    assert response.closed


def test_generated_image_url_streams_to_file_with_download_options(tmp_path):
    response = {"data": [{"url": "https://example.test/image"}]}
    destination = tmp_path / "image.png"
    with patch.object(Agent, "_update_model_support"), patch(
        "llm_toolkit.agents.agent.litellm.image_generation", return_value=response
    ) as generate, patch.object(file_source, "urlopen", return_value=Response(b"image")) as open_url:
        agent = Agent()
        agent.generate_image("draw", file_path=str(destination), download_max_bytes=10, download_timeout=3)
    assert destination.read_bytes() == b"image"
    assert agent.last_image_path == str(destination)
    open_url.assert_called_once_with("https://example.test/image", timeout=3)
    assert "download_timeout" not in generate.call_args.kwargs
