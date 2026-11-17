import threading
from threading import Thread
from unittest.mock import Mock, patch

import numpy as np
import pytest

from llm_toolkit.media import camera_source


@pytest.fixture
def camera(tmp_path):
    capture = Mock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((24, 32, 3), dtype=np.uint8))
    with patch.object(camera_source.cv2, "VideoCapture", return_value=capture), patch.object(
        camera_source.threading, "Thread"
    ), patch.object(camera_source.cv2, "destroyWindow"):
        source = camera_source.CameraSource(save_dir=tmp_path)
        source._capture_thread.is_alive.return_value = False
        try:
            yield source
        finally:
            source.quit()


def test_preview_requested_by_worker_uses_only_main_thread_window_calls(camera):
    threads = []
    def window_call(*args):
        threads.append(threading.current_thread())

    with patch.object(camera_source.cv2, "namedWindow", side_effect=window_call) as window, patch.object(
        camera_source.cv2, "imshow", side_effect=window_call
    ), patch.object(camera_source.cv2, "waitKey", return_value=-1) as pump, patch.object(
        camera_source.cv2, "getWindowProperty", return_value=1
    ), patch.object(camera_source.cv2, "destroyWindow", side_effect=window_call) as destroy:
        worker = Thread(target=camera.toggle_capture)
        worker.start()
        worker.join()
        window.assert_not_called()
        camera.poll_preview()
        camera.toggle_capture()
        camera.poll_preview()
        pump.assert_called_once()
        destroy.assert_called_once()
    assert threads and all(thread is threading.main_thread() for thread in threads)


def test_capture_worker_never_calls_window_functions(camera):
    def read():
        camera.running = False
        return True, np.zeros((24, 32, 3), dtype=np.uint8)

    camera._capture.read.side_effect = read
    camera._show_preview = True
    with patch.object(camera_source.cv2, "imshow") as show, patch.object(camera_source.cv2, "waitKey") as pump:
        camera._read_capture()
    show.assert_not_called()
    pump.assert_not_called()


def test_polling_preview_from_background_thread_fails_before_touching_gui(camera):
    errors = []
    def poll():
        try:
            camera.poll_preview()
        except RuntimeError as error:
            errors.append(str(error))
    worker = Thread(target=poll)
    worker.start()
    worker.join()
    assert errors == ["Camera preview must be updated on the main thread"]


def test_preview_still_pumps_window_events_when_capture_is_busy(camera):
    camera.toggle_capture()
    with camera._lock, patch.object(camera_source.cv2, "namedWindow"), patch.object(
        camera_source.cv2, "waitKey", return_value=-1
    ) as pump, patch.object(camera_source.cv2, "imshow") as show, patch.object(
        camera_source.cv2, "getWindowProperty", return_value=1
    ):
        camera.poll_preview()
    pump.assert_called_once()
    show.assert_called_once()


def test_camera_writer_is_released_when_final_frame_fails(camera):
    writer = Mock()
    camera._recording = True
    camera._video_writer = writer
    with patch.object(camera, "_write_recording_frame", side_effect=OSError("Disk full")):
        with pytest.raises(OSError, match="Disk full"):
            camera.toggle_recording()
    assert not camera.is_recording
    assert camera._video_writer is None
    writer.release.assert_called_once()


def test_recorded_frame_timestamps_follow_wall_clock_even_with_slow_camera(camera):
    writer = Mock()
    camera._video_writer = writer
    camera.recording_started_at = 10.0
    camera._recording_fps = 10
    with patch.object(camera_source.time, "monotonic", return_value=11.0):
        camera._write_recording_frame(camera._frame)
    assert writer.write.call_count == 10
