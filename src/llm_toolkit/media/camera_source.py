import threading
import time
import math
from datetime import datetime
from pathlib import Path
from typing import final

import cv2

from llm_toolkit.logging import get_logger

logger = get_logger(__name__)


@final
class CameraSource:
    def __init__(self, source: int = 0, save_dir: str = "recordings"):
        self._show_preview = False
        self._recording = False
        self._capture = cv2.VideoCapture(source)
        self._lock = threading.Lock()
        self._save_dir = Path(save_dir)
        self._video_writer = None
        self.last_snapshot_path = None
        self.last_recording_path = None
        self._closed = False

        self.running = True
        try:
            if not self._capture.isOpened():
                raise RuntimeError(f"Unable to open camera source: {source}")
            self.success, self._frame = self._capture.read()
            if self.success:
                logger.info("Camera source %s started", source)
            else:
                logger.warning("Camera source %s started without an initial frame", source)

            self._capture_thread = threading.Thread(target=self._read_capture)
            self._capture_thread.daemon = True
            self._capture_thread.start()
        except BaseException:
            self.running = False
            self._capture.release()
            raise

    def _read_capture(self):
        while self.running:
            with self._lock:
                self.success, self._frame = self._capture.read()
                frame = self._frame.copy() if self.success else None
                show_preview = self._show_preview

                if frame is not None and self._recording and self._video_writer is not None:
                    self._video_writer.write(frame)

            if frame is not None:
                if show_preview:
                    cv2.imshow("Capture", frame)
                    cv2.waitKey(1)
            else:
                time.sleep(0.01)

    def toggle_capture(self):
        with self._lock:
            self._ensure_open()
            if self._show_preview:
                self._hide_capture()
            else:
                self._show_capture()

    def _show_capture(self):
        cv2.namedWindow("Capture", cv2.WINDOW_NORMAL)
        self._show_preview = True
        logger.info("Camera preview started")

    def _hide_capture(self):
        self._show_preview = False
        try:
            cv2.destroyWindow("Capture")
        except cv2.error:
            logger.debug("Camera preview window was already closed")
        else:
            logger.info("Camera preview stopped")

    def snapshot(self) -> str:
        self._save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        file_path = self._save_dir / f"camera_{timestamp}.jpg"

        with self._lock:
            self._ensure_open()
            if not self.success or self._frame is None:
                raise RuntimeError("No camera frame is available; check the device and camera permission")
            image = self._frame.copy()
            saved = cv2.imwrite(str(file_path), image)

        if not saved:
            raise RuntimeError(f"Unable to save camera snapshot: {file_path}")

        self.last_snapshot_path = str(file_path)
        logger.info("Camera snapshot saved to %s", self.last_snapshot_path)
        return self.last_snapshot_path

    def toggle_recording(self):
        with self._lock:
            self._ensure_open()
            if not self._recording:
                return self._start_recording()
            return self._stop_recording()

    def _start_recording(self):
        if not self.success or self._frame is None:
            raise RuntimeError("No camera frame is available; check the device and camera permission")
        self._save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        file_path = self._save_dir / f"camera_{timestamp}.mp4"
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30
        height, width = self._frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(str(file_path), fourcc, fps, (width, height))
        try:
            if not writer.isOpened():
                raise RuntimeError(f"Unable to start camera recording: {file_path}")
        except BaseException:
            writer.release()
            raise
        self._video_writer = writer

        self.last_recording_path = str(file_path)
        self._recording = True
        logger.info("Camera recording started: %s", self.last_recording_path)
        return self.last_recording_path

    def _stop_recording(self):
        self._recording = False
        writer, self._video_writer = self._video_writer, None
        if writer is not None:
            writer.release()
        logger.info("Camera recording stopped: %s", self.last_recording_path)
        return self.last_recording_path

    @property
    def is_recording(self) -> bool:
        """
        Gets whether the camera is currently recording.

        Returns:
            bool: Whether the camera is recording.
        """

        return self._recording

    def quit(self):
        logger.info("Stopping camera source")
        if self._closed:
            return
        self._closed = True
        self.running = False

        if self._capture_thread.is_alive():
            self._capture_thread.join()

        error = None
        try:
            with self._lock:
                self._show_preview = False
                self._stop_recording()
        except Exception as exc:
            error = exc
        try:
            try:
                cv2.destroyWindow("Capture")
            except cv2.error:
                logger.debug("Camera preview window was already closed")
        finally:
            self._capture.release()
        if error is not None:
            raise error
        logger.info("Camera source stopped")

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("Camera source is closed")
