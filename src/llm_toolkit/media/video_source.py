import threading
from datetime import datetime
from typing import final

import cv2


@final
class Camera:
    def __init__(self, source: int = 0):
        self._show_capture = False
        self._capture = cv2.VideoCapture(source)
        self._lock = threading.Lock()

        self.running = True
        self.success, self._frame = self._capture.read()

        self._capture_thread = threading.Thread(target=self._read_capture)
        self._capture_thread.daemon = True
        self._capture_thread.start()

    def _read_capture(self):
        while self.running:
            with self._lock:
                self.success, self._frame = self._capture.read()

                if self._show_capture:
                    cv2.imshow("Capture", self._frame.copy())

    def show_capture(self):
        with self._lock:
            self._show_capture = True

    def hide_capture(self):
        with self._lock:
            self._show_capture = False
            cv2.destroyWindow("Capture")

    def snapshot(self) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_name = f"{timestamp}.jpg"

        with self._lock:
            image = self._frame.copy()
            _ = cv2.imwrite(file_name, image)

        return file_name

    def start_recording(self):
        return

    def stop_recording(self):
        return

    def quit(self):
        self.running = False

        if self._capture_thread.is_alive():
            self._capture_thread.join()

        cv2.destroyWindow("Capture")

        self._capture.release()
