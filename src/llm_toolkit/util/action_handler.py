import threading
from typing import final
from media.camera_source import CameraSource
from media.microphone_source import MicrophoneSource

from pynput import keyboard


@final
class ActionHandler:
    def __init__(
        self,
        camera_source: CameraSource = None,
        microphone_source: MicrophoneSource = None,
    ):
        self.push_to_talk_key = keyboard.KeyCode.from_char("a")  # (a)udio
        self.type_message_key = keyboard.KeyCode.from_char("t")  # (t)ext
        self.monitor_audio_key = keyboard.KeyCode.from_char("m")  # (m)onitor
        self.preview_camera_key = keyboard.KeyCode.from_char("p")  # (c)amera
        self.record_key = keyboard.KeyCode.from_char("r")  # (r)ecord
        self.quit_key = keyboard.KeyCode.from_char("q")  # (q)uit
        self.send_key = keyboard.Key.enter

        self.action_handler = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

        self.action_handler.start()

    def _on_press(self, key):
        match key:
            case self.push_to_talk_key:
                if self.microphone_source is not None:
                    self.microphone_source.start_recording()
            case self.type_message_key:
                # TODO: allow input to the terminal to type a message
                return
            case self.monitor_audio_key:
                # TODO: playback microphone audio on toggle for debugging
                return
            case self.preview_camera_key:
                if self.camera_source is not None:
                    self.camera_source.toggle_capture()
            case self.record_key:
                # TODO: make this record for all active sources
                return
            case self.quit_key:
                # TODO: safely stop all processes and terminate the program
                return
            case self.send_key:
                return

    def _on_release(self, key):
        match key:
            case self.push_to_talk_key:
                if self.microphone_source is not None:
                    self.microphone_source.stop_recording()

    def quit(self):
        self.action_handler.join()
