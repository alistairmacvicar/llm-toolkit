import queue
import threading
from collections.abc import Callable
from typing import Any, final

from llm_toolkit.logging import get_logger
from llm_toolkit.media.camera_source import CameraSource
from llm_toolkit.media.microphone_source import MicrophoneSource
from llm_toolkit.types import ActionEvent, InputEvent
from llm_toolkit.util.input_buffer import InputBuffer

logger = get_logger(__name__)


@final
class ActionHandler:
    def __init__(
        self,
        camera_source: CameraSource = None,
        microphone_source: MicrophoneSource = None,
        auto_send: bool = False,
        send_handler: Callable[[list[InputEvent]], Any] = None,
    ):
        from pynput import keyboard

        self.camera_source = camera_source
        self.microphone_source = microphone_source
        self.messages = queue.Queue()
        self.input_buffer = InputBuffer(
            auto_send=auto_send,
            send_handler=send_handler,
        )
        self.events = self.input_buffer.events
        self.running = True
        self._pressed_keys = set()
        self._reading_terminal = False

        self.push_to_talk_key = keyboard.KeyCode.from_char("a")  # (a)udio
        self.type_message_key = keyboard.KeyCode.from_char("t")  # (t)ext
        self.monitor_audio_key = keyboard.KeyCode.from_char("m")  # (m)onitor
        self.preview_camera_key = keyboard.KeyCode.from_char("p")  # (c)amera
        self.snapshot_key = keyboard.KeyCode.from_char("s")  # (s)napshot
        self.record_key = keyboard.KeyCode.from_char("r")  # (r)ecord
        self.quit_key = keyboard.KeyCode.from_char("q")  # (q)uit
        self.confirm_key = keyboard.Key.enter
        self.discard_key = keyboard.Key.esc

        self.action_handler = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )

        self.action_handler.start()
        logger.info("Action handler started")

    def _on_press(self, key):
        if not self.running or self._reading_terminal or key in self._pressed_keys:
            return
        self._pressed_keys.add(key)
        match key:
            case self.push_to_talk_key:
                if self.microphone_source is not None:
                    logger.debug("Push-to-talk pressed")
                    self.microphone_source.start_recording()
            case self.type_message_key:
                logger.debug("Terminal message input requested")
                self._reading_terminal = True
                threading.Thread(target=self._read_terminal_message, daemon=True).start()
            case self.monitor_audio_key:
                if self.microphone_source is not None:
                    logger.debug("Microphone monitoring toggled")
                    self.microphone_source.toggle_monitor()
            case self.preview_camera_key:
                if self.camera_source is not None:
                    logger.debug("Camera preview toggled")
                    self.camera_source.toggle_capture()
            case self.snapshot_key:
                if self.camera_source is not None:
                    snapshot_path = self.camera_source.snapshot()
                    self.input_buffer.stage("image", snapshot_path)
                    logger.debug("Camera snapshot staged")
            case self.record_key:
                logger.debug("Recording toggled for active sources")
                completed_inputs = []

                if self.microphone_source is not None:
                    was_recording = self.microphone_source.is_recording
                    audio = self.microphone_source.toggle_recording()
                    if was_recording and audio is not None:
                        completed_inputs.append({"type": "audio", "value": audio})

                if self.camera_source is not None:
                    was_recording = self.camera_source.is_recording
                    video_path = self.camera_source.toggle_recording()
                    if was_recording and video_path is not None:
                        completed_inputs.append({"type": "video", "value": video_path})

                self.input_buffer.stage_many(completed_inputs)
            case self.quit_key:
                logger.info("Quit key pressed")
                self.events.put({"type": "quit", "value": None})
                self.stop()
            case self.confirm_key:
                self.input_buffer.confirm()
            case self.discard_key:
                self.input_buffer.discard()

    def _on_release(self, key):
        was_pressed = key in self._pressed_keys
        self._pressed_keys.discard(key)
        if not self.running or not was_pressed:
            return
        match key:
            case self.push_to_talk_key:
                if self.microphone_source is not None:
                    logger.debug("Push-to-talk released")
                    audio = self.microphone_source.stop_recording()
                    if audio is not None:
                        self.input_buffer.stage("audio", audio)

    def quit(self):
        self.stop()
        self.action_handler.join()

    def stop(self):
        if not self.running:
            return

        logger.info("Stopping action handler")
        self.running = False
        self.action_handler.stop()
        if self.microphone_source is not None:
            self.microphone_source.quit()
        if self.camera_source is not None:
            self.camera_source.quit()
        logger.info("Action handler stopped")

    def _read_terminal_message(self):
        try:
            message = input("Message: ")
            if message:
                self.messages.put(message)
                self.input_buffer.stage("text", message)
                logger.info("Terminal message queued")
        except EOFError:
            self.events.put({"type": "quit", "value": None})
            self.stop()
        finally:
            self._reading_terminal = False

    def get_pending_inputs(self) -> list[InputEvent]:
        """
        Gets the inputs waiting for confirmation.

        Returns:
            list[InputEvent]: The currently staged inputs.
        """

        return self.input_buffer.get_pending()

    def set_auto_send(self, auto_send: bool):
        """
        Updates whether captured inputs are sent immediately.

        Args:
            auto_send (bool): Whether to automatically send captured inputs.
        """

        self.input_buffer.set_auto_send(auto_send)

    def get_event(self, block: bool = True, timeout: float = None) -> ActionEvent:
        """
        Gets the next queued action event.

        Args:
            block (bool): Whether to wait for an event. Defaults to True.
            timeout (float): How long to wait for an event. Defaults to None.

        Returns:
            ActionEvent: The next queued event.
        """

        return self.events.get(block=block, timeout=timeout)
