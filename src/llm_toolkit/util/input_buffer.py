import queue
import threading
from collections.abc import Callable, Iterable
from typing import Any, final

from llm_toolkit.logging import get_logger
from llm_toolkit.types import ActionEvent, InputEvent, InputType

logger = get_logger(__name__)


@final
class InputBuffer:
    def __init__(
        self,
        auto_send: bool = False,
        send_handler: Callable[[list[InputEvent]], Any] = None,
    ):
        self.auto_send = auto_send
        self.send_handler = send_handler
        self.events: queue.Queue[ActionEvent] = queue.Queue()
        self._pending: dict[InputType, InputEvent] = {}
        self._lock = threading.Lock()

    def stage(self, input_type: InputType, value: Any) -> list[InputEvent]:
        """
        Stages an input, replacing an existing input of the same type.

        Args:
            `input_type` (InputType): Type of input being staged.
            `value` (Any): Captured input value.

        Returns:
            list[InputEvent]: The currently staged inputs.
        """

        return self.stage_many([{"type": input_type, "value": value}])

    def stage_many(self, inputs: Iterable[InputEvent]) -> list[InputEvent]:
        """
        Stages multiple inputs as one capture.

        Args:
            `inputs` (Iterable[InputEvent]): Inputs to stage.

        Returns:
            list[InputEvent]: The currently staged inputs.
        """

        with self._lock:
            for input_event in inputs:
                self._pending[input_event["type"]] = input_event

            pending = list(self._pending.values())
            auto_send = self.auto_send
            if auto_send:
                self._pending = {}

        if not pending:
            return pending

        if auto_send:
            self._send(pending)
            return pending

        self.events.put({"type": "pending", "value": pending})
        logger.debug("Staged input types: %s", [event["type"] for event in pending])
        return pending

    def confirm(self) -> list[InputEvent]:
        """
        Confirms and sends all staged inputs.

        Returns:
            list[InputEvent]: The confirmed inputs.
        """

        with self._lock:
            confirmed = list(self._pending.values())
            self._pending = {}

        if not confirmed:
            return confirmed

        self._send(confirmed)
        return confirmed

    def _send(self, inputs: list[InputEvent]):
        self.events.put({"type": "send", "value": inputs})
        if self.send_handler is not None:
            self.send_handler(inputs)

        logger.info("Confirmed %s staged inputs", len(inputs))

    def discard(self) -> list[InputEvent]:
        """
        Discards all staged inputs.

        Returns:
            list[InputEvent]: The discarded inputs.
        """

        with self._lock:
            discarded = list(self._pending.values())
            self._pending = {}

        if not discarded:
            return discarded

        self.events.put({"type": "discard", "value": discarded})
        logger.info("Discarded %s staged inputs", len(discarded))
        return discarded

    def get_pending(self) -> list[InputEvent]:
        """
        Gets the currently staged inputs.

        Returns:
            list[InputEvent]: The currently staged inputs.
        """

        with self._lock:
            return list(self._pending.values())

    def set_auto_send(self, auto_send: bool):
        """
        Updates whether newly staged inputs are sent immediately.

        Args:
            `auto_send` (bool): Whether to automatically send staged inputs.
        """

        with self._lock:
            self.auto_send = auto_send
        logger.info("Input auto-send %s", "enabled" if auto_send else "disabled")

        if auto_send:
            self.confirm()

    def get_event(self, block: bool = True, timeout: float = None) -> ActionEvent:
        """
        Gets the next queued input event.

        Args:
            `block` (bool): Whether to wait for an event. Defaults to True.
            `timeout` (float): How long to wait for an event. Defaults to None.

        Returns:
            ActionEvent: The next queued event.
        """

        return self.events.get(block=block, timeout=timeout)
