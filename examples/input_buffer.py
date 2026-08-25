from queue import Empty

from llm_toolkit import InputBuffer, configure_logging


def show_sent(inputs):
    """
    Displays inputs dispatched by the buffer.

    Args:
        `inputs` (list[InputEvent]): Confirmed input values.
    """

    print(f"Send handler received: {inputs}")


def main():
    buffer = InputBuffer(send_handler=show_sent)
    print("Commands: stage TYPE VALUE, pending, confirm, discard, auto, event, quit")

    while True:
        command = input("Buffer: ").strip()
        if command == "quit":
            break
        if command == "pending":
            print(buffer.get_pending())
        elif command == "confirm":
            print(f"Confirmed: {buffer.confirm()}")
        elif command == "discard":
            print(f"Discarded: {buffer.discard()}")
        elif command == "auto":
            buffer.set_auto_send(not buffer.auto_send)
            print(f"Auto-send: {buffer.auto_send}")
        elif command == "event":
            try:
                print(buffer.get_event(block=False))
            except Empty:
                print("No queued events.")
        elif command.startswith("stage "):
            _, input_type, value = command.split(maxsplit=2)
            if input_type not in {"text", "audio", "image", "video"}:
                print("Type must be text, audio, image, or video.")
                continue
            print(f"Pending: {buffer.stage(input_type, value)}")
        elif command:
            print("Unknown command.")


if __name__ == "__main__":
    configure_logging()
    main()
