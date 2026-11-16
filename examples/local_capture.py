import argparse
from contextlib import ExitStack
from queue import Empty

from llm_toolkit import ActionHandler, Agent, InputBuffer, configure_logging
from llm_toolkit.media.camera_source import CameraSource
from llm_toolkit.media.microphone_source import MicrophoneSource

from _common import CHAT_STYLE, get_response_text


TERMINAL_HELP = (
    "Type a message to stage it, then press Enter on an empty line or use /send to get a reply.\n"
    "Commands: /audio (start/stop microphone), /snapshot, /record (start/stop all sources),\n"
    "          /preview, /monitor, /pending, /discard, /help, /quit"
)


def show_pending(inputs):
    if inputs:
        print(f"Pending: {', '.join(item['type'] for item in inputs)}. Confirm to get a reply, or discard.")
    else:
        print("Nothing staged. Type a message or capture something first.")


def handle_event(agent, event):
    if event["type"] == "send":
        print("Getting a reply...")
        agent.send_inputs(event["value"])
        print(f"Agent: {get_response_text(agent.get_response())}")
    elif event["type"] == "pending":
        show_pending(event["value"])
    elif event["type"] == "discard":
        print("Pending inputs discarded.")


def toggle_recording(buffer, sources):
    completed = []
    for input_type, source in sources:
        was_recording = source.is_recording
        value = source.toggle_recording()
        print(f"{input_type.capitalize()} recording {'stopped' if was_recording else 'started'}.")
        if was_recording and value is not None:
            completed.append({"type": input_type, "value": value})
    buffer.stage_many(completed)


def run_terminal(agent, camera, microphone, *, auto_send):
    buffer = InputBuffer(auto_send=auto_send)
    print("Terminal controls are active; global keyboard monitoring is not used.")
    print(TERMINAL_HELP)
    if auto_send:
        print("Auto-send is on: each completed capture requests a reply immediately.")
    while True:
        try:
            command = input("Capture: ").strip()
        except EOFError:
            break
        if command == "/quit":
            break
        if command == "/help":
            print(TERMINAL_HELP)
        elif command in {"", "/send", "/confirm"}:
            if not buffer.confirm():
                show_pending([])
        elif command == "/discard":
            if not buffer.discard():
                show_pending([])
        elif command == "/pending":
            show_pending(buffer.get_pending())
        elif command in {"/audio", "/monitor"}:
            if microphone is None:
                print("Microphone is off. Restart with --microphone to enable it.")
            elif command == "/audio":
                toggle_recording(buffer, [("audio", microphone)])
            else:
                enabled = microphone.toggle_monitor()
                print(f"Microphone monitoring {'on' if enabled else 'off'}.")
        elif command in {"/snapshot", "/preview"}:
            if camera is None:
                print("Camera is off. Restart with --camera to enable it.")
            elif command == "/snapshot":
                buffer.stage("image", camera.snapshot())
            else:
                camera.toggle_capture()
                print("Camera preview toggled.")
        elif command == "/record":
            sources = [(kind, source) for kind, source in (("audio", microphone), ("video", camera))
                       if source is not None]
            if sources:
                toggle_recording(buffer, sources)
            else:
                print("Restart with --microphone and/or --camera to record.")
        elif command.startswith("/"):
            print("Unknown command. Use /help for controls.")
        else:
            buffer.stage("text", command)

        while True:
            try:
                event = buffer.get_event(block=False)
            except Empty:
                break
            handle_event(agent, event)


def run_hotkeys(agent, actions):
    print("Global hotkeys: t = type, hold a = talk, m = monitor, p = preview,")
    print("s = snapshot, r = record, Enter = confirm, Esc = discard, q = quit.")
    print("These keys are active even when another app has focus. Ctrl-C also exits.")
    while actions.running:
        event = actions.get_event()
        if event["type"] == "quit":
            break
        handle_event(agent, event)


def main(arguments):
    with ExitStack() as resources:
        camera = CameraSource(save_dir=arguments.save_dir) if arguments.camera else None
        if camera is not None:
            resources.callback(camera.quit)
        microphone = (
            MicrophoneSource(save=arguments.save, save_dir=arguments.save_dir)
            if arguments.microphone
            else None
        )
        if microphone is not None:
            resources.callback(microphone.quit)
        agent = Agent(model=arguments.model, system_prompt="You are a friendly assistant. " + CHAT_STYLE)
        print(f"Model: {agent.model} | Microphone: {'on' if microphone is not None else 'off'} | "
              f"Camera: {'on' if camera is not None else 'off'}")
        actions = None
        if arguments.global_hotkeys:
            try:
                actions = ActionHandler(
                    camera_source=camera,
                    microphone_source=microphone,
                    auto_send=arguments.auto_send,
                )
            except (PermissionError, ImportError) as error:
                print(f"Global hotkeys unavailable: {error}")
                print("Using terminal controls instead.")
            else:
                resources.callback(actions.quit)
        try:
            if actions is not None:
                run_hotkeys(agent, actions)
            else:
                run_terminal(agent, camera, microphone, auto_send=arguments.auto_send)
        except KeyboardInterrupt:
            print("\nCapture stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture local text, camera, and audio.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--global-hotkeys", action="store_true",
                        help="Use system-wide keyboard shortcuts instead of terminal commands.")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--microphone", action="store_true")
    parser.add_argument("--auto-send", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-dir", default="recordings")
    arguments = parser.parse_args()
    configure_logging(level="WARNING")
    main(arguments)
