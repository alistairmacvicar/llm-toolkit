import argparse

from llm_toolkit import ActionHandler, Agent, configure_logging
from llm_toolkit.media.camera_source import CameraSource
from llm_toolkit.media.microphone_source import MicrophoneSource

from _common import get_response_text


def main(arguments):
    camera = CameraSource(save_dir=arguments.save_dir) if arguments.camera else None
    microphone = (
        MicrophoneSource(save=arguments.save, save_dir=arguments.save_dir)
        if arguments.microphone
        else None
    )
    agent = Agent(model=arguments.model)
    actions = ActionHandler(
        camera_source=camera,
        microphone_source=microphone,
        auto_send=arguments.auto_send,
    )

    try:
        while actions.running:
            event = actions.get_event()
            if event["type"] == "send":
                agent.send_inputs(event["value"])
                print(get_response_text(agent.get_response()))
            elif event["type"] == "quit":
                break
    finally:
        actions.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture local text, camera, and audio.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--microphone", action="store_true")
    parser.add_argument("--auto-send", action="store_true")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--save-dir", default="recordings")
    arguments = parser.parse_args()
    configure_logging()
    main(arguments)
