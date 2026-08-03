import argparse

from llm_toolkit import ActionHandler, Agent, configure_logging
from llm_toolkit.media.camera_source import CameraSource
from llm_toolkit.media.microphone_source import MicrophoneSource


def run_local_session(
    model: str,
    use_camera: bool = False,
    use_microphone: bool = False,
    auto_send: bool = False,
    save_recordings: bool = False,
):
    """
    Runs an interactive local multimodal session.

    Args:
        `model` (str): LiteLLM model used for responses.
        `use_camera` (bool): Whether to start the default camera.
        `use_microphone` (bool): Whether to start the default microphone.
        `auto_send` (bool): Whether to send captures without confirmation.
        `save_recordings` (bool): Whether to save microphone recordings.
    """

    camera = CameraSource() if use_camera else None
    microphone = (
        MicrophoneSource(save=save_recordings) if use_microphone else None
    )
    agent = Agent(model=model)
    actions = ActionHandler(
        camera_source=camera,
        microphone_source=microphone,
        auto_send=auto_send,
    )

    try:
        while actions.running:
            event = actions.get_event()

            match event["type"]:
                case "send":
                    agent.send_inputs(event["value"])
                    response = agent.get_response()
                    print(_get_response_text(response))
                case "quit":
                    break
    finally:
        actions.quit()


def _get_response_text(response) -> str:
    message = response.choices[0].message
    content = message.content

    if content is not None:
        return str(content)

    audio = getattr(message, "audio", None)
    if audio is not None:
        return str(audio.transcript)

    return ""


def _parse_args():
    parser = argparse.ArgumentParser(description="Run a local llm-toolkit session.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--microphone", action="store_true")
    parser.add_argument("--auto-send", action="store_true")
    parser.add_argument("--save-recordings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    configure_logging()
    arguments = _parse_args()
    run_local_session(
        model=arguments.model,
        use_camera=arguments.camera,
        use_microphone=arguments.microphone,
        auto_send=arguments.auto_send,
        save_recordings=arguments.save_recordings,
    )
