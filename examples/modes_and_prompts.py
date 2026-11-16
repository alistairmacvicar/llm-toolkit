import argparse
import base64
from pathlib import Path
from tempfile import NamedTemporaryFile

from openai import APIError

from llm_toolkit import Agent, AudioOutput, configure_logging

from _common import CHAT_STYLE, get_response_text

DEFAULT_VOICE_MODEL = "openai/gpt-audio-1.5"


MODES = {
    "assistant": {
        "system_prompt": "You are a friendly assistant. " + CHAT_STYLE,
        "modalities": ["text"],
    },
    "critic": {
        "system_prompt": (
            "You are a constructive critic. Identify assumptions, then offer one "
            "specific improvement."
        ),
        "modalities": ["text"],
    },
    "tutor": {
        "system_prompt": (
            "You are a patient tutor. Ask one diagnostic question before explaining."
        ),
        "modalities": ["text"],
    },
    "voice": {
        "system_prompt": "Answer conversationally and keep spoken replies concise.",
        "modalities": ["text", "audio"],
        "audio": {"voice": "alloy", "format": "wav"},
    },
}


def show_status(agent: Agent, mode: str):
    print(f"Mode: {mode} | Model: {agent.model} | Output: {', '.join(agent.modalities)}")


def show_response(response, *, output_dir: str, playback: bool):
    print(f"Agent: {get_response_text(response)}")
    audio = getattr(response.choices[0].message, "audio", None)
    data = audio.get("data") if isinstance(audio, dict) else getattr(audio, "data", None)
    if not data:
        return
    wav = base64.b64decode(data, validate=True)
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix="voice-", suffix=".wav", dir=directory, delete=False) as output:
        output.write(wav)
        path = output.name
    print(f"Voice reply saved: {path}")
    if playback:
        try:
            AudioOutput().play_wav(wav)
        except Exception:
            print("Audio playback is unavailable; you can open the saved WAV file.")


def main(model: str, voice_model: str = DEFAULT_VOICE_MODEL, *,
         mode: str = "assistant", playback: bool = True, output_dir: str = "recordings"):
    models = {"text": model, "voice": voice_model}
    route = "voice" if mode == "voice" else "text"
    try:
        agent = Agent(model=models[route], **MODES[mode])
    except (ValueError, APIError) as error:
        print(f"Unable to start: {error}")
        return
    print("Commands: /mode assistant|critic|tutor|voice, /model NAME, /status, /clear, /quit")
    print("Modes automatically select the configured text or voice model.")
    show_status(agent, mode)

    while True:
        text = input("You: ").strip()
        if text == "/quit":
            break
        if text == "/clear":
            agent.clear_messages()
            print("Conversation cleared.")
            continue
        if text == "/status":
            show_status(agent, mode)
            continue
        if text.startswith("/mode "):
            mode_name = text.removeprefix("/mode ").strip()
            if mode_name not in MODES:
                print(f"Unknown mode: {mode_name}")
                continue
            route = "voice" if mode_name == "voice" else "text"
            try:
                agent.set_mode(model=models[route], **MODES[mode_name])
            except (ValueError, APIError) as error:
                print(f"Unable to change mode: {error}")
                continue
            mode = mode_name
            show_status(agent, mode)
            continue
        if text.startswith("/model "):
            selected_model = text.removeprefix("/model ").strip()
            if not selected_model:
                print("Use /model NAME to change the model for the current mode.")
                continue
            try:
                agent.set_mode(model=selected_model, **MODES[mode])
            except (ValueError, APIError) as error:
                print(f"Unable to change model: {error}")
                continue
            models["voice" if mode == "voice" else "text"] = selected_model
            show_status(agent, mode)
            continue
        if not text:
            continue

        previous_messages = list(agent.messages)
        try:
            agent.send_text(text)
            response = agent.get_response()
        except (ValueError, APIError) as error:
            agent.messages[:] = previous_messages
            print(f"Unable to get a reply: {error}")
            print("Change /mode or /model and re-enter your message. Earlier conversation is preserved.")
            continue
        try:
            show_response(response, output_dir=output_dir, playback=playback)
        except (OSError, ValueError):
            print("The reply was received, but its audio could not be saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch agent modes while chatting.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--voice-model", default=DEFAULT_VOICE_MODEL)
    parser.add_argument("--mode", choices=MODES, default="assistant")
    parser.add_argument("--no-playback", action="store_true", help="Save voice replies without playing them.")
    parser.add_argument("--output-dir", default="recordings", help="Directory for voice reply WAV files.")
    arguments = parser.parse_args()
    configure_logging(level="WARNING")
    main(arguments.model, arguments.voice_model, mode=arguments.mode,
         playback=not arguments.no_playback, output_dir=arguments.output_dir)
