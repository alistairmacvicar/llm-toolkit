import argparse

from llm_toolkit import Agent, configure_logging

from _common import get_response_text


MODES = {
    "assistant": {
        "system_prompt": "You are a friendly, practical assistant.",
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


def main(model: str):
    agent = Agent(model=model)
    print("Commands: /mode assistant|critic|tutor|voice, /model NAME, /clear, /quit")

    while True:
        text = input("You: ").strip()
        if text == "/quit":
            break
        if text == "/clear":
            agent.clear_messages()
            continue
        if text.startswith("/mode "):
            mode_name = text.removeprefix("/mode ").strip()
            if mode_name not in MODES:
                print(f"Unknown mode: {mode_name}")
                continue
            agent.set_mode(**MODES[mode_name])
            print(f"Mode: {mode_name}")
            continue
        if text.startswith("/model "):
            agent.set_model(text.removeprefix("/model ").strip())
            print(f"Model: {agent.model}")
            continue
        if not text:
            continue

        agent.send_text(text)
        print(f"Agent: {get_response_text(agent.get_response())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Switch agent modes while chatting.")
    parser.add_argument("--model", default="gpt-5")
    arguments = parser.parse_args()
    configure_logging()
    main(arguments.model)
