import argparse
from pathlib import Path

from llm_toolkit import Agent, configure_logging

from _common import get_response_text


def main(model: str, session_path: str):
    path = Path(session_path)
    agent = Agent(model=model)

    if path.exists():
        agent.load_session(str(path))
        print(f"Loaded {len(agent.messages)} stored messages.")

    print("Enter a message, /clear to reset the history, or /quit to save and stop.")

    while True:
        message = input("You: ").strip()
        if message == "/quit":
            break
        if message == "/clear":
            agent.clear_messages()
            print("Conversation history cleared.")
            continue
        if not message:
            continue

        agent.send_text(message)
        print(f"Agent: {get_response_text(agent.get_response())}")
        agent.save_session(str(path))

    print(f"Saved: {agent.save_session(str(path))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume and persist an agent session.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--session", default="recordings/example-session.json")
    arguments = parser.parse_args()
    configure_logging()
    main(arguments.model, arguments.session)
