import argparse

from llm_toolkit import Agent, configure_logging

from _common import get_response_text


def main(model: str):
    agent = Agent(model=model)
    print("Enter a message, or /quit to stop.")

    while True:
        text = input("You: ").strip()
        if text == "/quit":
            break
        if not text:
            continue

        agent.send_text(text)
        response = agent.get_response()
        print(f"Agent: {get_response_text(response)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with one text agent.")
    parser.add_argument("--model", default="gpt-5")
    arguments = parser.parse_args()
    configure_logging()
    main(arguments.model)
