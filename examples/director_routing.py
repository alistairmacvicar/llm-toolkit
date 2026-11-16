import argparse

from llm_toolkit import Agent, Director, configure_logging

from _common import CHAT_STYLE


def run_round(director: Director, topic: str):
    """
    Runs one manually directed discussion round.

    Args:
        `director` (Director): Director containing the discussion agents.
        `topic` (str): Topic sent to both agents.
    """

    director.broadcast(topic)
    director.whisper(
        "director",
        "Iris",
        "React naturally to what the user just said. Keep it brief.",
    )
    iris_reply = director.speak("Iris")
    print(f"Iris: {iris_reply}")

    director.whisper("director", "Rowan", "Pick up on something Iris said in your own way.")
    rowan_reply = director.speak("Rowan")
    print(f"Rowan: {rowan_reply}")


def main(model: str, topic: str = None):
    iris = Agent(
        model=model,
        system_prompt="You are Iris, warm and curious, chatting over coffee. " + CHAT_STYLE,
    )
    rowan = Agent(
        model=model,
        system_prompt="You are Rowan, laid-back and dryly funny, chatting over coffee. " + CHAT_STYLE,
    )
    director = Director({"Iris": iris, "Rowan": rowan})

    print("Manual routing demo: Python chooses when Iris and Rowan speak.")
    print("Plain message       Both hear you; Iris replies, then Rowan reacts to Iris")
    print("/converse MESSAGE   Iris and Rowan talk to each other; you watch")
    print("/overhear MESSAGE   Both hear the message without replying")
    print("/quit               Stop")
    print("For an AI-facilitated group chat with you, use facilitated_conversation.py.")

    while True:
        current_topic = topic or input("Topic: ").strip()
        if current_topic == "/quit":
            break
        if current_topic.startswith("/overhear "):
            message = current_topic.removeprefix("/overhear ").strip()
            director.overhear("user", message)
            print("Both agents heard the message.")
            continue
        if current_topic.startswith("/converse "):
            message = current_topic.removeprefix("/converse ").strip()
            print("Iris and Rowan are talking to each other:")
            transcript = director.converse(
                ["Iris", "Rowan"],
                opening_message=message,
                rounds=1,
            )
            for entry in transcript:
                print(f"{entry['speaker'].title()}: {entry['content']}")
            continue
        if not current_topic:
            continue

        run_round(director, current_topic)
        if topic is not None:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Route messages between agents.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--topic",
        help="Run one discussion instead of opening the interactive prompt.",
    )
    arguments = parser.parse_args()
    configure_logging(level="WARNING")
    main(arguments.model, arguments.topic)
