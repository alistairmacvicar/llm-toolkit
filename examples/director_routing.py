import argparse

from llm_toolkit import Agent, Director, configure_logging


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
        "analyst",
        "Begin with the strongest available evidence.",
    )
    analyst_reply = director.speak("analyst")
    print(f"Analyst: {analyst_reply}")

    director.whisper("director", "skeptic", "Focus on one testable risk.")
    skeptic_reply = director.speak("skeptic")
    print(f"Skeptic: {skeptic_reply}")


def main(model: str, topic: str = None):
    analyst = Agent(
        model=model,
        system_prompt="You are an analyst. State one evidence-based observation.",
    )
    skeptic = Agent(
        model=model,
        system_prompt="You are a skeptic. Test claims constructively.",
    )
    director = Director({"analyst": analyst, "skeptic": skeptic})

    print("Enter a discussion topic.")
    print("Commands: /overhear MESSAGE, /converse MESSAGE, /quit")

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
            transcript = director.converse(
                ["analyst", "skeptic"],
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
    configure_logging()
    main(arguments.model, arguments.topic)
