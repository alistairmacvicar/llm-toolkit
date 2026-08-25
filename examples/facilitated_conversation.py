import argparse
import json
from pathlib import Path

from llm_toolkit import Agent, CharacterSheet, Director, configure_logging


DIRECTOR_PROMPT = """You facilitate a public conversation between a user and a cast
of characters. Choose the speaker who can move the conversation forward. Consider the
character sheets, who the user addressed, what has already been said, and the remaining
AI turn budget. Characters may respond to the user or converse with one another. Keep
private direction brief and do not invent dialogue for the selected speaker."""


def load_cast(file_path: str) -> list[CharacterSheet]:
    """
    Loads a list of character sheets from JSON.

    Args:
        `file_path` (str): Path to a JSON list of character sheet dictionaries.

    Returns:
        list[CharacterSheet]: Loaded character sheets.
    """

    with open(file_path, encoding="utf-8") as cast_file:
        return [
            CharacterSheet.from_dict(character) for character in json.load(cast_file)
        ]


def create_conversation(arguments) -> Director:
    """
    Creates the director and character agents.

    Args:
        `arguments`: Parsed command-line arguments.

    Returns:
        Director: Configured conversation director.
    """

    director_agent = Agent(
        model=arguments.director_model or arguments.model,
        system_prompt=DIRECTOR_PROMPT,
    )
    director = Director(
        director_agent=director_agent,
        ai_turns=arguments.turns,
    )

    for character in load_cast(arguments.cast):
        director.add_character(character, model=arguments.model)

    return director


def parse_directed_message(command: str) -> tuple[str, str]:
    """
    Parses a character name and message separated by a colon.

    Args:
        `command` (str): Text containing a target and message.

    Returns:
        tuple[str, str]: Target character name and message.
    """

    target, separator, message = command.partition(":")
    if not separator or not target.strip() or not message.strip():
        raise ValueError("Use NAME: message")
    return target.strip(), message.strip()


def print_cast(director: Director):
    print("Cast: " + ", ".join(director.agents))


def print_public_history(director: Director):
    transcript = director.get_public_transcript()
    if not transcript:
        print("No public conversation yet.")
        return

    for entry in transcript:
        target = entry.get("target")
        target_label = f" to {target}" if target and target != "public" else ""
        print(f"{entry['speaker']}{target_label}: {entry['content']}")


def print_private_history(director: Director, name: str):
    transcript = director.get_private_transcript(name)
    if not transcript:
        print(f"No private conversation with {name}.")
        return

    for entry in transcript:
        print(f"{entry['speaker']}: {entry['content']}")


def run_public_turn(director: Director, message: str, target: str = None):
    responses = director.public_turn(message, target=target)
    for response in responses:
        print(f"{response['speaker']}: {response['content']}")


def show_help():
    print("Public message                 Everyone hears it")
    print("@NAME: message                Publicly address one character")
    print("/whisper NAME: message        Private one-to-one turn")
    print("/turns NUMBER                 Set AI turns per public user turn")
    print("/history                      Show public transcript")
    print("/private NAME                 Show one private transcript")
    print("/cast                         Show character names")
    print("/help                         Show commands")
    print("/quit                         Stop")


def handle_command(director: Director, command: str) -> bool:
    """
    Handles one interactive conversation command.

    Args:
        `director` (Director): Active conversation director.
        `command` (str): User message or slash command.

    Returns:
        bool: Whether the conversation should continue.
    """

    if command == "/quit":
        return False
    if command == "/help":
        show_help()
    elif command == "/cast":
        print_cast(director)
    elif command == "/history":
        print_public_history(director)
    elif command.startswith("/private "):
        name = command.removeprefix("/private ").strip()
        print_private_history(director, name)
    elif command.startswith("/turns "):
        director.set_ai_turns(int(command.removeprefix("/turns ").strip()))
        print(f"AI turns per public user turn: {director.ai_turns}")
    elif command.startswith("/whisper "):
        target, message = parse_directed_message(command.removeprefix("/whisper "))
        response = director.private_turn(target, message)
        print(f"{target} [private]: {response}")
    elif command.startswith("@"):
        target, message = parse_directed_message(command[1:])
        run_public_turn(director, message, target=target)
    elif command:
        run_public_turn(director, command)

    return True


def main(arguments):
    director = create_conversation(arguments)
    print_cast(director)
    show_help()

    while True:
        command = input("You: ").strip()
        try:
            if not handle_command(director, command):
                break
        except (KeyError, ValueError) as exception:
            print(f"Unable to process command: {exception}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a facilitated public and private multi-agent conversation."
    )
    parser.add_argument("--model", default="openai/gpt-4.1-nano")
    parser.add_argument("--director-model")
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument(
        "--cast",
        default=str(Path(__file__).with_name("ensemble_cast.json")),
    )
    arguments = parser.parse_args()
    configure_logging()
    main(arguments)
