import argparse
import json
from pathlib import Path

from llm_toolkit import Agent, CharacterSheet, Director, configure_logging

from _common import DIRECTOR_PROMPT


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
    director.optional_replies = True
    director.avoid_repeat = False  # Let a character answer back after someone else speaks.

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
    print("/turns NUMBER                 Set the maximum replies per message")
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
        print(f"Up to {director.ai_turns} replies per message; the conversation may pause sooner.")
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
    print("Everyone hears public messages and can talk to you or each other.")
    print(f"Up to {director.ai_turns} replies before your next turn; the group may pause sooner.")
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
        description="Have a relaxed group chat with public messages and private asides."
    )
    parser.add_argument("--model", default="openai/gpt-5")
    parser.add_argument("--director-model")
    parser.add_argument(
        "--turns", type=int, default=4,
        help="Maximum character replies per message; the director may stop earlier.",
    )
    parser.add_argument(
        "--cast",
        default=str(Path(__file__).with_name("ensemble_cast.json")),
    )
    arguments = parser.parse_args()
    configure_logging(level="WARNING")
    main(arguments)
