import argparse
import json

from llm_toolkit import CharacterSheet, Director, configure_logging


DEFAULT_CHARACTER = CharacterSheet(
    name="Mara Vale",
    description="A navigator aboard a remote research vessel.",
    personality="Observant, dryly funny, and careful with uncertain facts.",
    background="Mara maps unstable routes through deep space.",
    scenario="A visitor has reached the ship's navigation room.",
    goals=["Help the visitor understand the current route."],
    instructions=["Stay in character.", "Use concise dialogue."],
)


def load_character(file_path: str = None) -> CharacterSheet:
    if file_path is None:
        return DEFAULT_CHARACTER

    with open(file_path, encoding="utf-8") as character_file:
        return CharacterSheet.from_dict(json.load(character_file))


def main(model: str, character_path: str = None):
    character = load_character(character_path)
    director = Director()
    director.add_character(character, model=model)
    print(f"Talking to {character.name}. Enter /quit to stop.")

    while True:
        text = input("You: ").strip()
        if text == "/quit":
            break
        if not text:
            continue

        director.send_text(character.name, text)
        print(f"{character.name}: {director.speak(character.name, listeners=[])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with a character sheet agent.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--character", help="Path to a character sheet JSON file.")
    arguments = parser.parse_args()
    configure_logging()
    main(arguments.model, arguments.character)
