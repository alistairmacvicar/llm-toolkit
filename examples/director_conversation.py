import argparse

from llm_toolkit import Agent, CharacterSheet, Director, configure_logging

from _common import DIRECTOR_PROMPT, GROUP_CHAT_STYLE


CHARACTERS = [
    CharacterSheet(
        name="Iris",
        description="Someone who enjoys finding little cafes and going for walks.",
        personality="Warm, curious, and easygoing.",
        scenario="Having a casual coffee with the user, Rowan, and Sol.",
        instructions=[GROUP_CHAT_STYLE, "Speak only as Iris, without narrating actions."],
    ),
    CharacterSheet(
        name="Rowan",
        description="Someone who enjoys cooking, films, and a quiet weekend.",
        personality="Laid-back, thoughtful, and dryly funny without trying too hard.",
        scenario="Having a casual coffee with the user, Iris, and Sol.",
        instructions=[GROUP_CHAT_STYLE, "Speak only as Rowan, without narrating actions."],
    ),
    CharacterSheet(
        name="Sol",
        description="Someone who enjoys music, games, and trying new hobbies.",
        personality="Playful, friendly, and happy to go with the flow.",
        scenario="Having a casual coffee with the user, Iris, and Rowan.",
        instructions=[GROUP_CHAT_STYLE, "Speak only as Sol, without narrating actions."],
    ),
]


def main(model: str, opening: str = None, turns: int = 4):
    coordinator = Agent(
        model=model,
        system_prompt=DIRECTOR_PROMPT,
    )
    director = Director(director_agent=coordinator, ai_turns=turns)
    director.optional_replies = True
    director.avoid_repeat = False  # Let a character answer back after someone else speaks.

    for character in CHARACTERS:
        director.add_character(character, model=model)

    print("You're having coffee with Iris, Rowan, and Sol. Enter /quit to stop.")
    print(f"Everyone hears the conversation; up to {turns} replies before your next turn.")
    print('Try: "Hey everyone, how do you like to spend a lazy Sunday?"')

    while True:
        current_opening = (opening if opening is not None else input("You: ")).strip()
        if current_opening == "/quit":
            break
        if not current_opening:
            if opening is not None:
                break
            continue

        if opening is not None:
            print(f"You: {current_opening}")
        transcript = director.public_turn(current_opening)

        for entry in transcript:
            print(f"{entry['speaker']}: {entry['content']}")

        if opening is not None:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Have a relaxed, AI-facilitated group chat.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--turns", type=int, default=4,
        help="Maximum character replies per message; the director may stop earlier.",
    )
    parser.add_argument(
        "--opening",
        help="Send one message instead of opening the interactive prompt.",
    )
    arguments = parser.parse_args()
    configure_logging(level="WARNING")
    main(arguments.model, arguments.opening, arguments.turns)
