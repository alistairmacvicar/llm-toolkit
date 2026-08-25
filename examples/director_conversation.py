import argparse

from llm_toolkit import Agent, CharacterSheet, Director, configure_logging


CHARACTERS = [
    CharacterSheet(
        name="Iris",
        description="An urban planner focused on public spaces.",
        personality="Optimistic, concrete, and collaborative.",
        goals=["Propose a welcoming town square."],
        instructions=["Build on useful ideas from other speakers."],
    ),
    CharacterSheet(
        name="Rowan",
        description="A municipal engineer responsible for delivery and maintenance.",
        personality="Pragmatic, candid, and detail-oriented.",
        goals=["Keep proposals affordable and maintainable."],
        instructions=["Point out tradeoffs without ending the discussion."],
    ),
    CharacterSheet(
        name="Sol",
        description="A local arts organizer representing community groups.",
        personality="Imaginative, warm, and persuasive.",
        goals=["Make room for diverse community events."],
        instructions=["Ask others to make their plans more inclusive."],
    ),
]


def main(model: str, opening: str = None, turns: int = 6):
    coordinator = Agent(
        model=model,
        system_prompt=(
            "You direct a focused ensemble conversation. Select the character whose "
            "perspective would move the discussion forward, and give brief direction."
        ),
    )
    director = Director(director_agent=coordinator)

    for character in CHARACTERS:
        director.add_character(character, model=model)

    print("Enter a scene prompt, or /quit to stop.")

    while True:
        current_opening = opening or input("Scene: ").strip()
        if current_opening == "/quit":
            break
        if not current_opening:
            continue

        print(f"User: {current_opening}")
        transcript = director.coordinate(
            [character.name for character in CHARACTERS],
            opening_message=current_opening,
            turns=turns,
        )

        for entry in transcript:
            print(f"{entry['speaker']}: {entry['content']}")

        if opening is not None:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AI-directed cast conversation.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--turns", type=int, default=6)
    parser.add_argument(
        "--opening",
        help="Run one scene instead of opening the interactive prompt.",
    )
    arguments = parser.parse_args()
    configure_logging()
    main(arguments.model, arguments.opening, arguments.turns)
