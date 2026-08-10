from dataclasses import asdict, dataclass, field
from typing import Any, final


@final
@dataclass
class CharacterSheet:
    name: str
    description: str = ""
    personality: str = ""
    background: str = ""
    scenario: str = ""
    goals: list[str] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """
        Renders the character sheet as a system prompt.

        Returns:
            str: The rendered system prompt.
        """

        sections = [f"You are {self.name}."]
        fields = [
            ("Description", self.description),
            ("Personality", self.personality),
            ("Background", self.background),
            ("Scenario", self.scenario),
        ]

        for heading, value in fields:
            if value:
                sections.append(f"{heading}:\n{value}")

        if self.goals:
            sections.append("Goals:\n" + "\n".join(f"- {goal}" for goal in self.goals))
        if self.instructions:
            sections.append(
                "Instructions:\n"
                + "\n".join(f"- {instruction}" for instruction in self.instructions)
            )

        return "\n\n".join(sections)

    def to_dict(self) -> dict[str, Any]:
        """
        Converts the character sheet into a dictionary.

        Returns:
            dict[str, Any]: The character sheet fields.
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, character: dict[str, Any]):
        """
        Creates a character sheet from a dictionary.

        Args:
            `character` (dict[str, Any]): Character sheet fields.

        Returns:
            CharacterSheet: The created character sheet.
        """

        return cls(**character)
