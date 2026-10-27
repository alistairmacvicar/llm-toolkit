import unittest

from llm_toolkit.agents.character import CharacterSheet
from llm_toolkit.agents.director import Director


class CharacterSheetTests(unittest.TestCase):
    def test_character_sheet_renders_only_configured_sections(self):
        character = CharacterSheet(
            name="Ada",
            personality="Curious and precise.",
            goals=["Explain difficult ideas clearly."],
            instructions=["Stay in character."],
        )

        prompt = character.to_system_prompt()

        self.assertIn("You are Ada.", prompt)
        self.assertIn("Personality:\nCurious and precise.", prompt)
        self.assertIn("Goals:\n- Explain difficult ideas clearly.", prompt)
        self.assertIn("Instructions:\n- Stay in character.", prompt)
        self.assertNotIn("Background:", prompt)

    def test_character_sheet_round_trips_through_dictionary(self):
        character = CharacterSheet(
            name="Lin",
            description="A systems engineer.",
            goals=["Find the simplest reliable solution."],
        )

        restored = CharacterSheet.from_dict(character.to_dict())

        self.assertEqual(restored, character)


class DirectorTests(unittest.TestCase):
    def test_hierarchy_delegates_only_to_children_and_returns_a_character(self):
        root = FakeAgent('{"speaker":"Sub","instruction":"Keep it brief","reason":"Their specialty"}')
        sub = FakeAgent('{"speaker":"Ada","reason":"Addressed by name"}')
        cast = {"Ada": FakeAgent("Hello"), "Lin": FakeAgent("Hi")}
        director = Director(cast)
        director.set_hierarchy({"Lead": ["Sub", "Lin"], "Sub": ["Ada"]}, {"Lead": root, "Sub": sub}, root="Lead")
        result = director.public_turn("Ada?", ai_turns=1)
        self.assertEqual(result, [{"speaker": "Ada", "content": "Hello"}])
        self.assertEqual([d["director"] for d in director.get_decisions()], ["Lead", "Sub"])
        self.assertIn("Parent direction:", sub.messages[0])
        self.assertIn("Ada: Hello", cast["Lin"].messages)
        with self.assertRaises(ValueError):
            director.set_hierarchy({"Lead": ["Sub"], "Sub": ["Lead", "Ada", "Lin"]}, {"Lead": root, "Sub": sub}, root="Lead")
        self.assertEqual(director.hierarchy["Sub"], ["Ada"])

    def test_director_routes_messages_without_loading_model_dependencies(self):
        first = FakeAgent("first response")
        second = FakeAgent("second response")
        director = Director({"first": first, "second": second})

        director.whisper("first", "second", "secret")
        director.overhear("first", "announcement")

        self.assertEqual(
            second.messages,
            ["first: secret", "first: announcement"],
        )
        self.assertEqual(first.messages, [])
        self.assertEqual(director.get_responses()["first"], "first response")

    def test_director_runs_bounded_conversation(self):
        first = FakeAgent("first response")
        second = FakeAgent("second response")
        director = Director({"first": first, "second": second})

        transcript = director.converse(
            ["first", "second"],
            opening_message="Begin",
        )

        self.assertEqual(
            transcript,
            [
                {"speaker": "first", "content": "first response"},
                {"speaker": "second", "content": "second response"},
            ],
        )
        self.assertEqual(
            first.messages,
            ["Begin", "second: second response"],
        )
        self.assertEqual(second.messages, ["first: first response"])

    def test_director_agent_selects_and_directs_speakers(self):
        first = FakeAgent("first response")
        second = FakeAgent("second response")
        coordinator = FakeAgent(
            [
                '{"speaker": "first", "instruction": "Open the debate."}',
                '{"speaker": "second", "instruction": "Challenge that claim."}',
            ]
        )
        director = Director(
            {"first": first, "second": second},
            director_agent=coordinator,
        )

        transcript = director.coordinate(
            ["first", "second"],
            opening_message="Discuss the proposal.",
            turns=2,
            context="The user addressed first directly.",
        )

        self.assertEqual(
            transcript,
            [
                {"speaker": "first", "content": "first response"},
                {"speaker": "second", "content": "second response"},
            ],
        )
        self.assertIn("Director: Open the debate.", first.contexts[-1])
        self.assertIn("Director: Challenge that claim.", second.contexts[-1])
        self.assertEqual(len(coordinator.messages), 1)
        self.assertEqual(len(director.get_decisions()), 2)
        self.assertIn("Reply slot: 1; maximum slots: 2", director.get_decisions()[0]["prompt"])
        self.assertIn("The user addressed first directly.", coordinator.messages[0])

    def test_public_turn_is_overheard_and_respects_configured_turns(self):
        first = FakeAgent("first response")
        second = FakeAgent("second response")
        coordinator = FakeAgent(
            '{"speaker": "second", "instruction": "Respond to the user."}'
        )
        director = Director(
            {"first": first, "second": second},
            director_agent=coordinator,
            ai_turns=1,
        )
        self.assertIs(director.director_agent, coordinator)
        self.assertNotIn(coordinator, director.agents.values())
        director.character_sheets["first"] = CharacterSheet(
            name="First Character",
            personality="Direct and thoughtful.",
        )

        responses = director.public_turn(
            "What do you think?",
            target="second",
        )

        self.assertEqual(
            responses,
            [{"speaker": "second", "content": "second response"}],
        )
        self.assertIn("User to second: What do you think?", first.messages)
        self.assertIn("User to second: What do you think?", second.messages)
        self.assertIn("second: second response", first.messages)
        self.assertIn("You are First Character.", coordinator.messages[0])
        self.assertIn("Generate 1 total AI speaking turns", coordinator.messages[0])
        self.assertIn("you must select second", coordinator.messages[0])

    def test_private_turn_is_only_delivered_to_the_target(self):
        first = FakeAgent("private response")
        second = FakeAgent("second response")
        coordinator = FakeAgent("unused")
        director = Director(
            {"first": first, "second": second},
            director_agent=coordinator,
        )

        response = director.private_turn("first", "Keep this quiet.")

        self.assertEqual(response, "private response")
        self.assertEqual(first.messages, ["User [private]: Keep this quiet."])
        self.assertEqual(second.messages, [])
        self.assertEqual(coordinator.messages, [])
        self.assertEqual(
            director.get_private_transcript("first"),
            [
                {"speaker": "User", "content": "Keep this quiet."},
                {"speaker": "first", "content": "private response"},
            ],
        )


class FakeAgent:
    def __init__(self, response):
        self.response = response
        self.messages = []
        self.response_index = 0
        self.contexts = []

    def send_text(self, text):
        self.messages.append(text)

    def get_response(self, *, context=None):
        self.contexts.append(context)
        if isinstance(self.response, list):
            response = self.response[self.response_index]
            self.response_index += 1
            return response
        return self.response


if __name__ == "__main__":
    unittest.main()
