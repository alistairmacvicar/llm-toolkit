import ast
import contextlib
import importlib.util
import io
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def load_example(name):
    examples_dir = Path(__file__).parents[1] / "examples"
    spec = importlib.util.spec_from_file_location(name, examples_dir / f"{name}.py")
    example = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "path", [str(examples_dir), *sys.path]):
        spec.loader.exec_module(example)
    return example


class ExampleTests(unittest.TestCase):
    def test_every_example_compiles(self):
        examples_dir = Path(__file__).parents[1] / "examples"
        examples = list(examples_dir.glob("*.py"))

        self.assertGreaterEqual(len(examples), 10)
        for example in examples:
            source = example.read_text(encoding="utf-8")
            compile(source, str(example), "exec")

    def test_every_user_facing_example_has_a_main_entry_point(self):
        examples_dir = Path(__file__).parents[1] / "examples"

        for example in examples_dir.glob("*.py"):
            if example.name == "_common.py":
                continue

            source = example.read_text(encoding="utf-8")
            syntax_tree = ast.parse(source)
            function_names = {
                node.name
                for node in syntax_tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }

            self.assertIn("main", function_names, example.name)
            self.assertIn('if __name__ == "__main__":', source, example.name)

    def test_input_buffer_example_runs_an_interactive_session(self):
        example = load_example("input_buffer")

        commands = iter(["stage text hello", "confirm", "quit"])
        output = io.StringIO()
        with (
            mock.patch("builtins.input", side_effect=lambda _: next(commands)),
            contextlib.redirect_stdout(output),
        ):
            example.main()

        self.assertIn("Pending:", output.getvalue())
        self.assertIn("Send handler received:", output.getvalue())


class ConversationExampleTests(unittest.TestCase):
    def setUp(self):
        from llm_toolkit import Agent

        support = mock.patch.object(Agent, "_update_model_support")
        support.start()
        self.addCleanup(support.stop)
        completion = mock.patch(
            "llm_toolkit.agents.agent.completion",
            side_effect=AssertionError("Unexpected provider call in an offline test"),
        )
        self.completion = completion.start()
        self.addCleanup(completion.stop)

    def queue_responses(self, *contents):
        self.completion.side_effect = [
            SimpleNamespace(
                choices=[SimpleNamespace(message={"role": "assistant", "content": content})],
                usage=None,
            )
            for content in contents
        ]

    def create_facilitated_conversation(self):
        example = load_example("facilitated_conversation")
        arguments = SimpleNamespace(
            model="test/model", director_model=None, turns=4,
            cast=str(Path(example.__file__).with_name("ensemble_cast.json")),
        )
        return example, example.create_conversation(arguments)

    def test_addressed_reply_can_end_before_the_reply_limit(self):
        example, director = self.create_facilitated_conversation()
        self.queue_responses(
            '{"speaker": "Rowan Hale"}', "Probably an old comedy. You?",
            '{"speaker": null, "reason": "Leave room for the user to answer."}',
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            example.handle_command(director, "@Rowan Hale: What would you watch?")

        self.assertEqual(self.completion.call_count, 3)
        self.assertEqual(output.getvalue().strip(), "Rowan Hale: Probably an old comedy. You?")
        self.assertEqual(len(director.get_public_transcript()), 2)
        self.assertEqual(director.get_decisions()[0]["candidates"], ["Rowan Hale"])
        self.assertIn("Rowan Hale", director.get_decisions()[1]["candidates"])
        self.assertIn("Probably an old comedy.", json.dumps(director.agents["Iris Chen"].messages))

    def test_every_character_hears_the_exchange_and_a_speaker_can_answer_back(self):
        _, director = self.create_facilitated_conversation()
        opening = "Going out for coffee defeats the point of a lazy Sunday."
        replies = [
            "Rowan, surely a coffee and a little walk count as lazy?",
            "Only if you're walking to the sofa, Iris. What about the queue?",
            "Fair point, Rowan. I'll find us somewhere with an empty window seat.",
        ]
        self.queue_responses(
            '{"speaker": "Iris Chen"}', replies[0],
            '{"speaker": "Rowan Hale"}', replies[1],
            '{"speaker": "Iris Chen"}', replies[2],
            '{"speaker": null}',
        )

        transcript = director.public_turn(opening)

        self.assertEqual([entry["speaker"] for entry in transcript],
                         ["Iris Chen", "Rowan Hale", "Iris Chen"])
        self.assertEqual(self.completion.call_count, 7)
        for agent in director.agents.values():
            history = json.dumps(agent.messages)
            for message in [opening, *replies]:
                self.assertIn(message, history)
        # The next speaker receives the previous reply before generating their own.
        rowan_request = json.dumps(self.completion.call_args_list[3].kwargs["messages"])
        iris_follow_up = json.dumps(self.completion.call_args_list[5].kwargs["messages"])
        self.assertIn(replies[0], rowan_request)
        self.assertIn(replies[1], iris_follow_up)
        self.assertEqual(len(director.get_public_transcript()), 4)

    def test_regular_director_demo_allows_a_back_and_forth_by_default(self):
        example = load_example("director_conversation")
        self.queue_responses(
            '{"speaker": "Iris"}', "Rowan, coffee outside?",
            '{"speaker": "Rowan"}', "Iris, is the sofa coming too?",
            '{"speaker": "Iris"}', "Only if you carry it.",
            '{"speaker": null}',
        )
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            example.main(model="test/model", opening="It's a lazy Sunday.")

        self.assertEqual(self.completion.call_count, 7)
        self.assertIn("Iris: Rowan, coffee outside?", output.getvalue())
        self.assertIn("Rowan: Iris, is the sofa coming too?", output.getvalue())
        self.assertIn("Iris: Only if you carry it.", output.getvalue())

    def test_director_chat_keeps_earlier_turns_in_facilitator_context(self):
        example = load_example("director_conversation")
        self.queue_responses(
            '{"speaker": "Iris"}', "A free Sunday sounds lovely.", '{"speaker": null}',
            '{"speaker": "Rowan"}', "You said Sunday was free.", '{"speaker": null}',
        )
        with (
            mock.patch("builtins.input", side_effect=[
                "I've got Sunday free.", "What did I just say?", "/quit",
            ]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            example.main(model="test/model")

        self.assertEqual(self.completion.call_count, 6)
        next_turn_context = json.dumps(self.completion.call_args_list[3].kwargs["messages"])
        self.assertIn("I've got Sunday free.", next_turn_context)
        self.assertIn("A free Sunday sounds lovely.", next_turn_context)
        self.assertIn("What did I just say?", next_turn_context)

    def test_reply_limit_and_private_aside_keep_their_routing(self):
        example, director = self.create_facilitated_conversation()
        self.queue_responses(
            '{"speaker": "Sol Rivera"}', "I'd put some music on.",
            "A surprise movie night sounds fun.",
        )
        with contextlib.redirect_stdout(io.StringIO()):
            example.handle_command(director, "/turns 1")
            example.handle_command(director, "@Sol Rivera: What would you do on Sunday?")
            example.handle_command(director, "/whisper Iris Chen: Let's surprise Rowan.")

        self.assertEqual(self.completion.call_count, 3)
        self.assertEqual(len(director.get_decisions()), 1)
        self.assertEqual(len(director.get_public_transcript()), 2)
        self.assertEqual(len(director.get_private_transcript("Iris Chen")), 2)
        self.assertIn("Let's surprise Rowan.", json.dumps(director.agents["Iris Chen"].messages))
        for agent in [director.director_agent, director.agents["Rowan Hale"], director.agents["Sol Rivera"]]:
            self.assertNotIn("Let's surprise Rowan.", json.dumps(agent.messages))


if __name__ == "__main__":
    unittest.main()
