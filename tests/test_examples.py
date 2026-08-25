import ast
import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock


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
        example_path = Path(__file__).parents[1] / "examples" / "input_buffer.py"
        module_spec = importlib.util.spec_from_file_location(
            "input_buffer_example",
            example_path,
        )
        example = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(example)

        commands = iter(["stage text hello", "confirm", "quit"])
        output = io.StringIO()
        with (
            mock.patch("builtins.input", side_effect=lambda _: next(commands)),
            contextlib.redirect_stdout(output),
        ):
            example.main()

        self.assertIn("Pending:", output.getvalue())
        self.assertIn("Send handler received:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
