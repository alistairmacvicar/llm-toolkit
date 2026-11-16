"""Verify automatic configuration in fresh Python processes."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


VALUE_KEY = "LLM_TOOLKIT_TEST_DOTENV_VALUE"
PARENT_KEY = "LLM_TOOLKIT_TEST_DOTENV_PARENT"


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.project = self.directory / "project"
        self.project.mkdir()
        self.environment = os.environ.copy()
        for key in (VALUE_KEY, PARENT_KEY, "PYTHON_DOTENV_DISABLED"):
            self.environment.pop(key, None)
        self.environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
        self.environment["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

    def import_environment(self, cwd=None, import_statement="import llm_toolkit"):
        # A real script outside the project catches searches rooted at the script
        # or installed library instead of the application's working directory.
        script = self.directory / "probe.py"
        script.write_text(
            f"import json, os\n{import_statement}\n"
            f"print(json.dumps([os.getenv({VALUE_KEY!r}), os.getenv({PARENT_KEY!r})]))\n",
            encoding="utf-8",
        )
        result = subprocess.run(
            [sys.executable, str(script)],
            cwd=cwd or self.project,
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return json.loads(result.stdout)

    def test_loads_dotenv_from_working_directory(self):
        (self.project / ".env").write_text(f'{VALUE_KEY}="from .env"\n', encoding="utf-8")
        self.assertEqual(self.import_environment(), ["from .env", None])

    def test_loads_dotenv_from_parent_directory(self):
        (self.project / ".env").write_text(f"{VALUE_KEY}=parent\n", encoding="utf-8")
        nested = self.project / "nested" / "scripts"
        nested.mkdir(parents=True)
        self.assertEqual(self.import_environment(cwd=nested), ["parent", None])

    def test_loads_only_nearest_dotenv(self):
        (self.directory / ".env").write_text(
            f"{VALUE_KEY}=parent\n{PARENT_KEY}=parent-only\n", encoding="utf-8"
        )
        (self.project / ".env").write_text(f"{VALUE_KEY}=nearest\n", encoding="utf-8")
        self.assertEqual(self.import_environment(), ["nearest", None])

    def test_preserves_existing_environment_including_empty_values(self):
        (self.project / ".env").write_text(f"{VALUE_KEY}=from-file\n", encoding="utf-8")
        for value in ("already-set", ""):
            with self.subTest(value=value):
                self.environment[VALUE_KEY] = value
                self.assertEqual(self.import_environment(), [value, None])

    def test_import_succeeds_without_dotenv(self):
        self.assertEqual(self.import_environment(), [None, None])

    def test_agent_imports_also_load_dotenv(self):
        (self.project / ".env").write_text(f"{VALUE_KEY}=from-file\n", encoding="utf-8")
        for statement in (
            "from llm_toolkit import Agent",
            "from llm_toolkit.agents.agent import Agent",
        ):
            with self.subTest(import_statement=statement):
                self.assertEqual(
                    self.import_environment(import_statement=statement), ["from-file", None]
                )


if __name__ == "__main__":
    unittest.main()
