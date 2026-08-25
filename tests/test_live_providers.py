import os
import unittest


LIVE_MODELS = {
    "OPENAI_API_KEY": "openai/gpt-4.1-nano",
    "ANTHROPIC_API_KEY": "anthropic/claude-haiku-4-5",
    "GEMINI_API_KEY": "gemini/gemini-3.5-flash-lite",
}


@unittest.skipUnless(
    os.getenv("RUN_LIVE_TESTS") == "1",
    "Set RUN_LIVE_TESTS=1 to make live provider requests",
)
class LiveProviderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dotenv import load_dotenv

        load_dotenv()

    def test_text_completion_for_configured_providers(self):
        from llm_toolkit import Agent

        tested_providers = 0

        for key_name, model in LIVE_MODELS.items():
            if not os.getenv(key_name):
                continue

            with self.subTest(model=model):
                agent = Agent(
                    model=model,
                    system_prompt="Follow the user's response format exactly.",
                )
                agent.send_text("Reply with exactly: PROVIDER_OK")
                response = agent.get_response()
                content = response.choices[0].message.content

                self.assertEqual(content.strip(), "PROVIDER_OK")
                self.assertGreater(agent.token_usage["total_tokens"], 0)
                tested_providers += 1

        self.assertGreater(tested_providers, 0, "No provider API keys were configured")


if __name__ == "__main__":
    unittest.main()
