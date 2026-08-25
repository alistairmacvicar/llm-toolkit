import argparse
import asyncio
import os

from llm_toolkit import RealtimeSession, configure_logging


async def print_response(session: RealtimeSession):
    """
    Prints realtime text events until the current response is complete.

    Args:
        `session` (RealtimeSession): Connected realtime session.
    """

    async for event in session.events():
        event_type = event.get("type")
        if event_type in {"response.output_text.delta", "response.text.delta"}:
            print(event.get("delta", ""), end="", flush=True)
        if event_type == "response.done":
            print()
            break


async def main(arguments):
    api_key = os.environ.get(arguments.api_key_env)
    session = RealtimeSession(
        url=arguments.url,
        model=arguments.model,
        api_key=api_key,
    )

    async with session:
        await session.update_session(
            instructions="Be concise and conversational.",
            output_modalities=["text"],
        )
        print("Enter a message, or /quit to stop.")

        while True:
            message = arguments.message or input("You: ").strip()
            if message == "/quit":
                break
            if not message:
                continue

            print("Agent: ", end="", flush=True)
            await session.send_text(message)
            await print_response(session)

            if arguments.message is not None:
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a LiteLLM realtime proxy.")
    parser.add_argument("--url", default="ws://localhost:4000/v1/realtime")
    parser.add_argument("--model", default="openai/gpt-realtime")
    parser.add_argument(
        "--message",
        help="Send one message instead of opening the interactive prompt.",
    )
    parser.add_argument("--api-key-env", default="LITELLM_API_KEY")
    arguments = parser.parse_args()
    configure_logging()
    asyncio.run(main(arguments))
