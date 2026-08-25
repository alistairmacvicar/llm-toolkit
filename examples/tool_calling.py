import argparse

from llm_toolkit import Agent, configure_logging

from _common import get_response_text


def convert_temperature(value: float, target_scale: str) -> dict:
    """Converts a temperature to Celsius or Fahrenheit."""

    if target_scale.lower() == "celsius":
        converted = (value - 32) * 5 / 9
    elif target_scale.lower() == "fahrenheit":
        converted = value * 9 / 5 + 32
    else:
        raise ValueError("target_scale must be celsius or fahrenheit")

    return {"value": round(converted, 2), "scale": target_scale.lower()}


def main(model: str, question: str = None):
    agent = Agent(model=model)
    agent.register_tool(convert_temperature)
    print("Ask for a temperature conversion, or enter /quit to stop.")

    while True:
        current_question = question or input("You: ").strip()
        if current_question == "/quit":
            break
        if not current_question:
            continue

        agent.send_text(current_question)
        response = agent.get_response()
        print(f"Agent: {get_response_text(response)}")
        print(f"Total tokens: {agent.token_usage['total_tokens']}")

        if question is not None:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Let an agent call a Python tool.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument(
        "--question",
        help="Run one question instead of opening the interactive prompt.",
    )
    arguments = parser.parse_args()
    configure_logging()
    main(arguments.model, arguments.question)
