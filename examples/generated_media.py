import argparse

from pathlib import Path

from openai import APIError

from llm_toolkit import Agent, configure_logging

DEFAULT_IMAGE_MODEL = "openai/gpt-image-2.5-flare"


def report_error(action, media_model, error):
    status = getattr(error, "status_code", None)
    status_label = f", HTTP {status}" if status is not None else ""
    print(f"{action} failed ({media_model}{status_label}): {error}")
    model = media_model.removeprefix("openai/")
    if status == 404 and (model == "sora-2" or model.startswith("sora-2-")):
        print("OpenAI retired Sora 2 and the Videos API on 24 September 2026.")
        print("Choose an available video provider/model; adding openai/ will not fix this endpoint.")
        print("Details: https://developers.openai.com/api/docs/deprecations")
    elif status == 404:
        print("Check the provider/model name, endpoint, and your account's access to that model.")


def try_generate(agent, kind, media_model, prompt, output):
    try:
        generate(agent, kind, media_model, prompt, output)
    except (APIError, OSError, ValueError) as error:
        report_error(f"{kind.capitalize()} generation", media_model, error)
        return False
    return True


def generate(agent: Agent, kind: str, media_model: str, prompt: str, output: str):
    """
    Generates one media item.

    Args:
        `agent` (Agent): Agent used for generation.
        `kind` (str): Image, speech, or video.
        `media_model` (str): LiteLLM media model name.
        `prompt` (str): Generation prompt or speech text.
        `output` (str): Destination path.
    """

    if kind == "image":
        agent.generate_image(
            prompt,
            model=media_model,
            file_path=output,
        )
        print(f"Saved image: {agent.last_image_path}")
    elif kind == "speech":
        agent.generate_speech(
            prompt,
            model=media_model,
            file_path=output,
        )
        print(f"Saved speech: {agent.last_speech_path}")
    else:
        job = agent.generate_video(prompt, model=media_model)
        video_id = job["id"] if isinstance(job, dict) else job.id
        print(f"Video job: {video_id}")
        print("Commands: status, save, or back")

        while True:
            command = input("Video: ").strip().lower()
            if command in {"back", "/quit"}:
                break
            try:
                if command == "status":
                    print(agent.get_video_status(video_id))
                elif command == "save":
                    print(f"Saved video: {agent.save_video(video_id, output)}")
                    break
                else:
                    print("Commands: status, save, or back")
            except (APIError, OSError, ValueError) as error:
                report_error(f"Video {command}", media_model, error)
                print("The video job is still selected. Try status, save, or back.")


def main(arguments):
    agent = Agent(model=arguments.model)
    media_model = arguments.media_model
    if not media_model and arguments.kind == "image":
        media_model = DEFAULT_IMAGE_MODEL

    if arguments.kind and media_model and arguments.prompt:
        output = arguments.output or get_default_output(arguments.kind)
        succeeded = try_generate(
            agent,
            arguments.kind,
            media_model,
            arguments.prompt,
            output,
        )
        return 0 if succeeded else 1

    print("Choose image, speech, or video. Enter /quit to stop.")
    print("Use a model for the selected media type.")

    while True:
        kind = input("Media type: ").strip().lower()
        if kind == "/quit":
            break
        if kind not in {"image", "speech", "video"}:
            print("Media type must be image, speech, or video.")
            continue

        default_model = arguments.media_model or (
            DEFAULT_IMAGE_MODEL if kind == "image" else ""
        )
        model_label = f"LiteLLM {kind} model"
        if default_model:
            model_label += f" [{default_model}]"
        media_model = input(f"{model_label}: ").strip() or default_model
        prompt_label = "Text" if kind == "speech" else "Prompt"
        prompt = input(f"{prompt_label}: ").strip()
        default_output = get_default_output(kind)
        output = input(f"Output path [{default_output}]: ").strip() or default_output

        if media_model and prompt:
            try_generate(agent, kind, media_model, prompt, output)

    return 0


def get_default_output(kind: str) -> str:
    extension = {"image": ".png", "speech": ".mp3", "video": ".mp4"}[kind]
    return str(Path("recordings") / f"generated-{kind}{extension}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate image, speech, or video media."
    )
    parser.add_argument("kind", nargs="?", choices=["image", "speech", "video"])
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Agent chat model; media generation uses --media-model instead.",
    )
    parser.add_argument(
        "--media-model",
        help=f"Model for the selected media type (images default to {DEFAULT_IMAGE_MODEL}).",
    )
    parser.add_argument("--prompt")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    configure_logging()
    raise SystemExit(main(arguments))
