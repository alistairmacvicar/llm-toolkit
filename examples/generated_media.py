import argparse

from pathlib import Path

from llm_toolkit import Agent, configure_logging


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
            if command == "status":
                print(agent.get_video_status(video_id))
            elif command == "save":
                print(f"Saved video: {agent.save_video(video_id, output)}")
                break
            elif command == "back":
                break


def main(arguments):
    agent = Agent(model=arguments.model)

    if arguments.kind and arguments.media_model and arguments.prompt:
        output = arguments.output or get_default_output(arguments.kind)
        generate(
            agent,
            arguments.kind,
            arguments.media_model,
            arguments.prompt,
            output,
        )
        return

    print("Choose image, speech, or video. Enter /quit to stop.")

    while True:
        kind = input("Media type: ").strip().lower()
        if kind == "/quit":
            break
        if kind not in {"image", "speech", "video"}:
            print("Media type must be image, speech, or video.")
            continue

        media_model = input("LiteLLM media model: ").strip()
        prompt_label = "Text" if kind == "speech" else "Prompt"
        prompt = input(f"{prompt_label}: ").strip()
        default_output = get_default_output(kind)
        output = input(f"Output path [{default_output}]: ").strip() or default_output

        if media_model and prompt:
            generate(agent, kind, media_model, prompt, output)


def get_default_output(kind: str) -> str:
    extension = {"image": ".png", "speech": ".mp3", "video": ".mp4"}[kind]
    return str(Path("recordings") / f"generated-{kind}{extension}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate image, speech, or video media.")
    parser.add_argument("kind", nargs="?", choices=["image", "speech", "video"])
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--media-model")
    parser.add_argument("--prompt")
    parser.add_argument("--output")
    arguments = parser.parse_args()
    configure_logging()
    main(arguments)
