import argparse

from llm_toolkit import Agent, configure_logging
from llm_toolkit.media.file_source import (
    load_audio,
    load_file,
    load_image_url,
    load_link,
    load_video_urls,
)

from _common import get_response_text


def send_response(agent: Agent):
    """
    Requests and prints a response for the current conversation.

    Args:
        `agent` (Agent): Agent to query.
    """

    print(f"Agent: {get_response_text(agent.get_response())}")


def run_interactive(agent: Agent, sample_fps: int, max_frames: int):
    print("Commands: /image, /audio, /video, /transcription, /clear, /quit")

    while True:
        text = input("You: ").strip()
        if text == "/quit":
            break
        if text == "/clear":
            agent.clear_messages()
            continue
        if text == "/transcription":
            print(f"Transcription: {agent.toggle_transcription()}")
            continue
        if text == "/image":
            path = input("Image path: ").strip()
            caption = input("Caption (optional): ").strip() or None
            agent.send_image(path, caption=caption)
            send_response(agent)
            continue
        if text == "/audio":
            path = input("Audio path: ").strip()
            caption = input("Caption (optional): ").strip() or None
            agent.send_audio(path, caption=caption)
            send_response(agent)
            continue
        if text == "/video":
            path = input("Video path: ").strip()
            caption = input("Caption (optional): ").strip() or None
            agent.send_video(
                path,
                caption=caption,
                sample_fps=sample_fps,
                max_frames=max_frames,
            )
            send_response(agent)
            continue
        if not text:
            continue

        agent.send_text(text)
        send_response(agent)


def main(arguments):
    agent = Agent(
        model=arguments.model,
        transcription_enabled=arguments.transcribe_audio,
    )

    if arguments.text:
        agent.send_text(arguments.text)
    if arguments.image:
        agent.send_image(arguments.image, caption="What is important in this image?")
    if arguments.audio:
        agent.send_audio(arguments.audio, caption="Respond to this recording.")
    if arguments.video:
        agent.send_video(
            arguments.video,
            caption="Summarize what happens in this video.",
            sample_fps=arguments.sample_fps,
            max_frames=arguments.max_frames,
        )

    if agent.messages:
        send_response(agent)
        return

    run_interactive(agent, arguments.sample_fps, arguments.max_frames)


def show_file_helpers(image_path: str, audio_path: str, video_path: str, url: str):
    """Shows the lower-level file helper return values."""

    if image_path:
        print(load_image_url(image_path)[:80])
        print(load_file(image_path)[:80])
    if audio_path:
        print(load_audio(audio_path)[:80])
    if video_path:
        print(f"Sampled frames: {len(load_video_urls(video_path))}")
    if url:
        print(load_link(url)[:80])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send multimodal files to an agent.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--text")
    parser.add_argument("--image")
    parser.add_argument("--audio")
    parser.add_argument("--video")
    parser.add_argument("--sample-fps", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=20)
    parser.add_argument("--transcribe-audio", action="store_true")
    parser.add_argument("--show-file-helpers", action="store_true")
    parser.add_argument("--url")
    arguments = parser.parse_args()
    configure_logging()

    if arguments.show_file_helpers:
        show_file_helpers(
            arguments.image,
            arguments.audio,
            arguments.video,
            arguments.url,
        )
    else:
        main(arguments)
