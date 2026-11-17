import argparse
import asyncio
import base64
import io
import json
import os
import sys
import wave
from pathlib import Path
from tempfile import NamedTemporaryFile

from websockets.exceptions import WebSocketException

from llm_toolkit import AudioOutput, RealtimeSession, configure_logging

from _common import CHAT_STYLE


class RealtimeDemoError(Exception):
    """A reported server error or an unfinished response."""


def check_error(event):
    if event.get("type") == "error":
        raise RealtimeDemoError(event.get("error", {}).get("message", "The server reported an error."))


async def configure_session(session, output="text"):
    await session.update_session(instructions=CHAT_STYLE, output_modalities=[output])
    while True:
        event = await session.receive_event()
        check_error(event)
        if event.get("type") == "session.updated":
            return event.get("session", {})


async def save_audio(chunks, *, sample_rate, output_dir, playback):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"".join(chunks))
    data = output.getvalue()
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(prefix="realtime-", suffix=".wav", dir=directory, delete=False) as file:
        file.write(data)
        print(f"Audio reply saved: {file.name}")
    if playback:
        try:
            await asyncio.to_thread(AudioOutput().play_wav, data)
        except Exception:
            print("Playback is unavailable; open the saved WAV file to hear the reply.")


async def print_response(session: RealtimeSession):
    """
    Prints text/transcripts and returns PCM chunks when the response completes.

    Args:
        `session` (RealtimeSession): Connected realtime session.
    """

    audio = []
    async for event in session.events():
        event_type = event.get("type")
        check_error(event)
        if event_type in {"response.output_text.delta", "response.text.delta",
                          "response.output_audio_transcript.delta", "response.audio_transcript.delta"}:
            print(event.get("delta", ""), end="", flush=True)
        if event_type in {"response.output_audio.delta", "response.audio.delta"}:
            audio.append(base64.b64decode(event.get("delta", ""), validate=True))
        if event_type == "response.done":
            print()
            response = event.get("response", {})
            if response.get("status") in {"failed", "cancelled", "incomplete"}:
                details = response.get("status_details") or {}
                error = details.get("error") or {}
                raise RealtimeDemoError(error.get("message") or f"Response {response['status']}.")
            return audio
    raise RealtimeDemoError("The connection closed before the reply finished.")


async def read_message(prompt="You: "):
    """Keep WebSocket heartbeats running while the terminal waits for a line."""
    loop = asyncio.get_running_loop()
    result = loop.create_future()

    def ready():
        if result.done():
            return
        line = sys.stdin.readline()
        if line:
            result.set_result(line.strip())
        else:
            result.set_exception(EOFError())

    try:
        descriptor = sys.stdin.fileno()
        loop.add_reader(descriptor, ready)
    except (AttributeError, OSError, ValueError, NotImplementedError):
        # Event loops without fd readers (e.g. Windows) use a daemon thread so
        # exiting with Ctrl-C does not wait for a blocked stdin executor worker.
        import threading

        def read():
            try:
                value = input(prompt).strip()
                loop.call_soon_threadsafe(lambda: None if result.done() else result.set_result(value))
            except EOFError:
                if not loop.is_closed():
                    loop.call_soon_threadsafe(lambda: None if result.done() else result.set_exception(EOFError()))
            except RuntimeError:
                pass  # The application exited while waiting for terminal input.

        threading.Thread(target=read, daemon=True).start()
        return await result
    print(prompt, end="", flush=True)
    try:
        return await result
    finally:
        loop.remove_reader(descriptor)


async def main(arguments):
    key_env = arguments.api_key_env or ("LITELLM_API_KEY" if arguments.url else None)
    api_key = os.environ.get(key_env) if key_env else None
    if arguments.api_key_env and not api_key:
        print(f"Set {arguments.api_key_env} before starting this example.")
        return 1
    try:
        session = RealtimeSession(
            url=arguments.url,
            model=arguments.model,
            api_key=api_key,
            provider_options=arguments.provider_options,
            connect_timeout=arguments.timeout,
        )
        print(f"Connecting: {arguments.model} ({'proxy URL' if arguments.url else 'provider credentials'})")
        await session.connect()
    except (OSError, ValueError, RuntimeError, WebSocketException) as error:
        print(f"Could not connect: {error or type(error).__name__}")
        print("Check the model's realtime support and its provider credentials/settings.")
        if arguments.url:
            print("An explicit --url requires a running server. Omit --url to connect by provider/model.")
        return 1

    try:
        configuration = await asyncio.wait_for(configure_session(session, arguments.output), timeout=arguments.timeout)
        output = configuration.get("output_modalities") or configuration.get("modalities") or [arguments.output]
        print(f"Connected. Output: {', '.join(output)}. Type a message, or /quit to stop.")

        while True:
            message = arguments.message if arguments.message is not None else await read_message()
            if message == "/quit":
                break
            if not message:
                if arguments.message is not None:
                    break
                continue

            print("Agent: ", end="", flush=True)
            await session.send_text(message)
            audio = await asyncio.wait_for(print_response(session), timeout=arguments.timeout)
            if audio:
                await save_audio(audio, sample_rate=session.sample_rate,
                                 output_dir=arguments.output_dir, playback=not arguments.no_playback)

            if arguments.message is not None:
                break
    except TimeoutError:
        print("\nTimed out waiting for the realtime server. The message was not retried automatically.")
        return 1
    except (RealtimeDemoError, OSError, ValueError, WebSocketException) as error:
        print(f"\nRealtime chat stopped: {error}")
        return 1
    except EOFError:
        pass
    finally:
        await session.close()
    return 0


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Chat with a realtime provider through LiteLLM.")
    parser.add_argument("--url", help="Optional OpenAI-compatible realtime proxy URL; otherwise connect by model.")
    parser.add_argument("--model", default="openai/gpt-realtime")
    parser.add_argument("--output", choices=("text", "audio"), default="text", help="Requested reply type; some providers require audio.")
    parser.add_argument("--no-playback", action="store_true", help="Save audio replies without playing them.")
    parser.add_argument("--output-dir", default="recordings", help="Directory for audio reply WAV files.")
    parser.add_argument(
        "--message",
        help="Send one message instead of opening the interactive prompt.",
    )
    parser.add_argument("--api-key-env", help="Override provider credential discovery with this environment variable.")
    parser.add_argument("--provider-options", default="{}", help='Provider settings as JSON, e.g. {"api_base":"https://..."}.')
    parser.add_argument("--timeout", type=float, default=30, help="Connection and response timeout in seconds.")
    arguments = parser.parse_args(argv)
    try:
        arguments.provider_options = json.loads(arguments.provider_options)
    except json.JSONDecodeError:
        parser.error("--provider-options must be a JSON object")
    if not isinstance(arguments.provider_options, dict):
        parser.error("--provider-options must be a JSON object")
    return arguments


if __name__ == "__main__":
    arguments = parse_args()
    configure_logging(level="WARNING")
    try:
        raise SystemExit(asyncio.run(main(arguments)))
    except KeyboardInterrupt:
        print("\nRealtime chat stopped.")
