"""Hands-free realtime conversation in the terminal with a native camera preview."""

import argparse
import asyncio
import base64
import json
import math
import queue
from collections import deque
from datetime import datetime, timezone
from uuid import uuid4

import cv2
import sounddevice as sd

from llm_toolkit import RealtimeSession, configure_logging
from llm_toolkit.media.camera_source import CameraSource
from llm_toolkit.media.live_audio import LiveAudio
from llm_toolkit.media.recording import PCMArchive, mux_recordings, recording_directory

from _common import CHAT_STYLE
from realtime_chat import check_error, read_message


class Recordings:
    """Save capture continuously and assemble the camera/microphone file on exit."""

    def __init__(self, directory, model):
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.directory = recording_directory(directory).resolve() / f"live_{stamp}_{uuid4().hex[:8]}"
        self.directory.mkdir(parents=True)
        self.microphone = self.agent = None
        self.agent_item = None
        self.microphone_started_at = None
        self.transcripts = set()
        self.metadata = {"model": model, "started_at": datetime.now(timezone.utc).isoformat(), "status": "starting"}
        self.write_metadata()

    def write_metadata(self):
        (self.directory / "session.json").write_text(json.dumps(self.metadata, indent=2) + "\n", encoding="utf-8")

    def capture(self, pcm, captured_at):
        if self.microphone is None:
            self.microphone = PCMArchive(self.directory, prefix="microphone")
            self.microphone_started_at = captured_at
        self.microphone.write(pcm)

    def reply(self, item, pcm):
        if self.agent is None or item != self.agent_item:
            self.finish_reply()
            self.agent = PCMArchive(self.directory, prefix="agent")
            self.agent_item = item
        self.agent.write(pcm)

    def finish_reply(self):
        archive, self.agent = self.agent, None
        if archive is not None:
            archive.close()

    def transcript(self, role, text, item):
        if not text or (role, item) in self.transcripts:
            return
        self.transcripts.add((role, item))
        entry = {"role": role, "text": text, "item_id": item, "time": datetime.now(timezone.utc).isoformat()}
        with (self.directory / "transcripts.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"{'You' if role == 'user' else 'Agent'}: {text}", flush=True)

    def finish(self, camera, failed=False):
        errors = []
        for archive in (self.microphone, self.agent):
            if archive is not None:
                try:
                    archive.close()
                except Exception as error:
                    errors.append(error)
        if camera is not None and camera.last_recording_path and self.microphone is not None:
            try:
                self.metadata["video_with_audio"] = mux_recordings(
                    camera.last_recording_path, self.microphone.path,
                    audio_offset=self.microphone_started_at - camera.recording_started_at,
                )
            except Exception as error:
                errors.append(error)
        self.metadata.update(status="interrupted" if failed or errors else "finished",
                             finished_at=datetime.now(timezone.utc).isoformat())
        self.write_metadata()
        if errors:
            raise errors[0]


async def configure_session(session, args):
    await session.connect()
    await session.update_session(
        instructions="Have a natural conversation using the person's speech and recent camera frames. "
        "Do not narrate the feed unless asked. " + CHAT_STYLE,
        output_modalities=["audio"],
        audio={
            "input": {"format": {"type": "audio/pcm", "rate": 24000},
                      "transcription": {"model": args.transcription_model},
                      "noise_reduction": {"type": "far_field"},
                      "turn_detection": {"type": "server_vad", "create_response": True,
                                         "interrupt_response": True, "silence_duration_ms": 500}},
            "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": args.voice},
        },
    )
    while True:
        event = await session.receive_event()
        check_error(event)
        if event["type"] == "session.updated":
            return


async def preview_loop(camera, stop):
    while not stop.is_set():
        key = camera.poll_preview()  # asyncio runs on the main thread, including on macOS.
        if key in (27, ord("q")) or not camera.preview_visible:
            stop.set()
            return
        await asyncio.sleep(0.01)


async def terminal_loop(stop):
    while not stop.is_set():
        try:
            command = await read_message(prompt="")
        except EOFError:
            stop.set()
            return
        if command.strip().lower() in {"/quit", "quit", "q"}:
            stop.set()
            return
        if command:
            print("Speak to the agent. Type /quit to stop and save.", flush=True)


async def send_microphone(session, audio, recordings):
    while True:
        audio.check()
        try:
            pcm, captured_at = audio.inputs.get_nowait()
        except queue.Empty:
            await asyncio.sleep(0.005)
            continue
        recordings.capture(pcm, captured_at)
        await session.append_audio(pcm)  # Also archives the exact provider input.


def frame_url(frame):
    height, width = frame.shape[:2]
    if width > 640:
        frame = cv2.resize(frame, (640, max(1, round(height * 640 / width))))
    success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    if not success:
        raise RuntimeError("Could not encode a camera frame")
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


async def send_camera(session, camera, interval):
    frames = deque()
    while True:
        frame = camera.get_frame()
        if frame is not None:
            item_id = "frame_" + uuid4().hex[:24]
            await session.send_event({"type": "conversation.item.create", "item": {
                "id": item_id, "type": "message", "role": "user",
                "content": [{"type": "input_image", "image_url": frame_url(frame)}],
            }})
            frames.append(item_id)
            if len(frames) > 30:
                await session.send_event({"type": "conversation.item.delete", "item_id": frames.popleft()})
        await asyncio.sleep(interval)


async def receive_events(session, audio, recordings):
    response_id = None
    interrupted = deque(maxlen=32)
    async for event in session.events():
        check_error(event)
        kind = event["type"]
        if kind == "response.created":
            response_id = event["response"]["id"]
        elif kind == "input_audio_buffer.speech_started":
            if response_id:
                interrupted.append(response_id)
            playback = audio.interrupt()
            if playback is not None:
                (item_id, content_index), heard_ms = playback
                await session.send_event({"type": "conversation.item.truncate", "item_id": item_id,
                                          "content_index": content_index, "audio_end_ms": heard_ms})
        elif kind in {"response.output_audio.delta", "response.audio.delta"}:
            if event.get("response_id", response_id) in interrupted:
                continue
            pcm = base64.b64decode(event["delta"], validate=True)
            item = (event["item_id"], event.get("content_index", 0))
            recordings.reply(item, pcm)
            audio.play(item, pcm)
        elif kind in {"response.output_audio.done", "response.audio.done"}:
            recordings.finish_reply()
        elif kind == "conversation.item.input_audio_transcription.completed":
            recordings.transcript("user", event.get("transcript"), event.get("item_id"))
        elif kind in {"response.output_audio_transcript.done", "response.audio_transcript.done"}:
            recordings.transcript("assistant", event.get("transcript"), event.get("item_id"))
        elif kind == "conversation.item.input_audio_transcription.failed":
            print("User transcription unavailable:", event.get("error", {}).get("message", "unknown error"))
        elif kind == "response.done" and event.get("response", {}).get("status") == "failed":
            raise RuntimeError(event["response"].get("status_details"))
    raise RuntimeError("The realtime connection closed")


async def run(args):
    recordings = Recordings(args.save_dir, args.model)
    stop = asyncio.Event()
    tasks = []
    camera = audio = session = None
    failed = False
    try:
        audio = LiveAudio(args.input_device, args.output_device)
        print(f"Microphone: {audio.input_info['name']} | Output: {audio.output_info['name']}", flush=True)
        print(f"Recordings: {recordings.directory}", flush=True)
        print("Use headphones. Speak naturally; pause for a reply or interrupt.\n"
              "Type /quit, press Ctrl-C, or press Q/Esc in the preview to stop and save.", flush=True)
        camera = CameraSource(source=args.camera, save_dir=recordings.directory)
        camera.toggle_capture()
        session = RealtimeSession(model=args.model, recording_dir=recordings.directory, connect_timeout=args.timeout)
        tasks = [asyncio.create_task(preview_loop(camera, stop)), asyncio.create_task(terminal_loop(stop)),
                 asyncio.create_task(stop.wait())]
        setup = asyncio.create_task(asyncio.wait_for(configure_session(session, args), timeout=args.timeout))
        tasks.append(setup)
        print(f"Connecting: {args.model}…", flush=True)
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
        if stop.is_set():
            return 0
        tasks.remove(setup)
        camera.toggle_recording()
        audio.start()
        recordings.metadata["status"] = "recording"
        recordings.write_metadata()
        print("Connected. Listening…", flush=True)
        tasks.extend(asyncio.create_task(worker) for worker in (
            send_microphone(session, audio, recordings), send_camera(session, camera, args.frame_interval),
            receive_events(session, audio, recordings),
        ))
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except asyncio.CancelledError:
        # asyncio.run converts Ctrl-C to cancellation. Cleanup still releases
        # devices and finalizes recordings before KeyboardInterrupt is raised.
        raise
    except Exception as error:
        failed = True
        print(f"Conversation stopped: {error or type(error).__name__}", flush=True)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for resource in (audio, camera):
            if resource is not None:
                try:
                    resource.close() if resource is audio else resource.quit()
                except Exception as error:
                    failed = True
                    print(f"Device cleanup failed: {error}", flush=True)
        try:
            if audio is not None:
                while not audio.inputs.empty():
                    recordings.capture(*audio.inputs.get_nowait())
        except Exception as error:
            failed = True
            print(f"Final microphone audio could not be saved: {error}", flush=True)
        if session is not None:
            try:
                await asyncio.wait_for(session.close(), timeout=args.timeout)
            except Exception as error:
                failed = True
                print(f"Session cleanup failed: {error}", flush=True)
        try:
            await asyncio.to_thread(recordings.finish, camera, failed)
            print(f"Recordings saved: {recordings.directory}", flush=True)
        except Exception as error:
            failed = True
            print(f"Recording could not be fully finalized: {error}\nOriginal files: {recordings.directory}", flush=True)
    return int(failed)


def device_argument(value):
    return int(value) if value.isdecimal() else value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="openai/gpt-realtime", help="OpenAI realtime model with image input.")
    parser.add_argument("--voice", default="marin")
    parser.add_argument("--transcription-model", default="gpt-4o-mini-transcribe")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    parser.add_argument("--input-device", type=device_argument, help="Microphone name or index; prefers the built-in Mac microphone.")
    parser.add_argument("--output-device", type=device_argument, help="Speaker/headphone name or index; defaults to system output.")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices without capturing or connecting.")
    parser.add_argument("--save-dir", help="Recording directory; defaults to LLM_TOOLKIT_RECORDING_DIR or recordings.")
    parser.add_argument("--frame-interval", type=float, default=1.0, help="Seconds between camera snapshots sent to the agent.")
    parser.add_argument("--timeout", type=float, default=30, help="Connection/setup timeout in seconds.")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="WARNING")
    args = parser.parse_args(argv)
    if not 0.25 <= args.frame_interval <= 30:
        parser.error("--frame-interval must be between 0.25 and 30 seconds")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    if args.camera < 0:
        parser.error("--camera must be non-negative")
    if "/" not in args.model:
        args.model = "openai/" + args.model
    if not args.model.startswith("openai/"):
        parser.error("This example requires an OpenAI realtime model with image input")
    return args


def main(arguments=None):
    args = arguments or parse_args()
    if args.list_devices:
        print(sd.query_devices())
        return 0
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\nConversation stopped.")
        return 0


if __name__ == "__main__":
    arguments = parse_args()
    configure_logging(level=arguments.log_level)
    raise SystemExit(main(arguments))
