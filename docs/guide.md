# Usage guide

## Media input

```python
from llm_toolkit import Agent

agent = Agent(model="openai/gpt-4.1-mini", transcription_enabled=True)
agent.send_image("photo.jpg", caption="Describe this scene.")
agent.send_audio("recording.wav")
agent.send_video("clip.mp4", sample_fps=2, max_frames=20)
response = agent.get_response()
```

Images are sent as URLs or data URLs; videos become sampled JPEG frames. When
the selected model lacks a modality, the configured transcription or vision
fallback model is used. Those fallback requests can also incur charges.
`send_inputs()` combines each confirmed capture into one user message, keeping
audio or its transcript alongside any images and sampled video frames. When a
batch includes text or audio, its video does not add a separate description
request. `send_video()` by itself still requests a description by default.
`max_frames` bounds retained JPEG frames during loading, including for realtime
video. Samples are evenly spaced across the video, including the first and last sample.
`max_frames=1` reads only the first frame.
Memory depends on frame resolution and the limit; processing time still grows
with video duration. Inputs must be stable, reopenable files.

`load_video()` and `load_video_urls()` default to 20 frames. Pass
`max_frames=None` explicitly to retain all samples. `load_link()` defaults to a
64 MiB response limit and a 30-second timeout for each blocking socket operation;
override these with `max_bytes` and `timeout`. Use `save_link()` to stream to disk.
It preserves an existing destination if downloading fails. Generated-image URL
saves use the same streaming path; `generate_image()` accepts
`download_max_bytes` and `download_timeout`. These limits do not cover media
already returned in memory by a provider, local image/audio files, or recordings.
HTTP responses that end before their declared content length are rejected.

`Agent` accepts encoded audio files/bytes, base64 strings, and audio arrays.
Arrays use 24 kHz. `RealtimeSession` takes raw mono PCM16 bytes or base64 PCM16;
audio files must match the session sample rate. Realtime arrays are downmixed to
mono; integer arrays represent PCM16 sample values and floats use the range
`[-1, 1]`. `RealtimeEnsemble` requires mono PCM16 at 24 kHz.

Audio sent through the library is automatically saved under `recordings/`,
including supplied files, encoded bytes, arrays, raw audio messages, and explicit
`transcribe_audio()` calls. `Agent` saves the encoded input before sending or
transcribing it; `RealtimeSession` appends outgoing PCM16 chunks to a WAV file
for each connection. Closing the session finalizes that file. Replayed agent
history reuses its existing archive. A storage failure raises before the affected
audio is sent. Set `recording_dir=` on either class or the
`LLM_TOOLKIT_RECORDING_DIR` environment variable to choose the directory;
`last_input_audio_path` holds the latest archive path.

With `configure_logging(level="INFO")`, the library logs available user audio
transcripts and assistant audio transcripts. User transcripts are available when
transcription runs or a realtime provider emits a completed transcription event.
Sending audio directly to an audio-capable chat model does not itself produce a
user transcript. The library does not make extra transcription calls just for
logging.

## Modes and output capabilities

Change the model and its output settings together when switching to voice:

```python
chat_model = agent.model
agent.set_mode(
    model="openai/gpt-audio-1.5",
    modalities=["text", "audio"],
    audio={"voice": "alloy", "format": "wav"},
)
# Later, return to text while keeping the conversation:
agent.set_mode(model=chat_model, modalities=["text"])
```

`set_mode()` validates the combined configuration before applying it. Unsupported
audio output raises `ValueError` instead of silently removing audio; a failed
change preserves the existing configuration and conversation. `set_model()` keeps
the current output settings, so use `set_mode()` to change both at once.

Outgoing history uses transcripts of previous assistant audio replies when
available. Original audio remains in the stored session and returned response.
Raw audio inputs are not discarded or automatically transcribed when changing
models: if the new model cannot read them, the next request raises a local error.
Use transcription when initially sending audio if later text-only model switches
are needed. Audio replies without transcripts also require an audio-input model
with usable audio references, or a new conversation.

Input and output capabilities do not imply every combination is valid. Known
OpenAI audio-chat models require audio input or audio output, so a text-only
request is rejected locally with guidance. Other provider constraints are checked
by LiteLLM and the provider. See [OpenAI audio chat](https://developers.openai.com/api/docs/guides/audio-chat-completions).
Applications choose their own model routes; the modes example automatically switches
between its configured text and voice models and saves/plays voice responses.

## Image generation

Use a dedicated image model when generating images from a chat agent:

```python
agent.generate_image(
    "A watercolor robot reading in a library",
    model="openai/gpt-image-2.5-flare",
    file_path="recordings/generated-image.png",
)
```

`generate_image()` calls LiteLLM's image-generation endpoint. If `model` is
omitted, it uses `agent.model`; the default chat model `gpt-5` is not valid for
that endpoint. Pass an image model explicitly, even if the chat model can
understand image inputs. See OpenAI's [image model documentation](https://developers.openai.com/api/docs/models/gpt-image-2.5-flare).

## Tools

```python
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right

agent.register_tool(add)
agent.send_text("Add 2 and 3 using the tool.")
response = agent.get_response()
```

Use ordinary runtime type annotations and NumPy-style parameter docstrings for
LiteLLM's schema generator. For complex schemas, use `add_tool(definition,
handler=callable)` explicitly. Handlers run synchronously in your application.
Only register operations the application permits the model to invoke.

`get_response(execute_tools=False)` leaves execution to the caller. Before the
next completion, append one tool-role message with the matching `tool_call_id`
for every outstanding tool call. This also applies if `max_tool_rounds` is
reached and the returned assistant message still contains tool calls.
An unresolved batch raises a local `RuntimeError` before another provider request.
If any tool in a response has no registered handler, the whole batch is returned
for manual execution; already registered handlers are not run for that batch.

## Sessions and request context

```python
agent.save_session("session.json")
restored = Agent()
restored.load_session("session.json")
response = restored.get_response(context="Keep this answer to one sentence.")
```

Sessions retain messages, tool definitions, generation options, and usage.
Executable tool handlers must be registered again after restoration.
`get_session()` returns an independent copy. `get_messages()` copies only the
outer list; use the session snapshot if you need to modify nested messages safely.
Temporary `context` is removed after the request, including after failure.
Constructor inputs are copied recursively. Failed session updates preserve the
existing session and registered handlers.

## Characters and a director

```python
from llm_toolkit import Agent, CharacterSheet, Director

director = Director(director_agent=Agent(model="openai/gpt-4.1-mini"), ai_turns=2)
director.add_character(CharacterSheet(name="Ada", personality="Curious and precise."))
director.add_character(CharacterSheet(name="Lin", personality="Practical and direct."))
replies = director.public_turn("What should we build?", target="Ada")
private_reply = director.private_turn("Ada", "Consider my private concern.")
# Keep this exchange out of the character's public request history:
isolated_reply = director.private_turn("Ada", "A separate confidential topic.", isolate=True)
```

Public replies are broadcast to the cast. Private input and output are routed
only to the selected character, which retains its own private memory on later
turns. Grounding instructions discourage disclosure; they are not a guarantee
that a model will never mention something it remembers. Coordinators can form
a validated hierarchy; see [Director](reference/director.md).

For a casual group conversation, set `director.optional_replies = True` so the
director may yield before reaching `ai_turns`, and leave `director.avoid_repeat`
false so a character can answer back within the exchange. The stopping prompt
allows relevant character-to-character follow-ups in open group chat while still
respecting a request for one answer or one person's contribution. Give the director
and characters conversational system prompts; see the director examples. Each
`public_turn()` is bounded by its reply limit and then returns control to the host.

For text turns, use `isolate=True` consistently when private exchanges must stay
out of public requests. The separate history starts from the character's current
history on its first isolated turn and then evolves independently. It remains
in memory on the Director; it is not part of the character's exported Agent
session. This does not retroactively remove earlier shared memories or change
realtime routing. Avoid sharing agents between Directors or concurrent calls.

Failed `coordinate()`, `public_turn()`, `converse()`, and `private_turn()` requests
restore character/coordinator message histories and turn metadata. Usage and
decision diagnostics remain available. External tool effects and provider charges
cannot be rolled back; applications must account for them before retrying.

## Attributed dialogue memory

Use `Agent.compress_dialogue()` when a memory must retain the exact wording and
speaker of selected dialogue. Unlike `Agent.summarize()`, it asks the model for
source IDs only and renders the selected records locally:

```python
records = [{"id": "public:0", "speaker": "Ada", "audience": "public",
            "content": "We might inspect the gate; nobody has volunteered yet."}]
memory = agent.compress_dialogue(records, max_records=12,
                                 max_characters=6000, keep_recent=4)
print(memory["text"])
```

Records require unique IDs and exactly the four fields shown. Filter inputs by
audience before calling: the toolkit does not infer access permissions. The last
`keep_recent` records in each audience stream are retained automatically; the
model selects additional older records within the same total limits. Unknown or
duplicate IDs, extra output fields and size violations raise `ValueError`.
Mandatory recent records that exceed either budget fail before a model call.
Limits count complete records and rendered Unicode characters, not tokens.

The method returns evidence and metadata without changing the owner's history.
It inherits the owner's model/options and accounts for selection usage. Callers
can validate all characters' results before replacing any histories, and should
retain original transcripts for future transitions rather than summarizing old
summaries. Exact retained quotations do not guarantee completeness, truthful
sources, correct interpretation or confidential later replies. Use full history
when selection loss is unacceptable; `summarize()` remains a free-form alternative.

## Realtime and local capture

Select a realtime model with the same `provider/model` convention as `Agent`.
Use `RealtimeSession` as an async context manager to close the connection:

```python
from llm_toolkit import RealtimeSession

async def chat():
    async with RealtimeSession(model="openai/gpt-realtime") as session:
        await session.update_session(
            instructions="Be conversational and brief.", output_modalities=["text"],
        )
        await session.send_text("Hello")
        async for event in session.events():
            print(event)
            if event.get("type") == "response.done":
                break
```

Without a URL, the toolkit runs LiteLLM's realtime provider bridge in process.
LiteLLM resolves credentials and translates provider events; there is no listening
port or separate proxy process. Supported conversation routes include OpenAI,
Azure OpenAI, xAI, Gemini, Vertex AI, and Bedrock, subject to the installed LiteLLM
adapter, model support, credentials, and optional provider dependencies.
For example, `RealtimeSession(model="azure/my-deployment")` reads Azure's normal
environment settings. Pass `provider_options={"api_base": "https://..."}` or cloud
project/region settings when needed. `api_key` is an explicit credential override.
On first import, the library loads the nearest `.env` in the current working
directory or its parents, preserving existing environment variables.

An explicit `url="ws://localhost:4000/v1/realtime"` retains the existing
OpenAI-compatible WebSocket transport, including proxy model aliases, headers,
and injected `websocket_factory` support. No provider fallback or automatic replay
occurs on failure. `connect_timeout` bounds startup and cancels partial connections.
In-process queues are bounded, and closing wakes blocked readers and writers.

Configure the session before sending input. Google Live accepts one initial
configuration; later changes require a new session. The toolkit defers Google's
setup so the initial instructions and output settings are preserved, without
changing LiteLLM's global settings. The current LiteLLM Google adapter does not
translate image inputs; these raise a local error instead of being silently lost.
Other modality, voice, tool, and event support is provider/model-specific. Separate
protocols such as OpenAI Live and transcription-only models are not conversation
model substitutes. See [LiteLLM realtime support](https://docs.litellm.ai/docs/realtime).
The bridge uses LiteLLM's internal realtime adapter API; its compatibility is covered
by provider-routing and event-translation tests, with LiteLLM 1.106.4 as the minimum.

`RealtimeEnsemble` adds cast routing and application callbacks for audio playback.
Its default peers use the same model-based routing and accept `provider_options`;
custom `session_factory` remains available. Ensemble models must support its text
coordinator decisions, tools, audio replies, transcripts, and chosen voice settings.
Always call `await ensemble.close()` in a `finally` block. Its default peer sessions
archive outgoing audio automatically; hosts must not automatically replay
uncertain utterances after failure.
Concurrent `connect()` calls wait for the same peer initialization. A failed
final turn callback restores durable conversation state and disconnects the
peers. Audio already played or events already delivered cannot be undone.
`RealtimeSession.connected` becomes false after a remote WebSocket close;
call `connect()` explicitly to establish a new connection without replaying data.

`InputBuffer` stages inputs without hardware. `ActionHandler` starts a global
keyboard listener; create it only in a desktop application. It raises
`PermissionError` if the listener reports that macOS has not granted access.
For terminal applications, drive `InputBuffer` directly with ordinary `input()`
commands, as the local capture example does by default. That path does not require
global keyboard monitoring; `--global-hotkeys` enables the optional listener.
Call `quit()` on
capture sources and action handlers to release their resources. `AudioOutput`
opens the selected output device only during playback.
PCM playback accepts samples split across chunks and writes at most 4,096 samples
at a time. A trailing incomplete sample raises an error after cleanup.

`CameraSource` raises if the device cannot open. Recording uses the dimensions of
the current captured frame and falls back to 30 fps when device metadata is
invalid. Frames are paced against elapsed time to keep saved video aligned with
microphone audio. `toggle_capture()` requests a preview; call
`camera.poll_preview()` regularly **on the main thread** to display it and process
window events. Capture workers and hotkey callbacks never call OpenCV's GUI
functions. `ActionHandler.get_event()` pumps the preview when called on the main
thread; `examples/local_capture.py` also keeps it responsive during terminal input
and provider requests. Custom applications must keep pumping while waiting for
other work. Closed sources reject snapshots, recording, and preview requests.
Shutdown attempts to close all devices even if another device reports an error.

`toggle_recording_sources([("audio", microphone), ("video", camera)])` starts or
stops both sources and returns captured inputs. When both stop together, it creates
an MP4 ending in `-with-audio.mp4` with the microphone track embedded and start
times aligned. It preserves the original WAV and silent camera recording. The
local capture example and `ActionHandler` use this helper for joint recording,
including shutdown. Combining files uses the bundled `imageio-ffmpeg` executable.

`MicrophoneSource` monitoring uses one reusable output stream and a preallocated
mono PCM buffer. `monitor_device` selects the output device;
`monitor_buffer_seconds` defaults to 0.25 seconds. Overflow discards the oldest
audio and underflow emits silence; callbacks skip rather than wait for a busy
buffer. The source's `status` attribute retains the most recent input/output status flag.
Recording can continue while monitoring is toggled and vice versa.
`MicrophoneSource` saves WAV files by default, including an active recording at
shutdown. Chunks remain in memory until recording stops, so hosts should limit
recording duration. `save=False` returns arrays instead; sending those arrays
through an agent still archives them. Call `quit()` to close both streams and stop
the worker.

For hands-free camera and microphone conversation, run
`python examples/realtime_video.py`. The example opens the native camera preview
and prints transcripts in the terminal. It streams microphone audio and plays
speech as it arrives over `RealtimeSession`, with server voice activity detection
and local playback interruption. On interruption it reports the amount actually
played so the server can truncate the unheard audio. Type `/quit`, press Ctrl-C,
or press Q/Esc in the preview to release devices and finish recordings.

The agent receives a JPEG snapshot every second alongside continuous audio;
the local preview and recording are full video. Camera recordings include the
microphone track, with separate microphone, sent-input, and agent WAV files and
available transcripts. Use headphones because native audio has no browser echo
cancellation. On macOS the example prefers the built-in microphone without
changing system settings; `--input-device` and `--output-device` override the
selection. See the [example instructions](https://github.com/alistairmacvicar/llm-toolkit/blob/main/examples/README.md#hands-free-video-and-audio)
for setup and options.
