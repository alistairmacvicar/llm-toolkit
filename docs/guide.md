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

## Realtime and local capture

Use `RealtimeSession` as an async context manager to close the websocket:

```python
from llm_toolkit import RealtimeSession

async def chat():
    async with RealtimeSession(
        "ws://localhost:4000/v1/realtime", model="your-realtime-model"
    ) as session:
        await session.send_text("Hello")
        async for event in session.events():
            print(event)
            if event.get("type") == "response.done":
                break
```

`RealtimeEnsemble` adds cast routing and application callbacks for audio playback.
Always call `await ensemble.close()` in a `finally` block. Hosts choose recording
boundaries and must not automatically replay uncertain utterances after failure.
Concurrent `connect()` calls wait for the same peer initialization. A failed
final turn callback restores durable conversation state and disconnects the
peers. Audio already played or events already delivered cannot be undone.
`RealtimeSession.connected` becomes false after a remote WebSocket close;
call `connect()` explicitly to establish a new connection without replaying data.

`InputBuffer` stages inputs without hardware. `ActionHandler` starts a global
keyboard listener; create it only in a desktop application. Call `quit()` on
capture sources and action handlers to release their resources. `AudioOutput`
opens the selected output device only during playback.
PCM playback accepts samples split across chunks and writes at most 4,096 samples
at a time. A trailing incomplete sample raises an error after cleanup.

`CameraSource` raises if the device cannot open. Recording uses the dimensions of
the current captured frame and falls back to 30 fps when device metadata is
invalid. Closed sources reject snapshots, recording, and preview requests.
Shutdown attempts to close all devices even if another device reports an error.

`MicrophoneSource` monitoring uses one reusable output stream and a preallocated
mono PCM buffer. `monitor_device` selects the output device;
`monitor_buffer_seconds` defaults to 0.25 seconds. Overflow discards the oldest
audio and underflow emits silence; callbacks skip rather than wait for a busy
buffer. The source's `status` attribute retains the most recent input/output status flag.
Recording can continue while monitoring is toggled and vice versa. Recordings
remain in memory, so hosts should limit recording duration. Call `quit()` to close
both streams and stop the worker.
