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
`max_frames` limits frames sent to the provider; sampling currently decodes the
whole video, so bound file size and duration in the hosting application.

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

## Characters and a director

```python
from llm_toolkit import Agent, CharacterSheet, Director

director = Director(director_agent=Agent(model="openai/gpt-4.1-mini"), ai_turns=2)
director.add_character(CharacterSheet(name="Ada", personality="Curious and precise."))
director.add_character(CharacterSheet(name="Lin", personality="Practical and direct."))
replies = director.public_turn("What should we build?", target="Ada")
private_reply = director.private_turn("Ada", "Consider my private concern.")
```

Public replies are broadcast to the cast. Private input and output are routed
only to the selected character, which retains its own private memory on later
turns. Grounding instructions discourage disclosure; they are not a guarantee
that a model will never mention something it remembers. Coordinators can form
a validated hierarchy; see [Director](reference/director.md).

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

`InputBuffer` stages inputs without hardware. `ActionHandler` starts a global
keyboard listener; create it only in a desktop application. Call `quit()` on
capture sources and action handlers to release their resources. `AudioOutput`
opens the selected output device only during playback.
