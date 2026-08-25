# llm-toolkit

Python package to make building llm apps easier.

# Features

The package is currently focused on a small, local-first loop for multimodal LLM experiments:
capture text/audio/image/video inputs, normalize them into provider-friendly message payloads,
and keep enough session artefacts around to inspect or replay an experiment.

Logging is available through `llm_toolkit.configure_logging()` and the shared
`llm_toolkit` logger namespace.

Realtime inputs are staged by default. Press Enter to confirm the latest input of
each type, or Escape to discard them and record again. Pass `auto_send=True` to
`ActionHandler` to send each completed capture immediately. A `send_handler`
callback can receive confirmed inputs directly; applications can instead consume
the same `send`, `pending`, `discard`, and `quit` events with `get_event()`.

```python
from llm_toolkit import ActionHandler, Agent

agent = Agent(transcription_enabled=False)
actions = ActionHandler(auto_send=False, send_handler=agent.send_inputs)
```

`Agent.send_video()` samples a video into JPEG data URLs so vision-capable models
can receive it through the same conversation interface. Audio transcription can
be changed at runtime with `set_transcription()` or `toggle_transcription()`.

Tools can be registered with `add_tool(tool_definition, handler=callable)` or
generated from a typed function with `register_tool(callable)`. Registered tools
are executed automatically by `get_response()` with their results retained in the
conversation history. Tools without handlers remain available for manual execution.

`CharacterSheet` stores reusable character details as plain Python data and renders
them into a system prompt. `Director.add_character()` creates and registers an agent
from a sheet while accepting the same model options as `Agent`.

Agent sessions can be inspected with `get_session()`, updated with `set_session()`,
and persisted as JSON with `save_session()` and `load_session()`. Tool definitions
are saved, but executable handlers must be registered again after loading. A Director
can route a generated response with `speak()` or run a bounded sequence with
`converse()`.

Facilitated conversations use `Director.public_turn()` to let the full cast
overhear user and character messages while a director agent selects each speaker.
`Director.private_turn()` keeps a user exchange isolated to one character. The
public AI-to-user turn ratio can be updated with `set_ai_turns()`.
The director agent is separate from `Director.agents`; each cast member remains an
independent agent configured from its own character sheet.

Generated media uses the provider-specific LiteLLM endpoints behind the same Agent:
`generate_image()` optionally saves the first image, `generate_speech()` can stream
audio to a file, and video generation is exposed through `generate_video()`,
`get_video_status()`, and `save_video()`.

`RealtimeSession` connects to a LiteLLM `/v1/realtime` websocket and provides
helpers for session updates, text, PCM16 audio, images, sampled video frames, and
raw server events. `create_send_handler()` bridges confirmed `ActionHandler` inputs
onto the application's asyncio event loop.

Run the local conversation example with text-only input or enable available devices:

```bash
python test.py --model openai/gpt-5
python test.py --model openai/gpt-5 --camera --microphone
```

Focused, runnable programs for text and multimodal chat, tools, character sheets,
runtime mode switching, AI-directed ensemble conversations, session persistence,
generated media, local capture, and realtime sessions are collected in
[`examples/`](examples/README.md).

- Media sources
  - Video/Image
    - File loading and base64 encoding
    - Camera snapshots and recording
    - Link loading
  - Audio
    - File loading and base64 encoding
    - Microphone capture, optional local saving, and input monitoring
    - Link loading
    - Toggleable transcription
  - For all realtime inputs, options for auto-sending or confirm-sending to allow re-recording

- LLM Providers
  - Single wrapper supporting:
    - Gemini
    - Claude
    - ChatGPT
  - Video, image, audio, and text inputs/outputs
  - Conversation APIs through LiteLLM
  - Provider-native realtime websocket sessions through a LiteLLM proxy
  - Tracking token usage

- LLM Interaction
  - Conversation history storage
  - Director integration:
    - Character-character interaction
    - Character-user interaction
    - "Overhearing" functionality
    - "whispering" functionality
    - Scalable list of character inputs
  - Character sheet templates
  - Tool call support
  - System prompt/session update functions

- Keybindings
  - Push to talk audio
  - View camera preview
  - Monitor input audio
  - Type message into the terminal queue
  - Record audio
  - Record video
  - Record audio and video
  - Confirm message/event
  - Discard pending inputs
