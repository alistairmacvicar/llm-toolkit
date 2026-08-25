# Examples

Install the project in editable mode before running an example:

```bash
python -m pip install -e .
```

Configure the environment variables required by the LiteLLM provider named in
`--model`. No example contains an API key. Most commands default to `gpt-5`, but
any LiteLLM model string can be supplied.

Every program can be launched directly and used interactively. Optional command-line
arguments make the same files useful for one-shot experiments and automation.

| Example | Demonstrates |
| --- | --- |
| `basic_chat.py` | Text messages, conversation history, and responses |
| `multimodal_chat.py` | Image, audio, video, combined inputs, transcription, and file helpers |
| `modes_and_prompts.py` | Runtime model, prompt, output, and transcription modes |
| `tool_calling.py` | Tool schema generation and automatic Python tool execution |
| `character_chat.py` | Character sheets loaded from Python or JSON |
| `director_routing.py` | Broadcasts, whispers, overhearing, and ordered conversations |
| `director_conversation.py` | Multi-character conversations coordinated by a director AI |
| `facilitated_conversation.py` | Public, targeted, and private cast conversation with tunable turns |
| `session_persistence.py` | Saving and restoring conversation sessions |
| `generated_media.py` | Image, speech, and asynchronous video generation |
| `input_buffer.py` | Staging, replacing, confirming, discarding, events, and auto-send |
| `local_capture.py` | InputBuffer and ActionHandler with camera and microphone sources |
| `realtime_chat.py` | Realtime websocket sessions through a LiteLLM proxy |

Run a text chat:

```bash
python examples/basic_chat.py --model openai/gpt-5
```

Launch the other interactive templates the same way without input-specific flags:

```bash
python examples/multimodal_chat.py
python examples/tool_calling.py
python examples/director_conversation.py
python examples/generated_media.py
python examples/session_persistence.py
```

Send local media. Unsupported audio and image inputs use the configured fallback
models automatically:

```bash
python examples/multimodal_chat.py --model openai/gpt-5 \
    --text "Describe the common theme." --image photo.jpg --audio note.wav
```

Load the included character sheet from JSON:

```bash
python examples/character_chat.py --character examples/character_sheet.json
```

Character sheet files use this format:

```json
{
  "name": "Mara Vale",
  "description": "A navigator aboard a remote research vessel.",
  "personality": "Observant and dryly funny.",
  "background": "Mara maps unstable routes through deep space.",
  "scenario": "A visitor has entered the navigation room.",
  "goals": ["Explain the current route."],
  "instructions": ["Stay in character.", "Use concise dialogue."]
}
```

```bash
python examples/director_conversation.py --turns 6
```

Run the complete facilitated conversation with the included cast:

```bash
.venv/bin/dotenv run -- .venv/bin/python examples/facilitated_conversation.py \
    --model openai/gpt-4.1-nano --turns 2
```

The facilitator and characters are separate model instances. The facilitator is
stored as `Director.director_agent` and only selects speakers and supplies private
direction. It is not present in `Director.agents` and cannot take a character turn.
Every cast member is its own `Agent` with the system prompt rendered from that
character's `CharacterSheet`.

Normal messages are public. Use `@Iris Chen: What would you change?` to publicly
address Iris while allowing the full cast to overhear. Use
`/whisper Iris Chen: Keep the north entrance in mind.` for a private one-to-one
turn. `/turns 4` changes the number of director-selected AI turns generated after
each public user turn.

The local capture controls are `t` for terminal text, `a` for push-to-talk,
`p` for camera preview, `s` for a snapshot, `r` for recording, Enter to confirm,
Escape to discard, and `q` to quit.

Realtime sessions require a running LiteLLM proxy with realtime routes configured:

```bash
export LITELLM_API_KEY=your-proxy-key
python examples/realtime_chat.py --model openai/gpt-realtime
```

Live provider smoke tests are opt-in so the normal test suite never spends API
credits. They load provider keys from `.env`:

```bash
RUN_LIVE_TESTS=1 .venv/bin/python -m unittest tests.test_live_providers -v
```
