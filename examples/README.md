# Examples

Install the project in editable mode before running an example:

```bash
python -m pip install -e .
```

Configure the environment variables required by the LiteLLM provider named in
`--model`, or put them in `.env`. The library automatically loads the nearest
`.env` in the working directory or its parents, preserving existing environment
variables. No example contains an API key. Most commands default to `gpt-5`, but
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
| `director_routing.py` | Manual message routing, including an agent-to-agent conversation |
| `director_conversation.py` | A casual group chat with AI speaker selection and natural pauses |
| `facilitated_conversation.py` | Public messages, addressed replies, and private asides with a JSON cast |
| `session_persistence.py` | Saving and restoring conversation sessions |
| `generated_media.py` | Image, speech, and asynchronous video generation |
| `input_buffer.py` | Staging, replacing, confirming, discarding, events, and auto-send |
| `local_capture.py` | Terminal capture with InputBuffer; optional global hotkeys via ActionHandler |
| `realtime_chat.py` | Model-based realtime connections through LiteLLM, with an optional proxy |
| `realtime_video.py` | Hands-free voice conversation in the terminal with a native camera preview and automatic recording |

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

Generate an image with a dedicated image model:

```bash
python examples/generated_media.py image \
    --media-model openai/gpt-image-2.5-flare \
    --prompt "A watercolor robot reading in a library"
```

Images default to `openai/gpt-image-2.5-flare` when `--media-model` is omitted
or you press Enter at the interactive image-model prompt. The example saves to
`recordings/generated-image.png` by default. `--model` configures the agent's
chat model; `--media-model` selects the generation model. A chat model such as
`gpt-5` cannot be passed directly to the image-generation endpoint. Supply a
compatible `--media-model` for speech or video.

OpenAI [retired Sora 2 and the Videos API on 24 September 2026](https://developers.openai.com/api/docs/deprecations).
Requests for `sora-2` or `openai/sora-2` can therefore return HTTP 404. Use an
available video provider with its own credentials. For example, the installed
LiteLLM adapter supports Google Cloud's
[Veo 3.1 production model](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/veo/3-1-generate):

```bash
python examples/generated_media.py video \
    --media-model vertex_ai/veo-3.1-generate-001 \
    --prompt "A suspenseful scene in the Australian outback at dusk"
```

This route needs Google Cloud credentials, a project with Veo access and billing,
and `VERTEXAI_PROJECT` and `VERTEXAI_LOCATION` (for example `us-central1`). See
[LiteLLM's Vertex AI setup](https://docs.litellm.ai/docs/providers/vertex_ai/videos).
An OpenAI or Gemini API key alone does not configure this route. Google also
[retired the Gemini API's Veo 3.1 preview models](https://ai.google.dev/gemini-api/docs/video);
use a currently available model rather than copying an older preview ID.

After a video job is created, enter `status` until it completes, then `save` to
download it. Creation failures return to the media chooser; status/download
failures let you retry the same job. A command-line generation failure exits
with a nonzero status.

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
python examples/director_conversation.py --turns 4
```

Run the complete facilitated conversation with the included cast:

```bash
python examples/facilitated_conversation.py \
    --model openai/gpt-5 --turns 4
```

The facilitator and characters are separate model instances. The facilitator is
stored as `Director.director_agent` and only selects speakers and supplies private
direction. It is not present in `Director.agents` and cannot take a character turn.
Every cast member is its own `Agent` with the system prompt rendered from that
character's `CharacterSheet`.

The default cast is Iris, Rowan, and Sol chatting over coffee. Their interests and
personalities shape their replies, and they have no workshop agenda or tasks to
complete. Character prompts ask for everyday language and usually one to three
sentences. Pass `--cast path/to/cast.json` to use your own characters and setting.

Every character receives each public user message and each public character reply,
including when someone else is addressed by name. They keep their own histories
and can speak to you or directly to each other. Only private whispers are limited
to their recipient.

Both AI-directed demos enable `director.optional_replies = True`: `--turns` is a
maximum, defaulting to four replies. They leave `director.avoid_repeat = False`,
so Iris can answer Rowan after he responds to her. The director can let a short
exchange develop without needing you to explicitly invite every contribution.
It can still stop after one reply for a focused question, or yield when the group
is waiting for you. Both demos retain public history between messages. Conversation
is turn-based: after the cap or an early stop, the program waits for your next input.

The manual `director_routing.py` example demonstrates lower-level routing. It has
no AI facilitator; Python fixes the speaking order. Plain text is sent to both
characters, then Iris replies and Rowan reacts to Iris. `/overhear MESSAGE`
delivers a message to both without asking for a response. `/converse MESSAGE`
starts an **agent-to-agent exchange**: only Iris receives the opening message,
her reply goes to Rowan, and his reply goes back to Iris. You watch the printed
exchange. Use `facilitated_conversation.py` for a group conversation that includes
you and lets an AI choose who should reply and when to yield.

Normal messages are public. Use `@Iris Chen: How would you spend a lazy Sunday?` to publicly
address Iris while allowing the full cast to overhear. Use
`/whisper Iris Chen: Help me think of a surprise for Rowan.` for a private one-to-one
turn. `/turns 3` allows up to three character replies after each public user turn;
it does not require everyone to answer. A private exchange goes only to its
recipient, who remembers it; the prompt asks them to keep it out of public replies.

## A five-minute walkthrough

Start with the basic chat, then move to the facilitated conversation above. The
text chat and director demos use warning-level logging to keep the terminal
focused on the conversation.

1. **A normal chat:** run `python examples/basic_chat.py` and say, "I've finally
   got a free Sunday and absolutely no plans." Continue with a follow-up to show
   conversation memory.
2. **A group conversation:** run `python examples/facilitated_conversation.py`
   and say, "I reckon going out for coffee defeats the point of a lazy Sunday."
   Watch for characters reacting to each other, then join back in. The director
   decides who replies and when to hand back to you.
3. **Address one person:** try `@Rowan Hale: What kind of film would you put on?`
   Rowan gets the first reply; everyone hears the exchange, but need not join in.
4. **Invite the group:** enter `/turns 3`, then "I'd like one idea from each of
   you: what's your ideal rainy afternoon?" This invites more participation.
5. **A private aside:** try `/whisper Iris Chen: Let's surprise Rowan with a movie
   night. Any ideas?` Use `/history` and `/private Iris Chen` to show where the
   exchanges were recorded.

Explain that the director handles the flow while each character supplies their
own dialogue. The model still chooses the wording and optional stopping, so use
these as conversation starters rather than a fixed transcript. Use `--model` to
select your configured model. Credentials in `.env` are loaded automatically.

## Switching text and voice modes

`modes_and_prompts.py` starts in relaxed assistant mode. Assistant, critic, and
tutor use the configured text model; `/mode voice` automatically selects the
configured voice model and requests both text and audio. Returning to a text mode
restores the text model. The conversation continues using transcripts of earlier
spoken replies.

```bash
python examples/modes_and_prompts.py \
    --model openai/gpt-5 --voice-model openai/gpt-audio-1.5
```

Try `Hello`, `/mode voice`, `What did I just say?`, then `/mode assistant` and a
follow-up. `/status` shows the active model and output type. `/model NAME` changes
the model for the current text or voice route; mode switches remember both choices.
Use `/clear` for a fresh conversation. Incompatible settings leave the previous mode
intact. Request errors are reported in the terminal and let you change settings
and re-enter the failed message without losing earlier conversation.

This example takes typed input. In voice mode, it prints the transcript, saves a
unique WAV file under `recordings/`, and plays it. Use `--no-playback` to save audio
without playing it, or `--output-dir PATH` to choose a destination. If playback is
unavailable, the WAV file remains available. `--mode voice` starts directly in voice
mode. Audio capture is demonstrated separately by `local_capture.py`.

Input and output support are separate capabilities. GPT-5 cannot produce audio;
GPT-Audio-1.5 accepts text and audio, but its requests need audio on at least one
side. Changing a style prompt alone cannot make an incompatible combination work.
The demo routes explicitly between the two configured models; `Agent.set_mode()`
also accepts a model, but does not choose an arbitrary provider automatically.
See [audio chat](https://developers.openai.com/api/docs/guides/audio-chat-completions).

## Capture and realtime

`local_capture.py` uses ordinary terminal commands by default, without global
keyboard monitoring or macOS Accessibility permission:

```bash
python examples/local_capture.py --microphone
```

Type a message and press Enter to stage it, then press Enter again on an empty
line (or type `/send`) to confirm and get an agent reply. `/audio` starts recording;
enter `/audio` again to stop and stage it. `/discard` clears staged inputs, `/pending`
shows their types, and `/quit` exits. `--auto-send` requests a reply as soon as an
input is captured, without separate confirmation. With `--camera`, use `/snapshot`
or `/preview`; `/record` starts/stops all enabled recording sources together.
`/monitor` toggles microphone monitoring. Use `/help` to display the controls.
Microphone and camera are off unless their flags are supplied; OS device permissions
still apply. The default agent returns text replies.

To capture speech and video together, run
`python examples/local_capture.py --camera --microphone`, enter `/record` to
start, then `/record` again to stop. `/pending` should list both `audio` and `video`;
use `/send` to request a reply. The saved MP4 ending in `-with-audio.mp4` includes
the microphone audio, aligned to the camera's start time. The original camera
file and separate microphone WAV are retained. Audio is saved automatically;
`--save` is kept for compatibility but is no longer needed. `--save-dir PATH`
changes the recording directory. The agent receives both inputs in one user
message. Models without audio input receive a transcript from
the configured transcription model, so spoken questions remain alongside the
video frames instead of being followed by a generic video-description request.

`/preview` keeps the camera window responsive while you type and while the agent
replies. OpenCV window operations run on the main thread, including on macOS.
When using `CameraSource` directly, call `poll_preview()` regularly on the main
thread. `--log-level INFO` (the default) shows available audio transcripts and
recording paths; it cannot show a transcript the selected provider did not return.

For system-wide shortcuts, add `--global-hotkeys`: `t` opens terminal text input,
hold `a` for push-to-talk, `m` toggles monitoring, `p` toggles camera preview, `s`
takes a snapshot, `r` toggles recording, Enter confirms, Escape discards, and `q`
quits. These shortcuts apply even when another application has focus.

If macOS reports that the process is not trusted, allow the hosting terminal or
editor in **System Settings > Privacy & Security > Accessibility**, then restart
that app. Input Monitoring may also need permission. See
[pynput's macOS requirements](https://pynput.readthedocs.io/en/stable/limitations.html#macos).
The example falls back to terminal controls when global keyboard permission is
missing; using the default controls avoids that permission requirement.

### Hands-free video and audio

Put `OPENAI_API_KEY` in your `.env`, then run:

```bash
python examples/realtime_video.py
```

The camera preview opens in a regular window. Speak normally, pause for a spoken
reply, or interrupt the agent. Transcripts appear in the terminal as `You:` and
`Agent:`. The microphone streams continuously and replies play as they arrive;
there is no push-to-talk or confirmation step. Type `/quit`, press Ctrl-C, or
press **Q/Esc in the preview window** to stop and save. Closing the preview also
ends the session. Camera and microphone permissions apply to the app running
Python.

Use headphones: native audio does not have the browser's echo cancellation.
On macOS the example prefers the **built-in microphone**, keeping the system's
selected output (such as AirPods). This avoids using the AirPods microphone by
default. It prints both chosen devices and does not change macOS sound settings.
List devices or choose them explicitly:

```bash
python examples/realtime_video.py --list-devices
python examples/realtime_video.py \
    --input-device "MacBook Air Microphone" --output-device "AirPods"
```

Device names or numeric indices are accepted. `--camera 1` selects a different
camera. Both audio streams are closed on exit, including setup failures.

The agent sees one JPEG camera snapshot per second alongside live audio. The
preview and local recording are full video; the model's visual input is sampled
frames. Only the latest 30 snapshots remain in its conversation. This example
uses `RealtimeSession` through LiteLLM's WebSocket bridge with an image-capable
OpenAI realtime model, defaulting to `openai/gpt-realtime`. Voice activity detection
starts replies after you pause. When you interrupt, queued speech is discarded
and the conversation is truncated to the audio already played.

Recordings stream to a unique `recordings/live_.../` folder during the session:

- `camera_...-with-audio.mp4`: camera video **with microphone audio**; the original
  silent camera file is retained.
- `microphone_....wav`: captured microphone audio, including queued audio at exit.
- `input-audio_....wav`: the exact PCM sent through `RealtimeSession`.
- `agent_....wav`: each received agent audio reply, including any unheard portion
  received before an interruption.
- `transcripts.jsonl`: available user and agent transcripts, also shown in the terminal.
- `session.json`: model and session start/finish times.

WAV chunks are flushed to disk during the session. On exit, the example combines
the microphone and video with their start times aligned. A recording, device,
storage, or connection failure ends the session and retains the captured files;
unfinished captures may be partial. Microphone and playback queues are bounded.

```bash
python examples/realtime_video.py --voice marin --frame-interval 1 \
    --save-dir recordings
```

`--model` selects the OpenAI realtime model; `--transcription-model` defaults to
`gpt-4o-mini-transcribe`. The terminal shows where it is saving. `--log-level INFO`
adds library diagnostics and available transcripts to the logs. Running the
example starts the preview and connection; microphone streaming starts once
session setup succeeds. `--list-devices` only lists devices and exits.

### Typed realtime chat

The realtime demo connects by `provider/model`, using LiteLLM's realtime adapters
inside the Python process. It does not require a separately running proxy:

```bash
python examples/realtime_chat.py --model openai/gpt-realtime
python examples/realtime_chat.py --model gemini/gemini-3.1-flash-live-preview --output audio
python examples/realtime_chat.py --model azure/YOUR_REALTIME_DEPLOYMENT
python examples/realtime_chat.py --model xai/grok-voice-latest --output audio
```

The library loads `.env` automatically without overwriting existing environment
variables. Set credentials for the selected provider: `OPENAI_API_KEY`,
`GEMINI_API_KEY`, `XAI_API_KEY`, or Azure's `AZURE_API_KEY` and `AZURE_API_BASE`.
Vertex AI and Bedrock use their usual cloud credentials and may require additional
provider SDK dependencies. Provider settings can be supplied as JSON, for example
`--provider-options '{"vertex_project":"my-project","vertex_location":"us-central1"}'`.
`--api-key-env NAME` explicitly chooses a different key variable.

Input is typed; text and transcripts stream to the terminal. `--output audio`
requests voice replies, which are saved as WAV files and played after each reply
finishes. `--no-playback` saves them without playing; `--output-dir PATH` changes
the destination. Some native-audio models return audio even when text is requested;
the demo shows the provider's negotiated output and handles returned audio too.
It does not capture the microphone. `--timeout` bounds connection/setup/reply waits,
and errors stop the demo with guidance instead of an unhandled traceback.

Use a model supported by the provider's **realtime** API. Chat models, transcription-only
models, and separate protocols such as OpenAI Live are not interchangeable with
this conversation API. See [LiteLLM realtime support](https://docs.litellm.ai/docs/realtime).

To use an existing OpenAI-compatible proxy instead, supply its URL explicitly:

```bash
python examples/realtime_chat.py --url ws://localhost:4000/v1/realtime \
    --model YOUR_PROXY_MODEL_ALIAS --api-key-env LITELLM_API_KEY
```

An explicit URL uses the running server's model aliases and credentials. It is
not automatically replaced with a direct connection if that server is unavailable.

Live provider smoke tests are opt-in so the normal test suite never spends API
credits. They load provider keys from `.env`:

```bash
RUN_LIVE_TESTS=1 .venv/bin/python -m unittest tests.test_live_providers -v
```
