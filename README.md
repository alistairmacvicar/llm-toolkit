# llm-toolkit

Python tools for multimodal LLM applications, local media capture, and conversations
between multiple agents. Provider requests use LiteLLM.

## Installation

Requires Python 3.11 or newer. From a checkout of this repository:

```bash
python -m pip install -e .
```

Configure the credentials for your provider through environment variables, such as
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`. See the
[getting-started guide](https://alistairmacvicar.github.io/llm-toolkit/getting-started/)
for setup and system dependencies.

## Quick start

```python
from llm_toolkit import Agent

agent = Agent(model="openai/gpt-4.1-mini", system_prompt="Answer briefly.")
agent.send_text("What is a Python generator?")
response = agent.get_response()
print(response.choices[0].message.content)
```

Use a model available to your provider account. Requests can incur provider charges.

## Features

- **Agents:** text, image, audio, and sampled video inputs; tool calls; generation
  options; token usage; and saved conversation sessions.
- **Generated media:** image generation, speech synthesis, transcription, and
  provider video-generation jobs.
- **Characters and directors:** character sheets, public and private conversations,
  speaker selection, director hierarchies, and conversation summaries.
- **Realtime:** WebSocket sessions, OpenAI WebRTC negotiation, and ensembles with
  streamed audio and application-controlled playback.
- **Local capture:** camera snapshots and recordings, microphone capture and
  monitoring, audio output, and keyboard controls.
- **Staged input:** confirm or discard captured text and media before sending them,
  or enable automatic sending.

The main imports are `Agent`, `CharacterSheet`, `Director`, `RealtimeSession`,
`RealtimeEnsemble`, `InputBuffer`, `ActionHandler`, and `AudioOutput` from
`llm_toolkit`. Capture sources and file helpers live in `llm_toolkit.media`.

## Documentation and examples

- [Usage guide](https://alistairmacvicar.github.io/llm-toolkit/guide/)
- [API reference](https://alistairmacvicar.github.io/llm-toolkit/reference/agent/)
- [Runnable examples](examples/README.md)
- [Development and GitHub Pages](docs/development.md)

```bash
python examples/basic_chat.py --model openai/gpt-4.1-mini
python examples/input_buffer.py
```

Camera, microphone, keyboard, and playback features require suitable local devices
and OS permissions. Text conversations do not require capture hardware.

## Development

```bash
python -m pip install -e '.[dev]'
LITELLM_LOCAL_MODEL_COST_MAP=True python -m pytest
```

Generate the documentation from source:

```bash
python -m pip install -r requirements-docs.txt
python -m mkdocs serve
python -m mkdocs build --strict
```

The static site is written to `site/`. GitHub Actions runs tests on pull requests
and deploys documentation from `main`.
