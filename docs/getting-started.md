# Getting started

## Install

Use Python 3.11 or newer. From a checkout of the repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

On Windows, activate with `.venv\Scripts\activate` instead. On Linux, local
audio capture needs PortAudio; Debian/Ubuntu users can install `libportaudio2`.
OpenCV may also need `libgl1` and `libglib2.0-0`. Keyboard capture and camera
previews need a desktop session; OS device permissions are granted separately.

Export the credential for the provider you intend to use, such as
`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`. The library does not
automatically load `.env` files. Keep credentials out of saved sessions and docs.

## Send a message

```python
from llm_toolkit import Agent

agent = Agent(model="openai/gpt-4.1-mini", system_prompt="Answer briefly.")
agent.send_text("What is a Python generator?")
response = agent.get_response()
print(response.choices[0].message.content)
print(agent.token_usage)
```

Use a LiteLLM model identifier available to your provider account. Model features
and generation parameters are checked at request time. Calling `get_response()`
can incur provider charges.

## Run examples

```bash
python examples/basic_chat.py --model openai/gpt-4.1-mini
python examples/input_buffer.py
```

The input-buffer example is local and does not call a model. The
[example catalogue](https://github.com/alistairmacvicar/llm-toolkit/blob/main/examples/README.md)
covers multimodal input, tools, character sheets, directed conversations,
persistence, media generation, and realtime sessions.
