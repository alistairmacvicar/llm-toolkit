# llm-toolkit

Build Python applications that combine text, images, audio, video, and multiple
LLM characters. The toolkit wraps LiteLLM for provider requests and provides
local capture, conversation persistence, and realtime session helpers.

- **[Get started](getting-started.md)** with installation and a text conversation.
- **[Use the library](guide.md)** for media, tools, sessions, and directed conversations.
- **[Browse the API](reference/agent.md)** generated from the library's source and docstrings.
- **[Develop and publish](development.md)** with offline tests and GitHub Pages.

The main imports are `Agent`, `Director`, `CharacterSheet`, `RealtimeSession`,
`RealtimeEnsemble`, `InputBuffer`, `ActionHandler`, and `AudioOutput` from
`llm_toolkit`. Media source classes and file helpers live under
`llm_toolkit.media`.

Provider access is configured through environment variables. Camera, keyboard,
microphone, and playback support depend on local devices and OS permissions.
Text conversations do not need a capture device.

See the [repository README](https://github.com/alistairmacvicar/llm-toolkit#readme)
for the complete feature overview and the
[examples directory](https://github.com/alistairmacvicar/llm-toolkit/tree/main/examples)
for runnable programs.
