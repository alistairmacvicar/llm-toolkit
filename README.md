# llm-toolkit

Python package to make building llm apps easier.

# TODO

- Media sources
  - Video/Image
    - File
    - Camera
    - Link
  - Audio
    - File
    - Microphone
    - Link
    - Toggleable transcription
  - For all realtime inputs, options for auto-sending or confirm-sending to allow re-recording

- LLM Providers
  - Single wrapper that supports:
    - Gemini
    - Claude
    - ChatGPT
  - The single wrapper will support video, image, audio, and text inputs/outputs.
  - Focus will be on "conversation" apis, realtime will be assessed afterwards.
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
  - Push to talk
  - View camera preview
  - Monitor input audio
  - Type message
  - Record audio
  - Record video
  - Record audio and video
  - Confirm message
