from typing import Any, Literal, TypedDict

ActionType = Literal["pending", "send", "discard", "quit"]
InputType = Literal["text", "audio", "image", "video"]
MessageRole = Literal["system", "user", "assistant", "tool"]


class TextContent(TypedDict):
    type: Literal["text"]
    text: str


class ImageContent(TypedDict):
    type: Literal["image_url"]
    image_url: dict[str, str]


class AudioContent(TypedDict):
    type: Literal["input_audio"]
    input_audio: dict[str, str]


Content = TextContent | ImageContent | AudioContent | dict[str, Any]


class Message(TypedDict, total=False):
    role: MessageRole
    content: str | list[Content] | None
    tool_calls: list[dict[str, Any]]
    tool_call_id: str
    name: str


class TokenUsage(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class SessionData(TypedDict, total=False):
    model: str
    modalities: list[str]
    audio: dict[str, Any]
    messages: list[Message]
    transcription_model: str
    transcription_enabled: bool
    vision_fallback_model: str
    tools: list[dict[str, Any]]
    tool_choice: Any
    system_prompt: str
    token_usage: TokenUsage
    last_image_path: str | None
    last_speech_path: str | None
    last_video_path: str | None


class InputEvent(TypedDict):
    type: InputType
    value: Any


class ActionEvent(TypedDict):
    type: ActionType
    value: Any
