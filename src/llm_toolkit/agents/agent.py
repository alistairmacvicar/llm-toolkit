from typing import final

from litellm import completion


@final
class Agent:
    def __init__(
        self,
        model="gpt-5",
        modalities=["text"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[],
        system_prompt="You are a friendly assistant.",
    ):
        self.model = model
        self.modalities = modalities
        self.audio = audio
        self.messages = messages
        self.system_prompt = system_prompt

    def _send_message(self, message, role="user"):
        # TODO: can I auto detect audio/text/image input or should I split this into 3? What if the user sends a text caption or question alongside an image? I'll need to support [{content: [{type:text, text:"hello"}, {type:image, url:"blah.png"}]}]

        self.messages.append({"role": role, "content": message})

    def send_audio(self, audio, caption=None):
        # TODO: Still need to figure out if I want to base64 encode before it gets here. I should probably make the user send it pre-encoded with the helpers in @file_source so that I can pre-encode it right after recording from the mic but idk.
        # TODO: make sure hte selected model supports audio, otherwise transcribe
        message = [
            {"type": "input_audio", "input_audio": {"data": audio, "format": "wav"}}
        ]

        if caption:
            message.append({"type": "text", "text": caption})

        self._send_message(message)

    def send_text(self, text):
        message = [{"type": "text", "text": text}]

        self._send_message(message)

    def send_image(self, image, caption=None):
        # TODO: make sure the selected model supports vision
        message = [{"type": "image_url", "image_url": {"url": image}}]

        if caption:
            message.append({"type": "text", "text": caption})

        self._send_message(message)

    def get_response(self):
        return
