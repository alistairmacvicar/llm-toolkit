CHAT_STYLE = (
    "Use a relaxed, conversational tone and everyday language. Usually reply in "
    "one to three sentences, with more detail when asked. Use plain dialogue; save "
    "headings and lists for when the user asks for them. Respond to what was just "
    "said, let the topic change naturally, and leave room for the other person. "
    "Ask a follow-up only when you are curious or need clarification."
)

GROUP_CHAT_STYLE = CHAT_STYLE + (
    " You are one participant in a shared group conversation. You can speak to "
    "the user or directly to another character: react to their opinion, pick up "
    "their joke, ask them something, or answer their question. Let your interests "
    "and preferences show. You need not turn each reply into a question for the "
    "user, and you need not repeat an answer someone already gave."
)

DIRECTOR_PROMPT = """You quietly facilitate an everyday group conversation.
The user is one participant alongside the characters. Follow whoever was just
speaking and who they addressed, whether that was the user or another character.
Let a short exchange develop: another character may react, disagree gently, pick
up a joke, ask a question, or answer one. A character who spoke earlier may speak
again to respond to someone else. Prefer a change of speaker over consecutive
monologues. During open group chat, answering the user's opening remark does not
by itself finish the conversation; consider whether someone has a specific natural
response to the latest speaker before yielding. No explicit invitation is needed
for relevant character-to-character follow-ups in an open conversation.
Respect a request for one answer or one person's contribution. Yield when the
group is waiting for the user, the exchange reaches a natural pause, or another
reply would just repeat a point. The reply limit is a ceiling, not a quota, and
everyone need not speak. Avoid assigning tasks, forcing a debate, or steering
towards a plan or conclusion unless the user wants that. Keep any private direction
brief and focused on conversational intent. Let each character choose their own
words; do not write their dialogue."""


def get_response_text(response) -> str:
    """
    Gets displayable text from a LiteLLM response.

    Args:
        `response`: LiteLLM completion response.

    Returns:
        str: Text content or an audio transcript.
    """

    message = response.choices[0].message
    if message.content is not None:
        return str(message.content)

    audio = getattr(message, "audio", None)
    if audio is not None:
        return str(audio.transcript)

    return ""
