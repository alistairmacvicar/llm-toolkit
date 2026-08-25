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
