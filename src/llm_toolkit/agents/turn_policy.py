"""Shared routing constraints and decision schemas for text and realtime directors."""

EMOTIONS = ("neutral", "frustrated", "nervous", "triumphant", "bored", "anxious",
            "amused", "laughing", "startled", "happy", "amazed", "sad", "curious")


CONVERSATION_GROUNDING = (
    "Ground claims about what has been said in the recorded dialogue. Character "
    "background, scene descriptions, goals and director instructions are not "
    "evidence of an earlier conversation. Keep private exchanges out of public "
    "replies unless the participant explicitly authorizes disclosure. If a public "
    "recap has no public discussion to draw on, say so briefly; do not invent a "
    "substitute discussion to conceal a private exchange."
)


def turn_instructions(max_turns, completed, *, optional_replies):
    """State the same continue/stop contract for text and realtime decisions."""
    progress = f"Completed character replies to the current user message: {completed}. "
    if not optional_replies:
        return (progress + f"Generate {max_turns} total AI speaking turns, unless no eligible "
                "characters remain. Select an eligible speaker; stopping is not enabled.")
    policy = (progress + f"At most {max_turns} character replies are permitted; this is a ceiling, "
              "not a target. Follow the participant's requested scope and leave room for their next turn. ")
    if completed == 0:
        return policy + "An initial reply is required. Select an eligible speaker."
    return (policy + "Decide whether to STOP before selecting another speaker. If the participant "
            "asked for one answer or one person's contribution and that answer has been given, "
            "STOP. If the request is otherwise satisfied, STOP. A character being available or "
            "able to comment is not a reason to continue. Continue only to supply a still-missing "
            "requested answer or a distinct perspective the participant invited. "
            'To STOP, return {"speaker":null,"instruction":"","reason":"The request is satisfied.",'
            '"emotion":"neutral"}; adapt the reason to the actual conversation. '
            "Use JSON null, not the string \"null\". Otherwise select an eligible speaker.")


def eligible_children(hierarchy, node, characters, *, excluded=(), target=None):
    """Return immediate children with at least one eligible character descendant."""
    allowed = set(characters) - set(excluded)
    if target is not None:
        allowed.intersection_update({target})
    children = hierarchy.get(node, list(characters))
    eligible = []
    for child in children:
        pending = [child]
        seen = set()
        while pending:
            name = pending.pop()
            if name in seen:
                raise ValueError("Cycle in director hierarchy")
            seen.add(name)
            if name in allowed:
                eligible.append(child)
                break
            pending.extend(hierarchy.get(name, []))
    return eligible


def decision_schema(candidates, *, allow_stop=False):
    """JSON Schema used for both completion responses and realtime function calls."""
    return {
        "type": "object",
        "properties": {
            "speaker": {"type": ["string", "null"] if allow_stop else "string",
                        "enum": [*candidates, None] if allow_stop else list(candidates)},
            "instruction": {"type": "string"},
            "reason": {"type": "string"},
            "emotion": {"type": "string", "enum": list(EMOTIONS)},
        },
        "required": ["speaker", "instruction", "reason", "emotion"],
        "additionalProperties": False,
    }


def validate_decision(decision, candidates, *, allow_stop=False):
    if not isinstance(decision, dict):
        raise ValueError("Director response must be a JSON object")
    speaker = decision.get("speaker")
    if not (allow_stop and "speaker" in decision and speaker is None):
        if not isinstance(speaker, str) or speaker not in candidates:
            raise ValueError(f"Director selected unknown speaker: {speaker}")
    for field in ("instruction", "reason"):
        if not isinstance(decision.get(field, ""), str):
            raise ValueError(f"Director {field} must be text")
    if decision.get("emotion", "neutral") not in EMOTIONS:
        raise ValueError("Director selected an unknown emotion")
    return {"speaker": speaker, "instruction": decision.get("instruction", ""),
            "reason": decision.get("reason", ""), "emotion": decision.get("emotion", "neutral")}
