"""llm-toolkit: api-agnostic helpers for building LLM apps.

Public API is re-exported here so consumers can write
``from llm_toolkit import ...`` without depending on the internal layout.
"""

from llm_toolkit.__about__ import __version__
from llm_toolkit.logging import configure_logging, get_logger

__all__ = [
    "__version__",
    "Agent",
    "AudioOutput",
    "RealtimeEnsemble",
    "ActionHandler",
    "CharacterSheet",
    "Director",
    "InputBuffer",
    "RealtimeSession",
    "configure_logging",
    "get_logger",
]


def __getattr__(name: str):
    if name == "AudioOutput":
        from llm_toolkit.media.audio_output import AudioOutput

        return AudioOutput
    if name == "RealtimeEnsemble":
        from llm_toolkit.agents.realtime_ensemble import RealtimeEnsemble

        return RealtimeEnsemble
    if name == "ActionHandler":
        from llm_toolkit.util.action_handler import ActionHandler

        return ActionHandler
    if name == "Agent":
        from llm_toolkit.agents.agent import Agent

        return Agent
    if name == "CharacterSheet":
        from llm_toolkit.agents.character import CharacterSheet

        return CharacterSheet
    if name == "Director":
        from llm_toolkit.agents.director import Director

        return Director
    if name == "InputBuffer":
        from llm_toolkit.util.input_buffer import InputBuffer

        return InputBuffer
    if name == "RealtimeSession":
        from llm_toolkit.agents.realtime import RealtimeSession

        return RealtimeSession
    raise AttributeError(f"module 'llm_toolkit' has no attribute {name!r}")
