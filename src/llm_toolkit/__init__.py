"""llm-toolkit: api-agnostic helpers for building LLM apps.

Public API is re-exported here so consumers can write
``from llm_toolkit import ...`` without depending on the internal layout.
"""

from llm_toolkit.__about__ import __version__

__all__ = ["__version__"]
