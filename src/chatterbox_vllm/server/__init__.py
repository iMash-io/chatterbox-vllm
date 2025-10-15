"""Server utilities for exposing Chatterbox as OpenAI compatible endpoints."""

from .openai_tts import create_openai_tts_app

__all__ = ["create_openai_tts_app"]
