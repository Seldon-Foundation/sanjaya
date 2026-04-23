"""Shared default model configuration for Sanjaya agents."""

from __future__ import annotations


DEFAULT_ROOT_MODEL = "google-vertex:gemini-3.1-pro-preview"
DEFAULT_SUB_MODEL = "google-vertex:gemini-3.1-pro-preview"
DEFAULT_VISION_MODEL = "google-vertex:gemini-3.1-pro-preview"
DEFAULT_AUDIO_MODEL = "google-vertex:gemini-3-flash-preview"
DEFAULT_CAPTION_MODEL = "moondream:moondream3-preview"
DEFAULT_FALLBACK_MODEL = None
DEFAULT_CRITIC_MODEL = "openrouter:qwen/qwen3-30b-a3b-thinking-2507"


def get_default_model_config() -> dict[str, str | None]:
    """Return the shared default model specs used by ``Agent``."""
    return {
        "root": DEFAULT_ROOT_MODEL,
        "sub": DEFAULT_SUB_MODEL,
        "vision": DEFAULT_VISION_MODEL,
        "audio": DEFAULT_AUDIO_MODEL,
        "caption": DEFAULT_CAPTION_MODEL,
        "fallback": DEFAULT_FALLBACK_MODEL,
        "critic": DEFAULT_CRITIC_MODEL,
    }
