"""LLM response types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class UsageSnapshot:
    """Token usage snapshot for a single model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0
    input_modality_tokens: dict[str, int] | None = None


@dataclass
class CallMetadata:
    """Metadata captured from a single model call."""

    requested_model: str | None = None
    model_used: str | None = None
    provider: str | None = None
    provider_response_id: str | None = None
    fallback_used: bool = False
    primary_error: str | None = None
    duration_seconds: float | None = None
    cost_usd: float | None = None
