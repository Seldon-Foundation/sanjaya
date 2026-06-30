"""Trace event normalization shared across API routes."""

from __future__ import annotations

from typing import Any

KIND_MAP: dict[str, str] = {
    "sanjaya.completion_start": "run_start",
    "sanjaya.completion_end": "run_end",
    "sanjaya.iteration_start": "iteration_start",
    "sanjaya.iteration_end": "iteration_end",
    "sanjaya.root_llm_call_start": "root_response_start",
    "sanjaya.root_llm_call_end": "root_response",
    "sanjaya.code_execution_start": "code_instruction",
    "sanjaya.code_execution_end": "code_execution",
    "sanjaya.tool_call_start": "tool_call_start",
    "sanjaya.tool_call_end": "tool_call",
    "sanjaya.sub_llm_call.regular_start": "sub_llm_start",
    "sanjaya.sub_llm_call.regular_end": "sub_llm",
    "sanjaya.sub_llm_call.vision_start": "vision_start",
    "sanjaya.sub_llm_call.vision_end": "vision",
    "sanjaya.rlm_subcall_start": "subcall_start",
    "sanjaya.rlm_subcall_end": "subcall",
    "sanjaya.video_inspection_start": "video_inspection_start",
    "sanjaya.video_inspection_end": "video_inspection",
    "sanjaya.frame_inspection_start": "frame_inspection_start",
    "sanjaya.frame_inspection_end": "frame_inspection",
    "sanjaya.audio_analysis_start": "audio_analysis_start",
    "sanjaya.audio_analysis_end": "audio_analysis",
    "sanjaya.schema_generation_start": "schema_generation_start",
    "sanjaya.schema_generation_end": "schema_generation",
}


def normalize_trace_event(raw: dict[str, Any]) -> tuple[str, float, dict[str, Any]]:
    """Map an internal tracer event to the frontend event shape."""
    internal_kind: str = raw.get("kind", "unknown") or "unknown"
    timestamp: float = raw.get("timestamp", 0.0) or 0.0
    payload = {k: v for k, v in raw.items() if k not in ("kind", "timestamp")}
    frontend_kind = KIND_MAP.get(internal_kind, internal_kind)
    return frontend_kind, timestamp, payload
