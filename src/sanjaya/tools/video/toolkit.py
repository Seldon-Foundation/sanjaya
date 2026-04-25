"""VideoToolkit — native slice-based video/audio analysis for Gemini."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ...answer import Evidence
from ..base import Tool, Toolkit, ToolParam
from .media import (
    extract_frame as _extract_frame_impl,
)
from .media import (
    get_video_info,
    video_duration_seconds,
)
from .mount import WorkspaceMount
from .workspace import ArtifactWorkspace

_DEFAULT_MAX_SPAN_S = 120.0

_VIDEO_STRATEGY_PROMPT = """\
## Video Strategy

Your goal is to avoid context rot. Do not ingest more video context than needed.
Every media-bearing call should target a small, relevant slice of the video.
Prefer multiple short calls over one large call.
If a span is too broad, split it before delegating.

### Native media tools

- `get_video_info()` — inspect source metadata such as duration, resolution,
  codec, fps, and file size before you plan your slices.

- `inspect_video(start_s, end_s)` — attach one explicit video slice to the
  current RLM layer's next root-model turn. This is promptless: you are not
  sending a separate inspection prompt. You are pulling that slice into your
  own current reasoning context so your next root response can directly use it.
  Call it in one iteration, then reason over the attached media in the next root turn.
  If `start_s == end_s`, this becomes a single-frame attachment at that second.

- `analyze_audio(start_s, end_s, prompt=None)` — transcribe and analyze the audio
  from one explicit, non-zero slice. Returns structured data with transcript,
  summary, and salient audio events.

### Direct delegation

- `llm_query(prompt, start_s=..., end_s=...)` sends exactly that slice to the
  sub-LLM. Omit timestamps for text-only reasoning.
- `rlm_query(prompt, start_s=..., end_s=...)` gives a child RLM an explicit
  slice assignment. Keep those slices small.
- Batched `llm_query_batched(...)` and `rlm_query_batched(...)` support dict
  items with `prompt`, `start_s`, and `end_s`.

### Rules

- Never attach the whole video unless the entire video is shorter than the span cap.
- Keep timestamps absolute to the original source video.
- Use `get_state()` to inspect video duration, active span, and recent media ops.
- Fan out across short, independent slices and synthesize after observing results.
"""


class _NullTrace:
    def record(self, **kwargs: Any) -> None:
        return

    def record_response(self, response: str) -> None:
        return


class AudioAnalysisResult(BaseModel):
    """Validated structured result returned by `analyze_audio()`."""

    transcript: str = Field(
        default="",
        description="Verbatim or near-verbatim spoken words when audible.",
    )
    audio_summary: str = Field(
        default="",
        description="Concise description of what is happening in the audio slice.",
    )
    salient_audio_events: list[str] = Field(
        default_factory=list,
        description="Short list of notable sounds, music, speaker changes, or non-speech events.",
    )


def _model_label(model: Any) -> str:
    if isinstance(model, str):
        return model
    return getattr(model, "model_name", type(model).__name__)


def _merge_ranges(ranges: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged: list[tuple[float, float]] = [ordered[0]]
    for start_s, end_s in ordered[1:]:
        last_start, last_end = merged[-1]
        if start_s <= last_end:
            merged[-1] = (last_start, max(last_end, end_s))
        else:
            merged.append((start_s, end_s))
    return merged


class VideoToolkit(Toolkit):
    """Native slice-based video toolkit with minimal, explicit tools."""

    def __init__(
        self,
        workspace_dir: str = "./sanjaya_artifacts",
        max_span_s: float = _DEFAULT_MAX_SPAN_S,
        trace_depth: int = 0,
    ):
        self.workspace_dir = workspace_dir
        self.max_span_s = max_span_s
        self._trace_depth = trace_depth

        self._video_path: str | None = None
        self._question: str | None = None
        self._duration_s: float | None = None
        self._workspace: ArtifactWorkspace | None = None
        self._mount: WorkspaceMount | None = None

        self._llm_client: Any = None
        self._inspect_llm_client: Any = None
        self._audio_llm_client: Any = None
        self._tracer: Any = None
        self._budget: Any = None

        self._modality: str = "balanced"
        self._active_span: tuple[float, float] | None = None

        self._video_slice_cache: dict[str, str] = {}
        self._audio_slice_cache: dict[str, str] = {}
        self._frame_cache: dict[str, str] = {}

        self._inspections: list[dict[str, Any]] = []
        self._audio_analyses: list[dict[str, Any]] = []
        self._single_frame_inspections: list[dict[str, Any]] = []
        self._pending_root_media: list[dict[str, Any]] = []

    def setup(self, context: dict[str, Any]) -> None:
        video = context.get("video")
        if not video:
            return

        self._video_path = str(Path(video).resolve())
        self._question = context.get("question")
        self._modality = context.get("modality", "balanced")
        self._duration_s = video_duration_seconds(self._video_path)

        run_context = context.get("context")
        run_id = run_context.get("run_id") if isinstance(run_context, dict) else None
        active_span = run_context.get("active_video_span") if isinstance(run_context, dict) else None
        self._active_span = self._parse_active_span(active_span)

        self._workspace = ArtifactWorkspace(base_dir=self.workspace_dir, run_id=run_id)
        self._mount = WorkspaceMount(str(self._workspace.run_dir))

    def spawn_child(
        self,
        *,
        active_span: tuple[float, float] | None = None,
        trace_depth: int | None = None,
    ) -> "VideoToolkit":
        child = VideoToolkit(
            workspace_dir=self.workspace_dir,
            max_span_s=self.max_span_s,
            trace_depth=self._trace_depth if trace_depth is None else trace_depth,
        )
        child._video_path = self._video_path
        child._question = self._question
        child._duration_s = self._duration_s
        child._workspace = self._workspace
        child._mount = self._mount
        child._llm_client = self._llm_client
        child._inspect_llm_client = self._inspect_llm_client
        child._audio_llm_client = self._audio_llm_client
        child._tracer = self._tracer
        child._budget = self._budget
        child._prompt_config = self._prompt_config
        child._modality = self._modality
        child._active_span = active_span if active_span is not None else self._active_span
        child._video_slice_cache = self._video_slice_cache
        child._audio_slice_cache = self._audio_slice_cache
        child._frame_cache = self._frame_cache
        child._inspections = self._inspections
        child._audio_analyses = self._audio_analyses
        child._single_frame_inspections = self._single_frame_inspections
        child._pending_root_media = []
        return child

    def teardown(self) -> None:
        return

    def tools(self) -> list[Tool]:
        return [
            self._make_get_video_info_tool(),
            self._make_inspect_video_tool(),
            self._make_analyze_audio_tool(),
        ]

    def get_state(self) -> dict[str, Any]:
        coverage_ranges = _merge_ranges([
            (entry["start_s"], entry["end_s"])
            for entry in self._inspections + self._audio_analyses
            if entry["end_s"] > entry["start_s"]
        ])
        total_coverage_s = round(sum(end_s - start_s for start_s, end_s in coverage_ranges), 3)

        return {
            "video_path": self._video_path,
            "video_duration_s": self._duration_s,
            "active_video_span": {
                "start_s": self._active_span[0],
                "end_s": self._active_span[1],
            } if self._active_span else None,
            "uploaded_file_status": {
                "status": "ready" if self._video_path else "missing",
                "mode": "native_attachment_with_slice_metadata",
                "source_video_path": self._video_path,
                "cached_video_slices": len(self._video_slice_cache),
                "cached_audio_slices": len(self._audio_slice_cache),
                "cached_single_frames": len(self._frame_cache),
            },
            "recent_inspected_spans": self._inspections[-8:],
            "recent_audio_spans": self._audio_analyses[-8:],
            "single_frame_inspections": self._single_frame_inspections[-8:],
            "pending_root_inspections": [
                {
                    "kind": entry["kind"],
                    "start_s": entry["start_s"],
                    "end_s": entry["end_s"],
                    "source": entry["source"],
                }
                for entry in self._pending_root_media[-8:]
            ],
            "total_coverage_s": total_coverage_s,
            "run_id": self._workspace.run_id if self._workspace else None,
        }

    def build_evidence(self) -> list[Evidence]:
        evidence: list[Evidence] = []
        seen: set[str] = set()

        for entry in self._inspections:
            if entry["kind"] == "frame":
                source = f"frame:{entry['start_s']:.1f}s"
            else:
                source = f"video:{entry['start_s']:.1f}s-{entry['end_s']:.1f}s"
            if source in seen:
                continue
            seen.add(source)
            evidence.append(
                Evidence(
                    source=source,
                    rationale=f"{entry['source']} inspected this span",
                    artifacts={
                        "artifact_path": entry.get("artifact_path"),
                        "prompt": entry.get("prompt"),
                    },
                )
            )

        for entry in self._audio_analyses:
            source = f"audio:{entry['start_s']:.1f}s-{entry['end_s']:.1f}s"
            if source in seen:
                continue
            seen.add(source)
            evidence.append(
                Evidence(
                    source=source,
                    rationale=f"{entry['source']} analyzed this audio span",
                    artifacts={
                        "artifact_path": entry.get("artifact_path"),
                        "transcript": entry.get("transcript"),
                    },
                )
            )

        return evidence

    def prompt_section(self) -> str | None:
        if not self._video_path:
            return None

        base = self._prompt_config.video_strategy if self._prompt_config and self._prompt_config.video_strategy else _VIDEO_STRATEGY_PROMPT
        parts = [base]

        if self._duration_s is not None:
            parts.append(f"\nVideo duration: {self._duration_s:.1f}s. Hard per-call span cap: {self._max_allowed_span_s():.1f}s.")

        if self._active_span is not None:
            start_s, end_s = self._active_span
            parts.append(
                f"\nActive child span: {start_s:.1f}s to {end_s:.1f}s. Stay within this range unless explicitly reassigned."
            )

        if self._modality == "transcript_primary":
            parts.append("\nThis question is audio/transcript-first. Start with `analyze_audio()` on relevant short spans.")
        elif self._modality == "vision_primary":
            parts.append("\nThis question is vision-first. Start with `inspect_video()` on relevant short spans.")
        else:
            parts.append("\nThis question is balanced. Use both `inspect_video()` and `analyze_audio()` where needed.")

        return "\n".join(parts)

    def get_os_access(self) -> Any | None:
        if self._mount:
            return self._mount.build_os_access()
        return None

    def _make_get_video_info_tool(self) -> Tool:
        toolkit = self

        def _get_video_info() -> dict[str, Any]:
            if toolkit._video_path is None:
                raise ValueError("No video loaded")
            return get_video_info(toolkit._video_path)

        return Tool(
            name="get_video_info",
            description=(
                "Get source video metadata such as duration, resolution, codec, fps, container, and file size."
            ),
            fn=_get_video_info,
            parameters={},
            return_type="dict[str, Any]",
        )

    def prepare_media_request(
        self,
        *,
        start_s: float,
        end_s: float,
        media_kind: str,
    ) -> dict[str, Any]:
        normalized_start, normalized_end = self._validate_span(
            start_s=start_s,
            end_s=end_s,
            allow_zero=(media_kind == "video"),
        )

        if media_kind == "video" and normalized_start == normalized_end:
            artifact_path = self._ensure_frame(normalized_start)
            return {
                "kind": "frame",
                "start_s": normalized_start,
                "end_s": normalized_end,
                "artifact_path": artifact_path,
                "media": [{"path": artifact_path, "media_type": "image/jpeg"}],
            }

        if media_kind == "audio":
            artifact_path = self._video_path
            self._audio_slice_cache[self._span_id(normalized_start, normalized_end)] = artifact_path
            return {
                "kind": "audio",
                "start_s": normalized_start,
                "end_s": normalized_end,
                "artifact_path": artifact_path,
                "media": [self._make_video_media_item(normalized_start, normalized_end)],
            }

        artifact_path = self._video_path
        self._video_slice_cache[self._span_id(normalized_start, normalized_end)] = artifact_path
        return {
            "kind": "video",
            "start_s": normalized_start,
            "end_s": normalized_end,
            "artifact_path": artifact_path,
            "media": [self._make_video_media_item(normalized_start, normalized_end)],
        }

    def record_inspection(
        self,
        *,
        start_s: float,
        end_s: float,
        prompt: str,
        response: str,
        artifact_path: str,
        kind: str,
        source: str,
        model: str,
    ) -> dict[str, Any]:
        entry = {
            "kind": kind,
            "start_s": start_s,
            "end_s": end_s,
            "prompt": prompt,
            "response_preview": response[:300] if response else None,
            "artifact_path": artifact_path,
            "source": source,
            "model": model,
        }
        self._inspections.append(entry)
        if kind == "frame":
            self._single_frame_inspections.append(entry)
        if self._workspace:
            self._workspace.record_media_operation(entry)
        return entry

    def record_audio_analysis(
        self,
        *,
        start_s: float,
        end_s: float,
        prompt: str,
        artifact_path: str,
        result: dict[str, Any],
        source: str,
        model: str,
    ) -> dict[str, Any]:
        entry = {
            "kind": "audio",
            "start_s": start_s,
            "end_s": end_s,
            "prompt": prompt,
            "artifact_path": artifact_path,
            "source": source,
            "model": model,
            "transcript": result.get("transcript"),
            "audio_summary": result.get("audio_summary"),
            "salient_audio_events": result.get("salient_audio_events"),
        }
        self._audio_analyses.append(entry)
        if self._workspace:
            self._workspace.record_media_operation(entry)
        return entry

    def queue_root_inspection(
        self,
        *,
        start_s: float,
        end_s: float,
    ) -> dict[str, Any]:
        request = self.prepare_media_request(
            start_s=start_s,
            end_s=end_s,
            media_kind="video",
        )
        model_label = _model_label(getattr(self._inspect_llm_client, "vision_model", None) or "root_multimodal")
        trace_cm = self._media_trace_context(
            kind=request["kind"],
            model=model_label,
            prompt="",
            start_s=request["start_s"],
            end_s=request["end_s"],
            source="inspect_video",
        )

        with trace_cm as trace:
            trace.record(
                attachment_mode="root_context",
                attachment_status="queued",
                media_kind=request["kind"],
                artifact_path=request["artifact_path"],
                response_preview="Attached to the current root context for the next turn.",
            )

        self.record_inspection(
            start_s=request["start_s"],
            end_s=request["end_s"],
            prompt="",
            response="",
            artifact_path=request["artifact_path"],
            kind=request["kind"],
            source="inspect_video",
            model=model_label,
        )

        queued = {
            "kind": request["kind"],
            "start_s": request["start_s"],
            "end_s": request["end_s"],
            "source": "inspect_video",
            "media": request["media"],
        }
        self._pending_root_media.append(queued)
        return queued

    def drain_pending_root_media(self) -> list[dict[str, Any]]:
        pending = list(self._pending_root_media)
        self._pending_root_media.clear()
        return pending

    def _make_inspect_video_tool(self) -> Tool:
        toolkit = self

        def _inspect_video(start_s: float, end_s: float) -> str:
            queued = toolkit.queue_root_inspection(start_s=start_s, end_s=end_s)
            if queued["kind"] == "frame":
                return (
                    f"Queued frame attachment at {queued['start_s']:.1f}s for the next "
                    "root-model turn."
                )
            return (
                f"Queued video attachment [{queued['start_s']:.1f}s - {queued['end_s']:.1f}s] "
                "for the next root-model turn."
            )

        return Tool(
            name="inspect_video",
            description=(
                "Attach one explicit slice of the video to the current root-model context. "
                "This is promptless: the next root turn will directly see the slice. "
                "If start_s == end_s, this attaches a single frame."
            ),
            fn=_inspect_video,
            parameters={
                "start_s": ToolParam(name="start_s", type_hint="float", description="Absolute start time in seconds."),
                "end_s": ToolParam(name="end_s", type_hint="float", description="Absolute end time in seconds."),
            },
            return_type="str",
        )

    def _make_analyze_audio_tool(self) -> Tool:
        toolkit = self

        def _analyze_audio(start_s: float, end_s: float, prompt: str | None = None) -> dict[str, Any]:
            audio_client = toolkit._audio_llm_client or toolkit._llm_client
            if audio_client is None:
                raise RuntimeError("Audio LLM client is not configured.")

            request = toolkit.prepare_media_request(
                start_s=start_s,
                end_s=end_s,
                media_kind="audio",
            )
            full_prompt = (
                "Analyze this audio slice. "
                "Use transcript for verbatim spoken words when audible, "
                "audio_summary for a concise description, and salient_audio_events "
                "for a short list of sounds, music, speaker changes, or notable non-speech events."
            )
            if prompt:
                full_prompt += f"\n\nAdditional task:\n{prompt}"

            model_label = _model_label(audio_client.model)
            trace_cm = toolkit._media_trace_context(
                kind="audio",
                model=model_label,
                prompt=full_prompt,
                start_s=request["start_s"],
                end_s=request["end_s"],
                source="analyze_audio",
            )

            with trace_cm as trace:
                parsed = audio_client.media_completion_structured(
                    prompt=full_prompt,
                    media=request["media"],
                    output_type=AudioAnalysisResult,
                    model=audio_client.model,
                )
                trace.record_response(parsed.model_dump_json())
                toolkit._record_client_usage(audio_client, trace)
                trace.record(media_kind="audio", artifact_path=request["artifact_path"])

            toolkit.record_audio_analysis(
                start_s=request["start_s"],
                end_s=request["end_s"],
                prompt=full_prompt,
                artifact_path=request["artifact_path"],
                result=parsed.model_dump(),
                source="analyze_audio",
                model=model_label,
            )
            return parsed.model_dump()

        return Tool(
            name="analyze_audio",
            description=(
                "Transcribe and analyze one explicit, non-zero audio slice. "
                "Returns transcript, audio summary, and salient audio events."
            ),
            fn=_analyze_audio,
            parameters={
                "start_s": ToolParam(name="start_s", type_hint="float", description="Absolute start time in seconds."),
                "end_s": ToolParam(name="end_s", type_hint="float", description="Absolute end time in seconds."),
                "prompt": ToolParam(
                    name="prompt",
                    type_hint="str | None",
                    default=None,
                    description="Optional extra audio-analysis instruction.",
                ),
            },
            return_type="dict[str, Any]",
        )

    def _media_trace_context(
        self,
        *,
        kind: str,
        model: str,
        prompt: str,
        start_s: float,
        end_s: float,
        source: str,
        trace_depth: int | None = None,
    ) -> Any:
        if self._tracer is None:
            return nullcontext(_NullTrace())
        effective_depth = self._trace_depth if trace_depth is None else trace_depth
        if kind == "frame":
            return self._tracer.frame_inspection(
                model=model,
                prompt=prompt,
                start_s=start_s,
                end_s=end_s,
                source=source,
                depth=effective_depth,
            )
        if kind == "audio":
            return self._tracer.audio_analysis(
                model=model,
                prompt=prompt,
                start_s=start_s,
                end_s=end_s,
                source=source,
                depth=effective_depth,
            )
        return self._tracer.video_inspection(
            model=model,
            prompt=prompt,
            start_s=start_s,
            end_s=end_s,
            source=source,
            depth=effective_depth,
        )

    def _record_client_usage(self, client: Any, trace: Any) -> None:
        usage = getattr(client, "last_usage", None)
        if usage and self._budget is not None:
            cost = getattr(client, "last_cost_usd", None) or 0.0
            model_name = _model_label(getattr(client, "model", None) or getattr(client, "vision_model", None) or "media")
            self._budget.record(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cost_usd=cost,
                model=model_name,
            )
            trace.record(
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=cost,
            )

        metadata = getattr(client, "last_call_metadata", None)
        if metadata is not None:
            trace.record(
                model_used=metadata.model_used,
                provider=metadata.provider,
                duration_seconds=metadata.duration_seconds,
                fallback_used=metadata.fallback_used,
                cost_usd=metadata.cost_usd,
            )

    def _parse_active_span(self, value: Any) -> tuple[float, float] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            start_s = value.get("start_s")
            end_s = value.get("end_s")
        elif isinstance(value, (list, tuple)) and len(value) == 2:
            start_s, end_s = value
        else:
            return None

        if start_s is None or end_s is None:
            return None
        return (float(start_s), float(end_s))

    def _validate_span(
        self,
        *,
        start_s: float,
        end_s: float,
        allow_zero: bool,
    ) -> tuple[float, float]:
        if self._video_path is None or self._duration_s is None:
            raise ValueError("No video loaded")

        normalized_start = max(0.0, float(start_s))
        normalized_end = max(0.0, float(end_s))

        if normalized_end < normalized_start:
            raise ValueError("end_s must be greater than or equal to start_s")

        normalized_start = min(normalized_start, self._duration_s)
        normalized_end = min(normalized_end, self._duration_s)

        if normalized_start == normalized_end and not allow_zero:
            raise ValueError("Zero-length spans are only valid for single-frame video inspection")

        if self._active_span is not None:
            active_start, active_end = self._active_span
            if normalized_start < active_start or normalized_end > active_end:
                raise ValueError(
                    f"Requested span {normalized_start:.1f}s-{normalized_end:.1f}s is outside the active child span "
                    f"{active_start:.1f}s-{active_end:.1f}s"
                )

        if normalized_end > normalized_start:
            span_s = normalized_end - normalized_start
            max_span_s = self._max_allowed_span_s()
            if span_s - max_span_s > 1e-6:
                raise ValueError(
                    f"Span {span_s:.1f}s exceeds the hard {max_span_s:.1f}s limit. Split it into smaller slices to avoid context rot."
                )

        return normalized_start, normalized_end

    def _max_allowed_span_s(self) -> float:
        if self._duration_s is None:
            return self.max_span_s
        if self._duration_s <= self.max_span_s:
            return self._duration_s
        return self.max_span_s

    def _make_video_media_item(self, start_s: float, end_s: float) -> dict[str, Any]:
        if self._video_path is None:
            raise ValueError("No video loaded")

        return {
            "path": self._video_path,
            "media_type": "video/mp4",
            "vendor_metadata": {
                "start_offset": f"{start_s:.3f}s",
                "end_offset": f"{end_s:.3f}s",
            },
        }

    def _ensure_frame(self, at_s: float) -> str:
        if self._workspace is None or self._video_path is None:
            raise ValueError("Workspace not initialized")
        frame_id = self._frame_id(at_s)
        cached = self._frame_cache.get(frame_id)
        if cached and Path(cached).exists():
            return cached

        output_path = self._workspace.frame_path(frame_id)
        frame_path = _extract_frame_impl(
            video_path=self._video_path,
            at_s=at_s,
            output_path=str(output_path),
        )
        self._frame_cache[frame_id] = frame_path
        return frame_path

    def _span_id(self, start_s: float, end_s: float) -> str:
        return f"{int(round(start_s * 1000)):010d}_{int(round(end_s * 1000)):010d}"

    def _frame_id(self, at_s: float) -> str:
        return f"frame_{int(round(at_s * 1000)):010d}"
